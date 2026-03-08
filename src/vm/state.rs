use super::*;
use log::debug;

impl<'a> VM<'a> {
    pub fn new(source_ref: SourceRef<'a>, is: InstructionSet) -> Self {
        let global_names = is.globals.get_all_names();
        let mut hotspot_detector = HotSpotDetector::new();

        for loop_meta in &is.loops {
            hotspot_detector.register_loop(
                loop_meta.header_ip,
                loop_meta.back_edge_ip,
                loop_meta.exit_ip,
            );
        }

        for func_meta in &is.functions {
            hotspot_detector.register_function(&func_meta.name, func_meta.start_ip);
        }

        let main_frame = CallFrame {
            return_ip: 0,
            frame_pointer: 0,
            stack_pointer: 0,
            instructions: Rc::new(is),
            function_name: "<main>".to_string(),
            return_override: None,
            module_binding: None,
            awaiting_task: None,
        };

        let (io_wakeup_tx, io_wakeup_rx) = mpsc::channel();

        Self {
            stack: Vec::new(),
            locals: Vec::new(),
            call_stack: vec![main_frame],
            exception_handlers: Vec::new(),
            ip: 0,
            gc_poll_counter: 0,
            globals: vec![Value::Void; global_names.len()],
            global_names,
            async_task_queue: VecDeque::new(),
            suspended_main: None,
            suspended_tasks: FxHashMap::default(),
            task_waiters: FxHashMap::default(),
            source_ref,
            io_wakeup_tx,
            io_wakeup_rx,
            user_channels: Vec::new(),
            debugger: None,
            debug_mode: false,
            hotspot_detector,
            type_profile: TypeProfile::new(),
            profiling_enabled: false,
            #[cfg(feature = "jit")]
            jit_compiler: crate::jit::JitCompiler::new().ok(),
            #[cfg(feature = "jit")]
            jit_enabled: false,
        }
    }

    /// Enable the debugger
    pub fn enable_debugger(&mut self) {
        self.debug_mode = true;
        self.debugger = Some(debugger::Debugger::new());
    }

    /// Create a VM with profiling disabled (for benchmarking baseline)
    pub fn new_without_profiling(source_ref: SourceRef<'a>, is: InstructionSet) -> Self {
        let mut vm = Self::new(source_ref, is);
        vm.profiling_enabled = false;
        vm
    }

    /// Enable or disable JIT profiling
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling_enabled = enabled;
    }

    /// Enable or disable JIT compilation and execution
    #[cfg(feature = "jit")]
    pub fn set_jit_enabled(&mut self, enabled: bool) {
        self.jit_enabled = enabled;
    }

    /// Get JIT compilation statistics
    #[cfg(feature = "jit")]
    pub fn jit_stats(&self) -> Option<crate::jit::JitStats> {
        self.jit_compiler.as_ref().map(|jit| jit.stats())
    }

    /// Get hot-spot detection statistics
    pub fn hotspot_stats(&self) -> crate::jit::hotspot::HotSpotStats {
        self.hotspot_detector.stats()
    }

    /// Get the type profile for analysis
    pub fn type_profile(&self) -> &TypeProfile {
        &self.type_profile
    }

    #[inline(always)]
    pub(super) fn current_frame(&self) -> &CallFrame {
        unsafe { self.call_stack.last().unwrap_unchecked() }
    }

    #[inline(always)]
    pub(super) fn current_frame_index(&self) -> usize {
        self.call_stack.len() - 1
    }

    #[inline(always)]
    pub(super) fn frame_pointer(&self) -> usize {
        self.current_frame().frame_pointer
    }

    #[inline(always)]
    pub(super) fn function_name(&self) -> &str {
        let name = &self.current_frame().function_name;
        if name.is_empty() { "<fn>" } else { name }
    }

    pub(super) fn take_context(&mut self) -> ExecutionContext {
        ExecutionContext {
            stack: std::mem::take(&mut self.stack),
            locals: std::mem::take(&mut self.locals),
            call_stack: std::mem::take(&mut self.call_stack),
            exception_handlers: std::mem::take(&mut self.exception_handlers),
            ip: self.ip,
        }
    }

    pub(super) fn restore_context(&mut self, context: ExecutionContext) {
        self.stack = context.stack;
        self.locals = context.locals;
        self.call_stack = context.call_stack;
        self.exception_handlers = context.exception_handlers;
        self.ip = context.ip;
    }

    #[inline]
    pub(crate) fn get_heap(&self) -> &crate::arenas::ValueHolder {
        unsafe { &*crate::arenas::get_arena_ptr() }
    }

    #[inline]
    pub(crate) fn get_heap_mut(&mut self) -> &mut crate::arenas::ValueHolder {
        unsafe { &mut *crate::arenas::get_arena_ptr() }
    }

    #[inline(always)]
    pub(crate) fn source_ref(&self) -> SourceRef<'a> {
        self.source_ref
    }

    /// Collect all root values that the GC needs to trace from
    pub(crate) fn collect_roots(&self) -> Vec<Value> {
        fn extend_context_roots(
            roots: &mut Vec<Value>,
            stack: &[Value],
            locals: &[Value],
            call_stack: &[CallFrame],
        ) {
            roots.extend(stack.iter().copied());
            roots.extend(locals.iter().copied());
            for frame in call_stack {
                roots.extend(frame.instructions.constants.iter().copied());
                if let Some(binding) = &frame.module_binding {
                    roots.push(Value::Module(binding.module_key));
                }
                if let Some(task_key) = frame.awaiting_task {
                    roots.push(Value::Task(task_key));
                }
            }
        }

        let mut roots = Vec::with_capacity(
            self.stack.len()
                + self.locals.len()
                + self.globals.len()
                + self.async_task_queue.len()
                + self.suspended_tasks.len() * 16
                + 128,
        );

        extend_context_roots(&mut roots, &self.stack, &self.locals, &self.call_stack);
        roots.extend(self.globals.iter().copied());

        for &task_key in &self.async_task_queue {
            roots.push(Value::Task(task_key));
        }

        if let Some(suspended) = &self.suspended_main {
            extend_context_roots(
                &mut roots,
                &suspended.context.stack,
                &suspended.context.locals,
                &suspended.context.call_stack,
            );
            roots.push(Value::Task(suspended.waiting_on));
        }

        for (&task_key, suspended) in &self.suspended_tasks {
            roots.push(Value::Task(task_key));
            roots.push(Value::Task(suspended.waiting_on));
            extend_context_roots(
                &mut roots,
                &suspended.context.stack,
                &suspended.context.locals,
                &suspended.context.call_stack,
            );
        }

        roots.extend(crate::program::cached_module_roots());
        roots
    }

    /// Run garbage collection if needed
    pub(super) fn maybe_collect_garbage(&mut self) {
        if self.get_heap().should_collect() {
            let roots = self.collect_roots();
            let freed = self.get_heap_mut().collect_garbage(&roots);
            if freed > 0 {
                debug!("GC: Freed {} objects", freed);
            }
        }
    }

    /// Stringify a value using the global heap
    pub(super) fn stringify_value(&self, value: Value) -> WalrusResult<String> {
        self.get_heap().stringify(value)
    }

    /// Call a native stdlib function
    pub(super) fn call_native(
        &mut self,
        native_fn: crate::function::NativeFunction,
        args: Vec<Value>,
        span: Span,
    ) -> WalrusResult<Value> {
        crate::native_registry::dispatch_native(self, native_fn, args, span)
    }

    /// Helper to extract string from Value
    pub(crate) fn value_to_string(&self, value: Value, span: Span) -> WalrusResult<String> {
        match value {
            Value::String(key) => Ok(self.get_heap().get_string(key)?.to_string()),
            _ => Err(WalrusError::TypeMismatch {
                expected: "string".to_string(),
                found: value.get_type().to_string(),
                span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            }),
        }
    }

    /// Helper to extract int from Value
    pub(crate) fn value_to_int(&self, value: Value, span: Span) -> WalrusResult<i64> {
        match value {
            Value::Int(n) => Ok(n),
            _ => Err(WalrusError::TypeMismatch {
                expected: "int".to_string(),
                found: value.get_type().to_string(),
                span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            }),
        }
    }

    /// Helper to extract numeric values (int or float) as f64
    pub(crate) fn value_to_number(&self, value: Value, span: Span) -> WalrusResult<f64> {
        match value {
            Value::Int(n) => Ok(n as f64),
            Value::Float(FloatOrd(f)) => Ok(f),
            _ => Err(WalrusError::TypeMismatch {
                expected: "number".to_string(),
                found: value.get_type().to_string(),
                span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            }),
        }
    }

    /// Normalize an index (supports negative indices) against a character length.
    pub(super) fn normalize_index(index: i64, len: usize) -> Option<usize> {
        let len = len as i64;
        let normalized = if index < 0 { index + len } else { index };
        if normalized < 0 || normalized >= len {
            None
        } else {
            Some(normalized as usize)
        }
    }

    /// Convert a character index to a byte offset.
    /// Returns `s.len()` when `char_index` is exactly one-past-the-end.
    pub(super) fn char_to_byte_offset(s: &str, char_index: usize) -> Option<usize> {
        if char_index == 0 {
            return Some(0);
        }
        if char_index == s.chars().count() {
            return Some(s.len());
        }
        s.char_indices().nth(char_index).map(|(offset, _)| offset)
    }

    /// Remove handlers that are no longer reachable from the current execution point.
    pub(super) fn prune_exception_handlers(&mut self) {
        if self.exception_handlers.is_empty() {
            return;
        }

        let current_frame = self.current_frame_index();
        while let Some(handler) = self.exception_handlers.last().copied() {
            let stale_frame = handler.frame_index > current_frame;
            let out_of_range_in_frame = handler.frame_index == current_frame
                && (self.ip < handler.start_ip || self.ip >= handler.end_ip);

            if stale_frame || out_of_range_in_frame {
                self.exception_handlers.pop();
            } else {
                break;
            }
        }
    }

    /// Clear all handlers that belong to `frame_index` or deeper.
    pub(super) fn clear_exception_handlers_from_frame(&mut self, frame_index: usize) {
        while let Some(handler) = self.exception_handlers.last().copied() {
            if handler.frame_index >= frame_index {
                self.exception_handlers.pop();
            } else {
                break;
            }
        }
    }

    /// Raise a thrown value, transferring control to the nearest active catch handler.
    pub(super) fn throw_value(&mut self, value: Value, span: Span) -> WalrusResult<()> {
        while let Some(handler) = self.exception_handlers.last().copied() {
            if handler.frame_index > self.current_frame_index() {
                self.exception_handlers.pop();
                continue;
            }

            self.exception_handlers.pop();

            while self.current_frame_index() > handler.frame_index {
                let frame = self
                    .call_stack
                    .pop()
                    .expect("Call stack should never be empty while unwinding");
                self.locals.truncate(frame.frame_pointer);
                self.stack.truncate(frame.stack_pointer);
            }

            self.locals.truncate(handler.locals_len);
            self.stack.truncate(handler.stack_len);
            self.ip = handler.catch_ip;
            self.push(value);
            return Ok(());
        }

        let message = self.stringify_value(value)?;
        Err(WalrusError::ThrownValue {
            message,
            span,
            src: self.source_ref.source().into(),
            filename: self.source_ref.filename().into(),
        })
    }

    #[inline(always)]
    pub(super) fn push(&mut self, value: Value) {
        self.stack.push(value);
    }

    #[inline]
    pub(super) fn pop(&mut self, op: Opcode, span: Span) -> WalrusResult<Value> {
        self.stack.pop().ok_or_else(|| WalrusError::StackUnderflow {
            op,
            span,
            src: self.source_ref.source().to_string(),
            filename: self.source_ref.filename().to_string(),
        })
    }

    /// Fast path pop - only use when stack is guaranteed to have values
    #[inline(always)]
    pub(super) fn pop_unchecked(&mut self) -> Value {
        unsafe { self.stack.pop().unwrap_unchecked() }
    }

    pub(super) fn pop_n(&mut self, n: usize, op: Opcode, span: Span) -> WalrusResult<Vec<Value>> {
        if self.stack.len() < n {
            return Err(WalrusError::StackUnderflow {
                op,
                span,
                src: self.source_ref.source().to_string(),
                filename: self.source_ref.filename().to_string(),
            });
        }

        let split_at = self.stack.len() - n;
        Ok(self.stack.split_off(split_at))
    }
}
