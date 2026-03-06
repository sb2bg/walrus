use std::cell::RefCell;
use std::collections::VecDeque;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use float_ord::FloatOrd;
use log::{debug, log_enabled};
use rustc_hash::{FxHashMap, FxHashSet};

use instruction_set::InstructionSet;

use crate::WalrusResult;
use crate::arenas::{FuncKey, HeapValue};
use crate::error::WalrusError;
use crate::function::{VmModuleBinding, WalrusFunction};
use crate::iter::ValueIterator;
use crate::jit::{HotSpotDetector, TypeProfile, WalrusType};
use crate::range::RangeValue;
use crate::source_ref::SourceRef;
use crate::span::Span;
use crate::value::{AsyncTask, IoChannel, IoResult, Value};
use crate::vm::opcode::Opcode;

pub mod compiler;
pub mod debugger;
pub mod handlers;
pub mod instruction_set;
pub mod methods;
pub mod opcode;
pub mod optimize;
mod symbol_table;

/// Represents a single call frame on the call stack.
///
/// Each function invocation creates a new CallFrame which tracks:
/// - Where to return to after the function completes
/// - Where this function's local variables start in the shared locals vector
/// - Where the operand stack was at call time (for cleanup on return)
/// - The function's bytecode (via Rc to avoid cloning)
/// - The function name for debugging/stack traces
#[derive(Debug, Clone)]
struct CallFrame {
    /// Instruction pointer to return to after this frame completes
    return_ip: usize,
    /// Index into the shared `locals` vector where this frame's variables start
    frame_pointer: usize,
    /// Index into the operand stack where this frame started (for cleanup on return)
    stack_pointer: usize,
    /// Reference to the function's InstructionSet (shared via Rc to avoid cloning)
    instructions: Rc<InstructionSet>,
    /// Function name for debugging and stack traces
    function_name: String,
    /// Optional value to return instead of the callee's explicit return value.
    /// Used by struct constructors so `init` can return the new instance.
    return_override: Option<Value>,
    /// Optional module binding context for exported module VM functions.
    module_binding: Option<Rc<VmModuleBinding>>,
    /// The task that this frame is currently resolving via `await`, if any.
    awaiting_task: Option<crate::arenas::TaskKey>,
}

#[derive(Debug, Clone, Copy)]
struct ExceptionHandler {
    /// Frame index where this handler was installed.
    frame_index: usize,
    /// Inclusive start IP for the protected lexical region.
    start_ip: usize,
    /// Exclusive end IP for the protected lexical region (catch block start).
    end_ip: usize,
    /// IP of the catch block entry.
    catch_ip: usize,
    /// Operand stack length to restore before entering catch.
    stack_len: usize,
    /// Locals length to restore before entering catch.
    locals_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaskResolution {
    Pending,
    Ready(Value),
    Failed(Value),
    Cancelled,
}

#[derive(Debug, Default, Clone)]
struct ExecutionContext {
    stack: Vec<Value>,
    locals: Vec<Value>,
    call_stack: Vec<CallFrame>,
    exception_handlers: Vec<ExceptionHandler>,
    ip: usize,
}

#[derive(Debug, Clone)]
struct SuspendedExecution {
    context: ExecutionContext,
    waiting_on: crate::arenas::TaskKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunSignal {
    Returned(Value),
    Suspended(crate::arenas::TaskKey),
}

/// The Walrus Virtual Machine executes compiled bytecode.
///
/// # Architecture
///
/// ## Stack-Based Execution
/// The VM uses a value stack for expression evaluation. Operations pop operands
/// from the stack and push results back.
///
/// ## Local Variables
/// Local variables are stored in a shared `locals` vector, indexed by compile-time
/// assigned indices plus the current frame pointer. Each call frame has a frame_pointer
/// that indicates where its local variables begin in this shared vector.
///
/// ## Call Frames
/// Function calls use a call frame stack instead of creating child VMs. Each frame
/// tracks:
/// - `return_ip`: where to resume execution after the function returns
/// - `frame_pointer`: where this frame's locals start in the shared locals vector
/// - `instructions`: the function's bytecode (shared via Rc, avoiding clones)
/// - `function_name`: for debugging and stack traces
///
/// This is more memory-efficient than creating child VMs and supports deep recursion.
///
/// ## Global Variables
/// Globals are stored in a shared vector across all call frames.
///
/// ## Memory Model
/// Heap-allocated values (strings, lists, dicts, functions) are stored in a global
/// arena (`ARENA`) and referenced by keys. The VM never directly holds heap data,
/// only keys that index into the arena.
///
/// ## JIT Compilation (Phase 2)
/// A user-facing async channel for task-to-task communication.
/// Single-threaded: uses Rc<RefCell<...>> since the VM is not Send.
struct UserChannel {
    buffer: Rc<RefCell<VecDeque<Value>>>,
    closed: Rc<RefCell<bool>>,
}

/// The VM tracks type information and execution counts at key points:
/// - Loop headers (for hot loop detection)
/// - Function calls (for hot function detection)
/// - Arithmetic operations (for type specialization)
/// When a loop becomes "hot" (>1000 iterations), it is compiled to native code
/// using Cranelift and executed directly, bypassing the interpreter.
pub struct VM<'a> {
    stack: Vec<Value>,          // Operand stack for expression evaluation
    locals: Vec<Value>,         // Shared across all call frames
    call_stack: Vec<CallFrame>, // Stack of call frames
    exception_handlers: Vec<ExceptionHandler>,
    ip: usize,            // Current instruction pointer
    gc_poll_counter: u32, // Throttle GC checks to avoid per-instruction overhead
    globals: Vec<Value>,
    global_names: Vec<String>,
    async_task_queue: VecDeque<crate::arenas::TaskKey>,
    suspended_main: Option<SuspendedExecution>,
    suspended_tasks: FxHashMap<crate::arenas::TaskKey, SuspendedExecution>,
    task_waiters: FxHashMap<crate::arenas::TaskKey, Vec<crate::arenas::TaskKey>>,
    source_ref: SourceRef<'a>,
    // I/O wakeup channel: worker threads send () to wake the event loop
    io_wakeup_tx: mpsc::Sender<()>,
    io_wakeup_rx: mpsc::Receiver<()>,
    // User-facing async channels for task-to-task communication
    user_channels: Vec<UserChannel>,
    // Debugger state
    debugger: Option<debugger::Debugger>,
    debug_mode: bool,
    // JIT profiling
    hotspot_detector: HotSpotDetector,
    type_profile: TypeProfile,
    profiling_enabled: bool,
    // JIT compiler (Phase 2)
    #[cfg(feature = "jit")]
    jit_compiler: Option<crate::jit::JitCompiler>,
    #[cfg(feature = "jit")]
    jit_enabled: bool,
}

impl<'a> VM<'a> {
    pub fn new(source_ref: SourceRef<'a>, is: InstructionSet) -> Self {
        let global_names = is.globals.get_all_names();
        // Initialize hot-spot detector with loop and function metadata from the bytecode
        let mut hotspot_detector = HotSpotDetector::new();

        // Register all loops detected during compilation
        for loop_meta in &is.loops {
            hotspot_detector.register_loop(
                loop_meta.header_ip,
                loop_meta.back_edge_ip,
                loop_meta.exit_ip,
            );
        }

        // Register all functions detected during compilation
        for func_meta in &is.functions {
            hotspot_detector.register_function(&func_meta.name, func_meta.start_ip);
        }

        // Create the initial call frame for the main program
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
            // Pre-size globals to declared symbol count so sparse/forward writes are safe.
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
            // Profiling is opt-in (enabled by Program when requested)
            hotspot_detector,
            type_profile: TypeProfile::new(),
            profiling_enabled: false,
            // JIT compiler (Phase 2)
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

    /// Get the current call frame (the one at the top of the call stack)
    #[inline(always)]
    fn current_frame(&self) -> &CallFrame {
        // SAFETY: call_stack is never empty during VM execution (always has at least main frame)
        unsafe { self.call_stack.last().unwrap_unchecked() }
    }

    #[inline(always)]
    fn current_frame_index(&self) -> usize {
        self.call_stack.len() - 1
    }

    /// Get the current frame pointer (start of current frame's locals)
    #[inline(always)]
    fn frame_pointer(&self) -> usize {
        self.current_frame().frame_pointer
    }

    /// Get the current function name
    #[inline(always)]
    fn function_name(&self) -> &str {
        let name = &self.current_frame().function_name;
        if name.is_empty() { "<fn>" } else { name }
    }

    fn take_context(&mut self) -> ExecutionContext {
        ExecutionContext {
            stack: std::mem::take(&mut self.stack),
            locals: std::mem::take(&mut self.locals),
            call_stack: std::mem::take(&mut self.call_stack),
            exception_handlers: std::mem::take(&mut self.exception_handlers),
            ip: self.ip,
        }
    }

    fn restore_context(&mut self, context: ExecutionContext) {
        self.stack = context.stack;
        self.locals = context.locals;
        self.call_stack = context.call_stack;
        self.exception_handlers = context.exception_handlers;
        self.ip = context.ip;
    }

    /// Helper to access heap - uses thread-local ARENA
    ///
    /// # Safety
    /// Uses unsafe get_arena_ptr() for performance. This is safe because:
    /// - The VM is single-threaded
    /// - We only access the arena within VM methods (no escaping references)
    /// - No concurrent borrows occur within VM execution
    #[inline]
    pub(crate) fn get_heap(&self) -> &crate::arenas::ValueHolder {
        unsafe { &*crate::arenas::get_arena_ptr() }
    }

    /// Helper to access heap mutably - uses thread-local ARENA
    ///
    /// # Safety
    /// See get_heap() for safety rationale.
    #[inline]
    pub(crate) fn get_heap_mut(&mut self) -> &mut crate::arenas::ValueHolder {
        unsafe { &mut *crate::arenas::get_arena_ptr() }
    }

    #[inline(always)]
    pub(crate) fn source_ref(&self) -> SourceRef<'a> {
        self.source_ref
    }

    #[inline(always)]
    fn current_module_binding(&self) -> Option<Rc<VmModuleBinding>> {
        self.call_stack
            .last()
            .and_then(|frame| frame.module_binding.clone())
    }

    fn undefined_global_error(
        &self,
        index: usize,
        span: Span,
        binding: Option<&VmModuleBinding>,
    ) -> WalrusError {
        let name = binding
            .and_then(|ctx| ctx.global_names.get(index))
            .or_else(|| self.global_names.get(index))
            .cloned()
            .unwrap_or_else(|| format!("<global[{index}]>"));

        let (src, filename) = if let Some(ctx) = binding {
            (ctx.source.to_string(), ctx.filename.to_string())
        } else {
            (
                self.source_ref.source().to_string(),
                self.source_ref.filename().to_string(),
            )
        };

        WalrusError::UndefinedVariable {
            name,
            span,
            src,
            filename,
        }
    }

    fn load_global_value(&mut self, index: usize, span: Span) -> WalrusResult<Value> {
        if let Some(binding) = self.current_module_binding() {
            let Some(name) = binding.global_names.get(index) else {
                return Err(self.undefined_global_error(index, span, Some(binding.as_ref())));
            };

            let key = Value::String(self.get_heap_mut().push_ident(name));
            let module = self.get_heap().get_module(binding.module_key)?;
            if let Some(value) = module.get(&key).copied() {
                return Ok(value);
            }

            Ok(binding
                .global_values
                .get(index)
                .copied()
                .unwrap_or(Value::Void))
        } else {
            self.globals
                .get(index)
                .copied()
                .ok_or_else(|| self.undefined_global_error(index, span, None))
        }
    }

    fn store_global_value(&mut self, index: usize, value: Value, span: Span) -> WalrusResult<()> {
        if let Some(binding) = self.current_module_binding() {
            let Some(name) = binding.global_names.get(index) else {
                return Err(self.undefined_global_error(index, span, Some(binding.as_ref())));
            };

            let key = Value::String(self.get_heap_mut().push_ident(name));
            self.get_heap_mut()
                .get_mut_module(binding.module_key)?
                .insert(key, value);
            Ok(())
        } else {
            if index >= self.globals.len() {
                self.globals.resize(index + 1, Value::Void);
            }
            self.globals[index] = value;
            Ok(())
        }
    }

    fn bind_exported_value_to_module(
        &mut self,
        value: Value,
        binding: &Rc<VmModuleBinding>,
    ) -> WalrusResult<Value> {
        match value {
            Value::Function(func_key) => {
                let function = self.get_heap().get_function(func_key)?.clone();
                match function {
                    WalrusFunction::Vm(mut vm_func) => {
                        vm_func.module_binding = Some(binding.clone());
                        Ok(self
                            .get_heap_mut()
                            .push(HeapValue::Function(WalrusFunction::Vm(vm_func))))
                    }
                    _ => Ok(value),
                }
            }
            Value::StructDef(struct_key) => {
                let original = self.get_heap().get_struct_def(struct_key)?.clone();
                let mut rebound =
                    crate::structs::StructDefinition::new(original.name().to_string());

                for (method_name, method_fn) in original.methods() {
                    let bound_method = match method_fn.clone() {
                        WalrusFunction::Vm(mut vm_func) => {
                            vm_func.module_binding = Some(binding.clone());
                            WalrusFunction::Vm(vm_func)
                        }
                        other => other,
                    };
                    rebound.add_method(method_name.clone(), bound_method);
                }

                Ok(self.get_heap_mut().push(HeapValue::StructDef(rebound)))
            }
            _ => Ok(value),
        }
    }

    pub fn export_globals_as_module(&mut self) -> WalrusResult<Value> {
        let global_names = self.global_names.clone();
        let global_values = self.globals.clone();
        let mut exports = FxHashMap::default();

        for (index, name) in global_names.iter().enumerate() {
            if name == "_" {
                continue;
            }

            let value = match self.globals.get(index).copied() {
                Some(value) => value,
                None => continue,
            };

            let key = self.get_heap_mut().push(HeapValue::String(name));
            exports.insert(key, value);
        }

        let module_key = match self.get_heap_mut().push(HeapValue::Module(exports)) {
            Value::Module(key) => key,
            _ => unreachable!("module export must allocate a module"),
        };

        let binding = Rc::new(VmModuleBinding {
            module_key,
            global_names: Rc::new(global_names),
            global_values: Rc::new(global_values),
            source: Rc::new(self.source_ref.source().to_string()),
            filename: Rc::new(self.source_ref.filename().to_string()),
        });

        // Rebind exported values that contain VM functions so global accesses
        // resolve against this module, not the importing VM's global vector.
        let names = binding.global_names.clone();
        for name in names.iter() {
            if name == "_" {
                continue;
            }

            let entry_value = {
                let key = Value::String(self.get_heap_mut().push_ident(name));
                let module = self.get_heap().get_module(module_key)?;
                module.get(&key).copied()
            };

            let Some(entry_value) = entry_value else {
                continue;
            };

            let bound_value = self.bind_exported_value_to_module(entry_value, &binding)?;

            let key = Value::String(self.get_heap_mut().push_ident(name));
            self.get_heap_mut()
                .get_mut_module(module_key)?
                .insert(key, bound_value);
        }

        Ok(Value::Module(module_key))
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
    fn maybe_collect_garbage(&mut self) {
        if self.get_heap().should_collect() {
            let roots = self.collect_roots();
            let freed = self.get_heap_mut().collect_garbage(&roots);
            if freed > 0 {
                debug!("GC: Freed {} objects", freed);
            }
        }
    }

    /// Stringify a value using the global heap
    fn stringify_value(&self, value: Value) -> WalrusResult<String> {
        self.get_heap().stringify(value)
    }

    /// Format the current call stack as a human-readable stack trace
    /// Truncates the middle if there are too many frames (like Python does)
    fn format_stack_trace(&self) -> String {
        if self.call_stack.len() <= 1 {
            return String::new();
        }

        const MAX_FRAMES_TOP: usize = 5; // Show first N frames (oldest)
        const MAX_FRAMES_BOTTOM: usize = 5; // Show last N frames (most recent)

        let mut trace = String::from("\nStack trace (most recent call last):\n");
        let len = self.call_stack.len();

        if len <= MAX_FRAMES_TOP + MAX_FRAMES_BOTTOM {
            // Show all frames
            for (i, frame) in self.call_stack.iter().enumerate() {
                let name = if frame.function_name.is_empty() {
                    "<fn>"
                } else {
                    frame.function_name.as_str()
                };
                trace.push_str(&format!("  {}: {}\n", i, name));
            }
        } else {
            // Show first N frames
            for (i, frame) in self.call_stack.iter().take(MAX_FRAMES_TOP).enumerate() {
                let name = if frame.function_name.is_empty() {
                    "<fn>"
                } else {
                    frame.function_name.as_str()
                };
                trace.push_str(&format!("  {}: {}\n", i, name));
            }

            // Show truncation message
            let hidden = len - MAX_FRAMES_TOP - MAX_FRAMES_BOTTOM;
            trace.push_str(&format!("  ... {} more frames ...\n", hidden));

            // Show last N frames
            for (i, frame) in self
                .call_stack
                .iter()
                .skip(len - MAX_FRAMES_BOTTOM)
                .enumerate()
            {
                let actual_index = len - MAX_FRAMES_BOTTOM + i;
                let name = if frame.function_name.is_empty() {
                    "<fn>"
                } else {
                    frame.function_name.as_str()
                };
                trace.push_str(&format!("  {}: {}\n", actual_index, name));
            }
        }
        trace
    }

    /// Call a native stdlib function
    fn call_native(
        &mut self,
        native_fn: crate::function::NativeFunction,
        args: Vec<Value>,
        span: Span,
    ) -> WalrusResult<Value> {
        crate::native_registry::dispatch_native(self, native_fn, args, span)
    }

    fn create_task_for_function_key(&mut self, function_key: FuncKey, args: Vec<Value>) -> Value {
        let task = self
            .get_heap_mut()
            .push(HeapValue::Task(AsyncTask::Pending {
                function: function_key,
                args,
            }));
        if let Value::Task(task_key) = task {
            self.enqueue_task(task_key);
        }
        task
    }

    fn create_task_for_cloned_function(
        &mut self,
        function: WalrusFunction,
        args: Vec<Value>,
    ) -> Value {
        let func_value = self.get_heap_mut().push(HeapValue::Function(function));
        let Value::Function(function_key) = func_value else {
            unreachable!("HeapValue::Function must allocate a function key")
        };
        self.create_task_for_function_key(function_key, args)
    }

    fn create_non_runnable_task(&mut self, task: AsyncTask) -> Value {
        self.get_heap_mut().push(HeapValue::Task(task))
    }

    pub(crate) fn spawn_task_from_callable(
        &mut self,
        callable: Value,
        args: Vec<Value>,
        span: Span,
    ) -> WalrusResult<Value> {
        match callable {
            Value::Task(task_key) => {
                if !args.is_empty() {
                    return Err(WalrusError::InvalidArgCount {
                        name: "spawn".to_string(),
                        expected: 0,
                        got: args.len(),
                        span,
                        src: self.source_ref.source().into(),
                        filename: self.source_ref.filename().into(),
                    });
                }
                Ok(Value::Task(task_key))
            }
            Value::Function(function_key) => {
                let function = self.get_heap().get_function(function_key)?;
                let expected = match function {
                    WalrusFunction::Vm(func) => func.arity,
                    WalrusFunction::Rust(func) => func.args,
                    WalrusFunction::Native(func) => func.arity(),
                    WalrusFunction::TreeWalk(_) => {
                        return Err(WalrusError::NodeFunctionNotSupportedInVm {
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        });
                    }
                };
                if args.len() != expected {
                    return Err(WalrusError::InvalidArgCount {
                        name: function.to_string(),
                        expected,
                        got: args.len(),
                        span,
                        src: self.source_ref.source().into(),
                        filename: self.source_ref.filename().into(),
                    });
                }
                Ok(self.create_task_for_function_key(function_key, args))
            }
            other => Err(WalrusError::TypeMismatch {
                expected: "function or task".to_string(),
                found: other.get_type().to_string(),
                span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            }),
        }
    }

    pub(crate) fn create_sleep_task(&mut self, delay_ms: u64) -> Value {
        let wake_at = Instant::now() + Duration::from_millis(delay_ms);
        self.create_non_runnable_task(AsyncTask::Sleep { wake_at })
    }

    pub(crate) fn create_timeout_task(
        &mut self,
        task_key: crate::arenas::TaskKey,
        timeout_ms: u64,
    ) -> Value {
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        self.create_non_runnable_task(AsyncTask::Timeout {
            task: task_key,
            deadline,
        })
    }

    /// Spawn a background I/O operation on a worker thread.
    /// Returns a Task value backed by a Channel that resolves when the worker completes.
    pub(crate) fn spawn_io<F>(&mut self, work: F) -> Value
    where
        F: FnOnce() -> Result<IoResult, String> + Send + 'static,
    {
        let (tx, rx) = mpsc::channel();
        let wakeup = self.io_wakeup_tx.clone();
        std::thread::spawn(move || {
            let result = work();
            let _ = tx.send(result);
            let _ = wakeup.send(());
        });
        self.create_non_runnable_task(AsyncTask::Channel(IoChannel::new(rx)))
    }

    /// Convert an IoResult from a worker thread into a Value on the VM thread.
    fn io_result_to_value(&mut self, result: IoResult) -> WalrusResult<Value> {
        match result {
            IoResult::Stream(stream) => {
                let handle = crate::stdlib::store_tcp_stream(stream);
                Ok(Value::Int(handle))
            }
            IoResult::Listener(listener) => {
                let handle = crate::stdlib::store_tcp_listener(listener);
                Ok(Value::Int(handle))
            }
            IoResult::Bytes(bytes) => {
                let text = String::from_utf8(bytes).map_err(|e| WalrusError::GenericError {
                    message: format!("I/O result contains non-UTF8 data: {e}"),
                })?;
                Ok(self.get_heap_mut().push(HeapValue::String(&text)))
            }
            IoResult::ByteCount(n) => Ok(Value::Int(n as i64)),
            IoResult::HttpOutcome(outcome) => self.http_outcome_to_value(outcome),
            IoResult::Void => Ok(Value::Void),
        }
    }

    /// Convert an IoHttpOutcome into a Value (dict or Void).
    fn http_outcome_to_value(
        &mut self,
        outcome: crate::value::IoHttpOutcome,
    ) -> WalrusResult<Value> {
        use crate::value::IoHttpOutcome;

        match outcome {
            IoHttpOutcome::Eof => Ok(Value::Void),
            IoHttpOutcome::BadRequest(message) => {
                let mut dict = FxHashMap::default();
                let ok_key = self.get_heap_mut().push(HeapValue::String("ok"));
                dict.insert(ok_key, Value::Bool(false));
                let err_key = self.get_heap_mut().push(HeapValue::String("error"));
                let err_val = self.get_heap_mut().push(HeapValue::String(&message));
                dict.insert(err_key, err_val);
                Ok(self.get_heap_mut().push(HeapValue::Dict(dict)))
            }
            IoHttpOutcome::Request(req) => {
                let mut headers = FxHashMap::default();
                let mut header_pairs = Vec::with_capacity(req.headers.len());
                for (name, value) in &req.headers {
                    let key = self.get_heap_mut().push(HeapValue::String(name));
                    let val = self.get_heap_mut().push(HeapValue::String(value));
                    headers.insert(key, val);
                    let pair = self.get_heap_mut().push(HeapValue::List(vec![key, val]));
                    header_pairs.push(pair);
                }
                let headers_value = self.get_heap_mut().push(HeapValue::Dict(headers));
                let header_pairs_value = self.get_heap_mut().push(HeapValue::List(header_pairs));

                let query_pairs = crate::stdlib::http_parse_query(&req.query);
                let mut query = FxHashMap::default();
                let mut query_pairs_values = Vec::with_capacity(query_pairs.len());
                for (name, value) in query_pairs {
                    let key = self.get_heap_mut().push(HeapValue::String(&name));
                    let val = self.get_heap_mut().push(HeapValue::String(&value));
                    query.insert(key, val);
                    let pair = self.get_heap_mut().push(HeapValue::List(vec![key, val]));
                    query_pairs_values.push(pair);
                }
                let query_value = self.get_heap_mut().push(HeapValue::Dict(query));
                let query_pairs_value = self
                    .get_heap_mut()
                    .push(HeapValue::List(query_pairs_values));

                let mut dict = FxHashMap::default();
                let ok_key = self.get_heap_mut().push(HeapValue::String("ok"));
                dict.insert(ok_key, Value::Bool(true));
                let method_key = self.get_heap_mut().push(HeapValue::String("method"));
                let method_val = self.get_heap_mut().push(HeapValue::String(&req.method));
                dict.insert(method_key, method_val);
                let target_key = self.get_heap_mut().push(HeapValue::String("target"));
                let target_val = self.get_heap_mut().push(HeapValue::String(&req.target));
                dict.insert(target_key, target_val);
                let path_key = self.get_heap_mut().push(HeapValue::String("path"));
                let path_val = self.get_heap_mut().push(HeapValue::String(&req.path));
                dict.insert(path_key, path_val);
                let query_key = self.get_heap_mut().push(HeapValue::String("query"));
                let query_text_val = self.get_heap_mut().push(HeapValue::String(&req.query));
                dict.insert(query_key, query_text_val);
                let qp_key = self.get_heap_mut().push(HeapValue::String("query_params"));
                dict.insert(qp_key, query_value);
                let qpp_key = self.get_heap_mut().push(HeapValue::String("query_pairs"));
                dict.insert(qpp_key, query_pairs_value);
                let ver_key = self.get_heap_mut().push(HeapValue::String("version"));
                let ver_val = self.get_heap_mut().push(HeapValue::String(&req.version));
                dict.insert(ver_key, ver_val);
                let hdr_key = self.get_heap_mut().push(HeapValue::String("headers"));
                dict.insert(hdr_key, headers_value);
                let hdr_pairs_key = self.get_heap_mut().push(HeapValue::String("header_pairs"));
                dict.insert(hdr_pairs_key, header_pairs_value);
                let body_key = self.get_heap_mut().push(HeapValue::String("body"));
                let body_val = self.get_heap_mut().push(HeapValue::String(&req.body));
                dict.insert(body_key, body_val);
                let cl_key = self
                    .get_heap_mut()
                    .push(HeapValue::String("content_length"));
                dict.insert(cl_key, Value::Int(req.content_length));

                Ok(self.get_heap_mut().push(HeapValue::Dict(dict)))
            }
        }
    }

    /// Check if a task tree has any pending I/O (Channel) tasks.
    fn task_has_pending_io(
        &self,
        task_key: crate::arenas::TaskKey,
        visited: &mut FxHashSet<crate::arenas::TaskKey>,
    ) -> WalrusResult<bool> {
        if !visited.insert(task_key) {
            return Ok(false);
        }
        let task = self.get_heap().get_task(task_key)?;
        match task {
            AsyncTask::Channel(_) => Ok(true),
            AsyncTask::UserRecv { .. } => Ok(false),
            AsyncTask::Timeout { task, .. } => self.task_has_pending_io(*task, visited),
            AsyncTask::Gather { tasks }
            | AsyncTask::Race { tasks }
            | AsyncTask::AllSettled { tasks } => {
                for &child in tasks {
                    if self.task_has_pending_io(child, visited)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            _ => Ok(false),
        }
    }

    pub(crate) fn create_gather_task(&mut self, tasks: Vec<crate::arenas::TaskKey>) -> Value {
        self.create_non_runnable_task(AsyncTask::Gather { tasks })
    }

    pub(crate) fn create_race_task(&mut self, tasks: Vec<crate::arenas::TaskKey>) -> Value {
        self.create_non_runnable_task(AsyncTask::Race { tasks })
    }

    pub(crate) fn create_all_settled_task(&mut self, tasks: Vec<crate::arenas::TaskKey>) -> Value {
        self.create_non_runnable_task(AsyncTask::AllSettled { tasks })
    }

    pub(crate) fn create_user_channel(&mut self) -> (Value, Value) {
        let id = self.user_channels.len();
        self.user_channels.push(UserChannel {
            buffer: Rc::new(RefCell::new(VecDeque::new())),
            closed: Rc::new(RefCell::new(false)),
        });
        // Sender and receiver are dicts with a magic __channel_id field
        let type_key = self.get_heap_mut().push(HeapValue::String("__type"));
        let id_key = self.get_heap_mut().push(HeapValue::String("__channel_id"));
        let id_val = Value::Int(id as i64);

        let sender_type = self.get_heap_mut().push(HeapValue::String("sender"));
        let mut sender_dict = FxHashMap::default();
        sender_dict.insert(type_key, sender_type);
        sender_dict.insert(id_key, id_val);
        let sender = self.get_heap_mut().push(HeapValue::Dict(sender_dict));

        let receiver_type = self.get_heap_mut().push(HeapValue::String("receiver"));
        let mut receiver_dict = FxHashMap::default();
        receiver_dict.insert(type_key, receiver_type);
        receiver_dict.insert(id_key, id_val);
        let receiver = self.get_heap_mut().push(HeapValue::Dict(receiver_dict));

        (sender, receiver)
    }

    pub(crate) fn channel_send(
        &mut self,
        sender_key: crate::arenas::DictKey,
        value: Value,
    ) -> WalrusResult<bool> {
        let dict = self.get_heap().get_dict(sender_key)?.clone();
        let id_key_str = self.get_heap_mut().push(HeapValue::String("__channel_id"));
        if let Some(Value::Int(id)) = dict.get(&id_key_str) {
            let id = *id as usize;
            if id < self.user_channels.len() {
                if *self.user_channels[id].closed.borrow() {
                    return Ok(false);
                }
                self.user_channels[id].buffer.borrow_mut().push_back(value);
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(crate) fn channel_recv(
        &mut self,
        receiver_key: crate::arenas::DictKey,
    ) -> WalrusResult<Value> {
        let dict = self.get_heap().get_dict(receiver_key)?.clone();
        let id_key_str = self.get_heap_mut().push(HeapValue::String("__channel_id"));
        if let Some(Value::Int(id)) = dict.get(&id_key_str) {
            let id = *id as usize;
            if id < self.user_channels.len() {
                // Check buffer first — if data is already available, return Ready task
                let buffered = self.user_channels[id].buffer.borrow_mut().pop_front();
                if let Some(value) = buffered {
                    return Ok(self.create_non_runnable_task(AsyncTask::Ready(value)));
                }
                // No data yet — create a pending recv task
                return Ok(self.create_non_runnable_task(AsyncTask::UserRecv { channel_id: id }));
            }
        }
        Err(WalrusError::GenericError {
            message: "asyncx.recv: invalid receiver".to_string(),
        })
    }

    pub(crate) fn channel_close(
        &mut self,
        endpoint_key: crate::arenas::DictKey,
    ) -> WalrusResult<bool> {
        let dict = self.get_heap().get_dict(endpoint_key)?.clone();
        let id_key_str = self.get_heap_mut().push(HeapValue::String("__channel_id"));
        if let Some(Value::Int(id)) = dict.get(&id_key_str) {
            let id = *id as usize;
            if id < self.user_channels.len() {
                *self.user_channels[id].closed.borrow_mut() = true;
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn cancelled_task_error_value(&mut self) -> Value {
        self.get_heap_mut()
            .push(HeapValue::String("task cancelled"))
    }

    fn cancel_task_recursive_internal(
        &mut self,
        task_key: crate::arenas::TaskKey,
        visited: &mut FxHashSet<crate::arenas::TaskKey>,
    ) -> WalrusResult<bool> {
        if !visited.insert(task_key) {
            return Ok(false);
        }

        let task = self.get_heap().get_task(task_key)?.clone();
        match task {
            AsyncTask::Pending { .. }
            | AsyncTask::Sleep { .. }
            | AsyncTask::Channel(_)
            | AsyncTask::UserRecv { .. } => {
                *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Cancelled;
                self.suspended_tasks.remove(&task_key);
                Ok(true)
            }
            AsyncTask::Timeout { task, .. } => {
                let _ = self.cancel_task_recursive_internal(task, visited)?;
                *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Cancelled;
                self.suspended_tasks.remove(&task_key);
                Ok(true)
            }
            AsyncTask::Gather { tasks }
            | AsyncTask::Race { tasks }
            | AsyncTask::AllSettled { tasks } => {
                for child in tasks {
                    let _ = self.cancel_task_recursive_internal(child, visited)?;
                }
                *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Cancelled;
                self.suspended_tasks.remove(&task_key);
                Ok(true)
            }
            AsyncTask::Ready(_) | AsyncTask::Failed(_) | AsyncTask::Cancelled => Ok(false),
        }
    }

    pub(crate) fn cancel_task(&mut self, task_key: crate::arenas::TaskKey) -> WalrusResult<bool> {
        let mut visited = FxHashSet::default();
        let cancelled = self.cancel_task_recursive_internal(task_key, &mut visited)?;
        if cancelled {
            self.suspended_tasks.remove(&task_key);
            self.wake_task_waiters(task_key)?;
        }
        Ok(cancelled)
    }

    pub(crate) fn run_queued_tasks(&mut self, span: Span) -> WalrusResult<()> {
        while let Some(task_key) = self.next_runnable_task()? {
            self.run_pending_task_to_completion(task_key, span)?;
            self.refresh_waiting_tasks()?;
        }
        Ok(())
    }

    pub(crate) fn task_status_string(
        &self,
        task_key: crate::arenas::TaskKey,
    ) -> WalrusResult<&'static str> {
        let task = self.get_heap().get_task(task_key)?;
        Ok(match task {
            AsyncTask::Pending { .. }
            | AsyncTask::Sleep { .. }
            | AsyncTask::Timeout { .. }
            | AsyncTask::Gather { .. }
            | AsyncTask::Race { .. }
            | AsyncTask::AllSettled { .. }
            | AsyncTask::Channel(_)
            | AsyncTask::UserRecv { .. } => "pending",
            AsyncTask::Ready(_) => "ready",
            AsyncTask::Failed(_) => "failed",
            AsyncTask::Cancelled => "cancelled",
        })
    }

    pub(crate) fn task_is_cancelled(&self, task_key: crate::arenas::TaskKey) -> WalrusResult<bool> {
        let task = self.get_heap().get_task(task_key)?;
        Ok(matches!(task, AsyncTask::Cancelled))
    }

    fn complete_task_on_frame_return(
        &mut self,
        task_key: Option<crate::arenas::TaskKey>,
        result: Value,
    ) -> WalrusResult<()> {
        if let Some(task_key) = task_key {
            let task = self.get_heap_mut().get_mut_task(task_key)?;
            if matches!(task, AsyncTask::Pending { .. }) {
                *task = AsyncTask::Ready(result);
            }
        }
        Ok(())
    }

    fn set_task_failed(
        &mut self,
        task_key: crate::arenas::TaskKey,
        failure: Value,
    ) -> WalrusResult<()> {
        *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Failed(failure);
        self.wake_task_waiters(task_key)?;
        Ok(())
    }

    fn task_failure_value_from_error(&mut self, err: WalrusError) -> Value {
        let message = match err {
            WalrusError::ThrownValue { message, .. } => message,
            WalrusError::RuntimeErrorWithStackTrace { error, stack_trace } => {
                format!("{error}\n{stack_trace}")
            }
            other => other.to_string(),
        };
        self.get_heap_mut().push(HeapValue::String(&message))
    }

    fn fail_task_with_error(
        &mut self,
        task_key: crate::arenas::TaskKey,
        err: WalrusError,
    ) -> WalrusResult<()> {
        let failure = self.task_failure_value_from_error(err);
        self.set_task_failed(task_key, failure)
    }

    fn enqueue_task(&mut self, task_key: crate::arenas::TaskKey) {
        if !self.async_task_queue.contains(&task_key) {
            self.async_task_queue.push_back(task_key);
        }
    }

    fn wake_task_waiters(&mut self, task_key: crate::arenas::TaskKey) -> WalrusResult<()> {
        let Some(waiters) = self.task_waiters.remove(&task_key) else {
            return Ok(());
        };

        for waiter in waiters {
            if self.suspended_tasks.contains_key(&waiter)
                && !matches!(
                    self.get_heap().get_task(waiter)?,
                    AsyncTask::Ready(_) | AsyncTask::Failed(_) | AsyncTask::Cancelled
                )
            {
                self.enqueue_task(waiter);
            }
        }

        Ok(())
    }

    fn refresh_waiting_tasks(&mut self) -> WalrusResult<()> {
        let mut watched = FxHashSet::default();
        if let Some(suspended) = &self.suspended_main {
            watched.insert(suspended.waiting_on);
        }
        watched.extend(self.task_waiters.keys().copied());

        for task_key in watched {
            if matches!(
                self.poll_task_resolution(task_key)?,
                TaskResolution::Ready(_) | TaskResolution::Failed(_) | TaskResolution::Cancelled
            ) {
                self.wake_task_waiters(task_key)?;
            }
        }

        Ok(())
    }

    fn is_task_runnable(&mut self, task_key: crate::arenas::TaskKey) -> WalrusResult<bool> {
        let task = self.get_heap().get_task(task_key)?;
        if !matches!(task, AsyncTask::Pending { .. }) {
            return Ok(false);
        }

        if let Some(suspended) = self.suspended_tasks.get(&task_key) {
            return Ok(matches!(
                self.poll_task_resolution(suspended.waiting_on)?,
                TaskResolution::Ready(_) | TaskResolution::Failed(_) | TaskResolution::Cancelled
            ));
        }

        Ok(true)
    }

    fn poll_task_resolution(
        &mut self,
        task_key: crate::arenas::TaskKey,
    ) -> WalrusResult<TaskResolution> {
        let task = self.get_heap().get_task(task_key)?.clone();
        match task {
            AsyncTask::Pending { .. } => Ok(TaskResolution::Pending),
            AsyncTask::Ready(value) => Ok(TaskResolution::Ready(value)),
            AsyncTask::Failed(value) => Ok(TaskResolution::Failed(value)),
            AsyncTask::Cancelled => Ok(TaskResolution::Cancelled),
            AsyncTask::Sleep { wake_at } => {
                if Instant::now() >= wake_at {
                    *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Ready(Value::Void);
                    Ok(TaskResolution::Ready(Value::Void))
                } else {
                    Ok(TaskResolution::Pending)
                }
            }
            AsyncTask::Timeout { task, deadline } => match self.poll_task_resolution(task)? {
                TaskResolution::Ready(value) => {
                    *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Ready(value);
                    Ok(TaskResolution::Ready(value))
                }
                TaskResolution::Failed(value) => {
                    *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Failed(value);
                    Ok(TaskResolution::Failed(value))
                }
                TaskResolution::Cancelled => {
                    *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Cancelled;
                    Ok(TaskResolution::Cancelled)
                }
                TaskResolution::Pending => {
                    if Instant::now() >= deadline {
                        let message = self
                            .get_heap_mut()
                            .push(HeapValue::String("task timed out"));
                        *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Failed(message);
                        Ok(TaskResolution::Failed(message))
                    } else {
                        Ok(TaskResolution::Pending)
                    }
                }
            },
            AsyncTask::Channel(ref channel) => match channel.try_recv() {
                Ok(Ok(io_result)) => {
                    let value = self.io_result_to_value(io_result)?;
                    *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Ready(value);
                    Ok(TaskResolution::Ready(value))
                }
                Ok(Err(error_msg)) => {
                    let error = self.get_heap_mut().push(HeapValue::String(&error_msg));
                    *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Failed(error);
                    Ok(TaskResolution::Failed(error))
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => Ok(TaskResolution::Pending),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    let error = self
                        .get_heap_mut()
                        .push(HeapValue::String("I/O worker thread disconnected"));
                    *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Failed(error);
                    Ok(TaskResolution::Failed(error))
                }
            },
            AsyncTask::Gather { tasks } => {
                let mut values = Vec::with_capacity(tasks.len());
                for child in tasks {
                    match self.poll_task_resolution(child)? {
                        TaskResolution::Ready(value) => values.push(value),
                        TaskResolution::Failed(value) => {
                            *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Failed(value);
                            return Ok(TaskResolution::Failed(value));
                        }
                        TaskResolution::Cancelled => {
                            *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Cancelled;
                            return Ok(TaskResolution::Cancelled);
                        }
                        TaskResolution::Pending => {
                            return Ok(TaskResolution::Pending);
                        }
                    }
                }

                let list = self.get_heap_mut().push(HeapValue::List(values));
                *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Ready(list);
                Ok(TaskResolution::Ready(list))
            }
            AsyncTask::Race { tasks } => {
                let mut all_failed = true;
                let mut last_failure = None;
                for child in tasks {
                    match self.poll_task_resolution(child)? {
                        TaskResolution::Ready(value) => {
                            *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Ready(value);
                            return Ok(TaskResolution::Ready(value));
                        }
                        TaskResolution::Failed(value) => {
                            last_failure = Some(value);
                        }
                        TaskResolution::Cancelled => {}
                        TaskResolution::Pending => {
                            all_failed = false;
                        }
                    }
                }
                if all_failed {
                    if let Some(failure) = last_failure {
                        *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Failed(failure);
                        Ok(TaskResolution::Failed(failure))
                    } else {
                        *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Cancelled;
                        Ok(TaskResolution::Cancelled)
                    }
                } else {
                    Ok(TaskResolution::Pending)
                }
            }
            AsyncTask::AllSettled { tasks } => {
                let mut results = Vec::with_capacity(tasks.len());
                for child in tasks {
                    match self.poll_task_resolution(child)? {
                        TaskResolution::Ready(value) => {
                            let status_str = self.get_heap_mut().push(HeapValue::String("ok"));
                            let value_str = self.get_heap_mut().push(HeapValue::String("value"));
                            let status_key = self.get_heap_mut().push(HeapValue::String("status"));
                            let mut dict = FxHashMap::default();
                            dict.insert(status_key, status_str);
                            dict.insert(value_str, value);
                            results.push(self.get_heap_mut().push(HeapValue::Dict(dict)));
                        }
                        TaskResolution::Failed(value) => {
                            let status_str = self.get_heap_mut().push(HeapValue::String("error"));
                            let error_str = self.get_heap_mut().push(HeapValue::String("error"));
                            let status_key = self.get_heap_mut().push(HeapValue::String("status"));
                            let mut dict = FxHashMap::default();
                            dict.insert(status_key, status_str);
                            dict.insert(error_str, value);
                            results.push(self.get_heap_mut().push(HeapValue::Dict(dict)));
                        }
                        TaskResolution::Cancelled => {
                            let status_str =
                                self.get_heap_mut().push(HeapValue::String("cancelled"));
                            let status_key = self.get_heap_mut().push(HeapValue::String("status"));
                            let mut dict = FxHashMap::default();
                            dict.insert(status_key, status_str);
                            results.push(self.get_heap_mut().push(HeapValue::Dict(dict)));
                        }
                        TaskResolution::Pending => {
                            return Ok(TaskResolution::Pending);
                        }
                    }
                }
                let list = self.get_heap_mut().push(HeapValue::List(results));
                *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Ready(list);
                Ok(TaskResolution::Ready(list))
            }
            AsyncTask::UserRecv { channel_id } => {
                if channel_id < self.user_channels.len() {
                    let mut buf = self.user_channels[channel_id].buffer.borrow_mut();
                    if let Some(value) = buf.pop_front() {
                        drop(buf);
                        *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Ready(value);
                        return Ok(TaskResolution::Ready(value));
                    }
                    drop(buf);
                    // Check if channel is closed with empty buffer
                    if *self.user_channels[channel_id].closed.borrow() {
                        *self.get_heap_mut().get_mut_task(task_key)? =
                            AsyncTask::Ready(Value::Void);
                        return Ok(TaskResolution::Ready(Value::Void));
                    }
                }
                Ok(TaskResolution::Pending)
            }
        }
    }

    fn next_deadline_for_task(
        &self,
        task_key: crate::arenas::TaskKey,
        visited: &mut FxHashSet<crate::arenas::TaskKey>,
    ) -> WalrusResult<Option<Instant>> {
        if !visited.insert(task_key) {
            return Ok(None);
        }

        let task = self.get_heap().get_task(task_key)?;
        match task {
            AsyncTask::Sleep { wake_at } => Ok(Some(*wake_at)),
            AsyncTask::Timeout { task, deadline } => {
                let nested = self.next_deadline_for_task(*task, visited)?;
                Ok(match nested {
                    Some(value) => Some((*deadline).min(value)),
                    None => Some(*deadline),
                })
            }
            AsyncTask::Gather { tasks }
            | AsyncTask::Race { tasks }
            | AsyncTask::AllSettled { tasks } => {
                let mut soonest: Option<Instant> = None;
                for &child in tasks {
                    if let Some(deadline) = self.next_deadline_for_task(child, visited)? {
                        soonest = Some(match soonest {
                            Some(current) => current.min(deadline),
                            None => deadline,
                        });
                    }
                }
                Ok(soonest)
            }
            AsyncTask::Pending { .. }
            | AsyncTask::Channel(_)
            | AsyncTask::UserRecv { .. }
            | AsyncTask::Ready(_)
            | AsyncTask::Failed(_)
            | AsyncTask::Cancelled => Ok(None),
        }
    }

    fn next_scheduler_deadline(&self) -> WalrusResult<Option<Instant>> {
        let mut watched = FxHashSet::default();
        let mut deadline: Option<Instant> = None;

        if let Some(suspended) = &self.suspended_main {
            watched.insert(suspended.waiting_on);
        }
        watched.extend(self.task_waiters.keys().copied());

        for task_key in watched {
            let mut visited = FxHashSet::default();
            if let Some(next) = self.next_deadline_for_task(task_key, &mut visited)? {
                deadline = Some(match deadline {
                    Some(current) => current.min(next),
                    None => next,
                });
            }
        }

        Ok(deadline)
    }

    fn scheduler_has_pending_io(&self) -> WalrusResult<bool> {
        let mut watched = FxHashSet::default();
        if let Some(suspended) = &self.suspended_main {
            watched.insert(suspended.waiting_on);
        }
        watched.extend(self.task_waiters.keys().copied());

        for task_key in watched {
            let mut visited = FxHashSet::default();
            if self.task_has_pending_io(task_key, &mut visited)? {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn wait_for_scheduler_progress(&mut self) -> WalrusResult<()> {
        let deadline = self.next_scheduler_deadline()?;
        let has_io = self.scheduler_has_pending_io()?;

        if deadline.is_none() && !has_io {
            return Err(WalrusError::GenericError {
                message:
                    "Event loop deadlock: all suspended work is waiting with no runnable tasks"
                        .to_string(),
            });
        }

        if let Some(deadline) = deadline {
            let now = Instant::now();
            if deadline > now {
                let timeout = deadline.duration_since(now);
                let _ = self.io_wakeup_rx.recv_timeout(timeout);
            }
        } else {
            let _ = self.io_wakeup_rx.recv();
        }

        Ok(())
    }

    fn main_waiting_task_resolved(&mut self) -> WalrusResult<bool> {
        let Some(suspended) = &self.suspended_main else {
            return Ok(false);
        };

        Ok(matches!(
            self.poll_task_resolution(suspended.waiting_on)?,
            TaskResolution::Ready(_) | TaskResolution::Failed(_) | TaskResolution::Cancelled
        ))
    }

    fn resume_main_if_ready(&mut self) -> WalrusResult<bool> {
        if !self.main_waiting_task_resolved()? {
            return Ok(false);
        }

        let suspended = self
            .suspended_main
            .take()
            .expect("main suspension should exist when ready");
        self.restore_context(suspended.context);
        Ok(true)
    }

    fn next_runnable_task(&mut self) -> WalrusResult<Option<crate::arenas::TaskKey>> {
        while let Some(task_key) = self.async_task_queue.pop_front() {
            if self.is_task_runnable(task_key)? {
                return Ok(Some(task_key));
            }
        }
        Ok(None)
    }

    fn run_pending_task_to_completion(
        &mut self,
        task_key: crate::arenas::TaskKey,
        span: Span,
    ) -> WalrusResult<()> {
        let task_snapshot = self.get_heap().get_task(task_key)?.clone();
        let AsyncTask::Pending { function, args } = task_snapshot else {
            return Ok(());
        };

        let caller_context = self.take_context();
        let suspended_task = self.suspended_tasks.remove(&task_key);
        let function = self.get_heap().get_function(function)?.clone();
        match function {
            WalrusFunction::Vm(func) => {
                if args.len() != func.arity {
                    self.restore_context(caller_context);
                    return self.fail_task_with_error(
                        task_key,
                        WalrusError::InvalidArgCount {
                            name: func.name.clone(),
                            expected: func.arity,
                            got: args.len(),
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        },
                    );
                }

                let context = if let Some(suspended) = suspended_task {
                    suspended.context
                } else {
                    ExecutionContext {
                        stack: Vec::new(),
                        locals: args,
                        call_stack: vec![CallFrame {
                            return_ip: 0,
                            frame_pointer: 0,
                            stack_pointer: 0,
                            instructions: Rc::clone(&func.code),
                            function_name: func.name.clone(),
                            return_override: None,
                            module_binding: func.module_binding.clone(),
                            awaiting_task: Some(task_key),
                        }],
                        exception_handlers: Vec::new(),
                        ip: 0,
                    }
                };

                self.restore_context(context);
                let outcome = self.run_inner();
                let task_context = self.take_context();
                self.restore_context(caller_context);

                match outcome {
                    Ok(RunSignal::Returned(_)) => {
                        if matches!(
                            self.poll_task_resolution(task_key)?,
                            TaskResolution::Pending
                        ) {
                            let failure = self.get_heap_mut().push(HeapValue::String(
                                "Async task exited without returning a result",
                            ));
                            self.set_task_failed(task_key, failure)?;
                        } else {
                            self.wake_task_waiters(task_key)?;
                        }
                    }
                    Ok(RunSignal::Suspended(waiting_on)) => {
                        self.suspended_tasks.insert(
                            task_key,
                            SuspendedExecution {
                                context: task_context,
                                waiting_on,
                            },
                        );
                        self.task_waiters
                            .entry(waiting_on)
                            .or_default()
                            .push(task_key);
                    }
                    Err(err) => {
                        let failure = self.task_failure_value_from_error(err);
                        self.set_task_failed(task_key, failure)?;
                    }
                }
            }
            WalrusFunction::Rust(rust_fn) => {
                if args.len() != rust_fn.args {
                    self.restore_context(caller_context);
                    return self.fail_task_with_error(
                        task_key,
                        WalrusError::InvalidArgCount {
                            name: rust_fn.name.clone(),
                            expected: rust_fn.args,
                            got: args.len(),
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        },
                    );
                }
                let result = match rust_fn.call(args, self.source_ref, span) {
                    Ok(value) => value,
                    Err(err) => {
                        self.restore_context(caller_context);
                        return self.fail_task_with_error(task_key, err);
                    }
                };
                self.complete_task_on_frame_return(Some(task_key), result)?;
                self.restore_context(caller_context);
                self.wake_task_waiters(task_key)?;
            }
            WalrusFunction::Native(native_fn) => {
                let result = match self.call_native(native_fn, args, span) {
                    Ok(value) => value,
                    Err(err) => {
                        self.restore_context(caller_context);
                        return self.fail_task_with_error(task_key, err);
                    }
                };
                self.complete_task_on_frame_return(Some(task_key), result)?;
                self.restore_context(caller_context);
                self.wake_task_waiters(task_key)?;
            }
            WalrusFunction::TreeWalk(_) => {
                self.restore_context(caller_context);
                return self.fail_task_with_error(
                    task_key,
                    WalrusError::NodeFunctionNotSupportedInVm {
                        span,
                        src: self.source_ref.source().into(),
                        filename: self.source_ref.filename().into(),
                    },
                );
            }
        }

        Ok(())
    }

    fn call_exported_function(
        &mut self,
        function: WalrusFunction,
        args: Vec<Value>,
        span: Span,
    ) -> WalrusResult<Option<Value>> {
        match function {
            WalrusFunction::Native(native) => {
                let result = self.call_native(native, args, span)?;
                Ok(Some(result))
            }
            WalrusFunction::Rust(rust_fn) => {
                if args.len() != rust_fn.args {
                    return Err(WalrusError::InvalidArgCount {
                        name: rust_fn.name.clone(),
                        expected: rust_fn.args,
                        got: args.len(),
                        span,
                        src: self.source_ref.source().into(),
                        filename: self.source_ref.filename().into(),
                    });
                }

                let result = rust_fn.call(args, self.source_ref, span)?;
                Ok(Some(result))
            }
            WalrusFunction::Vm(func) => {
                if args.len() != func.arity {
                    return Err(WalrusError::InvalidArgCount {
                        name: func.name.clone(),
                        expected: func.arity,
                        got: args.len(),
                        span,
                        src: self.source_ref.source().into(),
                        filename: self.source_ref.filename().into(),
                    });
                }

                if func.is_async {
                    let task = self
                        .create_task_for_cloned_function(WalrusFunction::Vm(func.clone()), args);
                    return Ok(Some(task));
                }

                let new_frame = CallFrame {
                    return_ip: self.ip,
                    frame_pointer: self.locals.len(),
                    stack_pointer: self.stack.len(),
                    instructions: Rc::clone(&func.code),
                    function_name: func.name.clone(),
                    return_override: None,
                    module_binding: func.module_binding.clone(),
                    awaiting_task: None,
                };

                self.call_stack.push(new_frame);
                self.locals.extend(args);
                self.ip = 0;
                Ok(None)
            }
            WalrusFunction::TreeWalk(_) => Err(WalrusError::NodeFunctionNotSupportedInVm {
                span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            }),
        }
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
    fn normalize_index(index: i64, len: usize) -> Option<usize> {
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
    fn char_to_byte_offset(s: &str, char_index: usize) -> Option<usize> {
        if char_index == 0 {
            return Some(0);
        }
        if char_index == s.chars().count() {
            return Some(s.len());
        }
        s.char_indices().nth(char_index).map(|(offset, _)| offset)
    }

    /// Remove handlers that are no longer reachable from the current execution point.
    ///
    /// Handlers become stale when:
    /// - Their frame has been popped.
    /// - Control flow in the same frame has jumped out of the protected try range.
    fn prune_exception_handlers(&mut self) {
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
    /// Used when a frame is reused (tail call) or dropped.
    fn clear_exception_handlers_from_frame(&mut self, frame_index: usize) {
        while let Some(handler) = self.exception_handlers.last().copied() {
            if handler.frame_index >= frame_index {
                self.exception_handlers.pop();
            } else {
                break;
            }
        }
    }

    /// Raise a thrown value, transferring control to the nearest active catch handler.
    fn throw_value(&mut self, value: Value, span: Span) -> WalrusResult<()> {
        while let Some(handler) = self.exception_handlers.last().copied() {
            if handler.frame_index > self.current_frame_index() {
                self.exception_handlers.pop();
                continue;
            }

            self.exception_handlers.pop();

            // Unwind call frames until we reach the frame that owns this handler.
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

    /// Run the VM and add stack trace information to any errors
    pub fn run(&mut self) -> WalrusResult<Value> {
        loop {
            match self.run_inner() {
                Ok(RunSignal::Returned(value)) => return Ok(value),
                Ok(RunSignal::Suspended(waiting_on)) => {
                    self.suspended_main = Some(SuspendedExecution {
                        context: self.take_context(),
                        waiting_on,
                    });

                    while self.suspended_main.is_some() {
                        self.refresh_waiting_tasks()?;
                        if self.resume_main_if_ready()? {
                            break;
                        }

                        if let Some(task_key) = self.next_runnable_task()? {
                            self.run_pending_task_to_completion(task_key, Span::default())?;
                            continue;
                        }

                        self.wait_for_scheduler_progress()?;
                    }
                }
                Err(err) => {
                    let stack_trace = self.format_stack_trace();
                    return if stack_trace.is_empty() {
                        Err(err)
                    } else {
                        Err(WalrusError::RuntimeErrorWithStackTrace {
                            error: err.to_string(),
                            stack_trace,
                        })
                    };
                }
            }
        }
    }

    fn run_inner(&mut self) -> WalrusResult<RunSignal> {
        let debug_logging_enabled = log_enabled!(log::Level::Debug);
        let profiling_enabled = self.profiling_enabled;

        'vm: loop {
            // Poll GC periodically instead of every instruction.
            self.gc_poll_counter = self.gc_poll_counter.wrapping_add(1);
            if self.gc_poll_counter & 0xFF == 0 {
                self.maybe_collect_garbage();
            }

            // Drop handlers that became unreachable because control flow
            // moved outside their protected lexical try range.
            self.prune_exception_handlers();

            // Check if we've reached the end of the current frame's instructions
            if self.ip >= self.current_frame().instructions.instructions.len() {
                // If we're in the main frame, we're done
                if self.call_stack.len() == 1 {
                    return Err(WalrusError::UnknownError {
                        message: "Instruction pointer out of bounds".to_string(),
                    });
                }
                // Otherwise this shouldn't happen (Return should have been called)
                return Err(WalrusError::UnknownError {
                    message: "Function ended without return".to_string(),
                });
            }

            if debug_logging_enabled {
                let instructions = self.current_frame().instructions.as_ref();
                instructions.disassemble_single(self.ip, self.function_name());
            }

            // Check if debugger should pause
            if self.debug_mode {
                if let Some(ref mut dbg) = self.debugger {
                    let call_depth = self.call_stack.len();
                    if dbg.should_break(self.ip, call_depth) || dbg.should_prompt {
                        dbg.trigger_prompt();
                        let instructions = Rc::clone(&self.current_frame().instructions);
                        match self.run_debugger_prompt(instructions.as_ref())? {
                            debugger::DebuggerCommand::Quit => {
                                return Err(WalrusError::UnknownError {
                                    message: "Debugger quit".to_string(),
                                });
                            }
                            _ => {} // Continue execution with the new mode
                        }
                    }
                }
            }

            let instruction = self.current_frame().instructions.get(self.ip);
            let opcode = instruction.opcode();
            let span = instruction.span();

            self.ip += 1;

            match opcode {
                Opcode::LoadConst(index) => {
                    self.push(self.current_frame().instructions.get_constant(index));
                }
                Opcode::LoadConst0 => {
                    self.push(self.current_frame().instructions.get_constant(0));
                }
                Opcode::LoadConst1 => {
                    self.push(self.current_frame().instructions.get_constant(1));
                }
                Opcode::Load(index) => {
                    let fp = self.frame_pointer();
                    let idx = fp + index as usize;
                    // SAFETY: compiler guarantees local indices are valid for this frame.
                    self.push(unsafe { *self.locals.get_unchecked(idx) });
                }
                Opcode::LoadLocal0 => {
                    let fp = self.frame_pointer();
                    // SAFETY: compiler emits LoadLocal0 only when local 0 exists.
                    self.push(unsafe { *self.locals.get_unchecked(fp) });
                }
                Opcode::LoadLocal1 => {
                    let fp = self.frame_pointer();
                    // SAFETY: compiler emits LoadLocal1 only when local 1 exists.
                    self.push(unsafe { *self.locals.get_unchecked(fp + 1) });
                }
                Opcode::LoadLocal2 => {
                    let fp = self.frame_pointer();
                    // SAFETY: compiler emits LoadLocal2 only when local 2 exists.
                    self.push(unsafe { *self.locals.get_unchecked(fp + 2) });
                }
                Opcode::LoadLocal3 => {
                    let fp = self.frame_pointer();
                    // SAFETY: compiler emits LoadLocal3 only when local 3 exists.
                    self.push(unsafe { *self.locals.get_unchecked(fp + 3) });
                }
                // Specialized increment/decrement for loop counters (hot path)
                Opcode::IncrementLocal(index) => {
                    let fp = self.frame_pointer();
                    let idx = fp + index as usize;
                    // SAFETY: compiler guarantees local indices are valid for this frame.
                    let local = unsafe { self.locals.get_unchecked_mut(idx) };
                    if let Value::Int(v) = *local {
                        *local = Value::Int(v + 1);
                    } else {
                        return Err(WalrusError::TypeMismatch {
                            expected: "int".to_string(),
                            found: local.get_type().to_string(),
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        });
                    }
                }
                Opcode::DecrementLocal(index) => {
                    let fp = self.frame_pointer();
                    let idx = fp + index as usize;
                    // SAFETY: compiler guarantees local indices are valid for this frame.
                    let local = unsafe { self.locals.get_unchecked_mut(idx) };
                    if let Value::Int(v) = *local {
                        *local = Value::Int(v - 1);
                    } else {
                        return Err(WalrusError::TypeMismatch {
                            expected: "int".to_string(),
                            found: local.get_type().to_string(),
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        });
                    }
                }
                // Optimized range loop - no heap allocation!
                Opcode::ForRangeInit(local_idx) => {
                    // Stack: [start, end] -> locals[idx] = start, locals[idx+1] = end
                    let end = self.pop_unchecked();
                    let start = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    let idx = fp + local_idx as usize;
                    // Ensure we have space for both values
                    while self.locals.len() <= idx + 1 {
                        self.locals.push(Value::Void);
                    }
                    self.locals[idx] = start;
                    self.locals[idx + 1] = end;
                }
                Opcode::ForRangeNext(jump_target, local_idx) => {
                    let loop_header_ip = self.ip - 1;
                    let exit_ip = jump_target as usize;

                    if profiling_enabled {
                        // Profile for hotspot detection
                        self.profile_loop_iteration(loop_header_ip, exit_ip);

                        // Try JIT execution if available
                        #[cfg(feature = "jit")]
                        if let Some(jit_exit) =
                            self.try_jit_range_loop(loop_header_ip, local_idx, jump_target)
                        {
                            self.ip = jit_exit;
                            continue;
                        }

                        // Try to compile hot loops
                        #[cfg(feature = "jit")]
                        self.try_compile_hot_range_loop(loop_header_ip, exit_ip);
                    }

                    // Standard interpreted execution
                    let fp = self.frame_pointer();
                    let idx = fp + local_idx as usize;
                    // SAFETY: compiler guarantees range-loop locals are allocated at idx and idx+1.
                    let current_value = unsafe { *self.locals.get_unchecked(idx) };
                    // SAFETY: see above.
                    let end_value = unsafe { *self.locals.get_unchecked(idx + 1) };
                    if let (Value::Int(current), Value::Int(end)) = (current_value, end_value) {
                        if current < end {
                            self.push(Value::Int(current));
                            // SAFETY: idx is valid (see above).
                            unsafe {
                                *self.locals.get_unchecked_mut(idx) = Value::Int(current + 1);
                            }
                        } else {
                            self.ip = jump_target as usize;
                        }
                    } else {
                        return Err(WalrusError::TypeMismatch {
                            expected: "int and int (range bounds)".to_string(),
                            found: format!(
                                "{} and {}",
                                current_value.get_type(),
                                end_value.get_type()
                            ),
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        });
                    }
                }
                Opcode::LoadGlobal(index) => {
                    let value = self.load_global_value(index as usize, span)?;
                    self.push(value);
                }
                Opcode::LoadGlobal0 => {
                    let value = self.load_global_value(0, span)?;
                    self.push(value);
                }
                Opcode::LoadGlobal1 => {
                    let value = self.load_global_value(1, span)?;
                    self.push(value);
                }
                Opcode::LoadGlobal2 => {
                    let value = self.load_global_value(2, span)?;
                    self.push(value);
                }
                Opcode::LoadGlobal3 => {
                    let value = self.load_global_value(3, span)?;
                    self.push(value);
                }
                Opcode::Store => {
                    // SAFETY: valid bytecode guarantees stack has a value to store.
                    let value = self.pop_unchecked();
                    self.locals.push(value);
                }
                Opcode::StoreAt(index) => {
                    // SAFETY: valid bytecode guarantees stack has a value to store.
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    let abs_index = fp + index as usize;

                    if abs_index == self.locals.len() {
                        self.locals.push(value);
                    } else {
                        // SAFETY: compiler guarantees local index is valid when not appending.
                        unsafe {
                            *self.locals.get_unchecked_mut(abs_index) = value;
                        }
                    }
                }
                Opcode::StoreGlobal(index) => {
                    // SAFETY: valid bytecode guarantees stack has a value to store.
                    let value = self.pop_unchecked();
                    self.store_global_value(index as usize, value, span)?;
                }
                Opcode::Reassign(index) => {
                    // SAFETY: valid bytecode guarantees stack has a value to assign.
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    let idx = fp + index as usize;
                    // SAFETY: compiler guarantees reassigned local exists.
                    unsafe {
                        *self.locals.get_unchecked_mut(idx) = value;
                    }
                }
                Opcode::ReassignGlobal(index) => {
                    // SAFETY: valid bytecode guarantees stack has a value to assign.
                    let value = self.pop_unchecked();
                    self.store_global_value(index as usize, value, span)?;
                }
                Opcode::List(cap) => {
                    let cap = cap as usize;

                    // Check we have enough items on the stack
                    if self.stack.len() < cap {
                        return Err(WalrusError::StackUnderflow {
                            op: opcode,
                            span,
                            src: self.source_ref.source().to_string(),
                            filename: self.source_ref.filename().to_string(),
                        });
                    }

                    // Extract items from stack without reverse
                    // split_off gives us the last cap items in correct order
                    let list = self.stack.split_off(self.stack.len() - cap);

                    let value = self.get_heap_mut().push(HeapValue::List(list));
                    self.push(value);
                }
                Opcode::Dict(cap) => {
                    let cap = cap as usize;
                    let mut dict = FxHashMap::with_capacity_and_hasher(cap, Default::default());

                    for _ in 0..cap {
                        let value = self.pop(opcode, span)?;
                        let key = self.pop(opcode, span)?;

                        dict.insert(key, value);
                    }

                    let value = self.get_heap_mut().push(HeapValue::Dict(dict));
                    self.push(value);
                }
                Opcode::Range => {
                    let left = self.pop(opcode, span)?;
                    let right = self.pop(opcode, span)?;

                    // fixme: the spans are wrong here
                    match (left, right) {
                        (Value::Void, Value::Void) => {
                            self.push(Value::Range(RangeValue::new(0, span, -1, span)));
                        }
                        (Value::Void, Value::Int(right)) => {
                            self.push(Value::Range(RangeValue::new(0, span, right, span)));
                        }
                        (Value::Int(left), Value::Void) => {
                            self.push(Value::Range(RangeValue::new(left, span, -1, span)));
                        }
                        (Value::Int(left), Value::Int(right)) => {
                            self.push(Value::Range(RangeValue::new(left, span, right, span)));
                        }
                        // fixme: this is a catch all for now, break it into
                        // errors for left and right and then both
                        (left, right) => {
                            return Err(WalrusError::TypeMismatch {
                                expected: "type: todo".to_string(),
                                found: format!("{} and {}", left.get_type(), right.get_type()),
                                span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            });
                        }
                    }
                }
                Opcode::True => self.push(Value::Bool(true)),
                Opcode::False => self.push(Value::Bool(false)),
                Opcode::Void => self.push(Value::Void),
                Opcode::Pop => {
                    // SAFETY: valid bytecode guarantees stack has a value to pop.
                    self.pop_unchecked();
                }
                Opcode::PopLocal(num) => {
                    for _ in 0..num {
                        self.locals.pop();
                    }
                }
                Opcode::JumpIfFalse(offset) => {
                    // SAFETY: compiler guarantees conditional jump has a condition value.
                    let value = self.pop_unchecked();

                    if !self.get_heap().is_truthy(value)? {
                        self.ip = offset as usize;
                    }
                }
                Opcode::Jump(offset) => {
                    // JIT PROFILING: Backward jumps indicate loops (while loops)
                    if profiling_enabled && (offset as usize) < self.ip {
                        let loop_header_ip = offset as usize;

                        // Register while loop dynamically
                        if !self.hotspot_detector.is_loop_header(loop_header_ip) {
                            // For while loops, the back edge is here (current IP - 1)
                            // and the header is the jump target
                            self.hotspot_detector.register_loop(
                                loop_header_ip,
                                self.ip - 1,
                                self.ip, // Exit is right after this jump
                            );
                        }

                        if self.hotspot_detector.record_loop_iteration(loop_header_ip) {
                            debug!("Hot while loop detected at IP {}", loop_header_ip);
                        }
                    }

                    self.ip = offset as usize;
                }
                Opcode::GetIter => {
                    let value = self.pop(opcode, span)?;
                    match value {
                        Value::StructInst(inst_key) => {
                            let (struct_name, iter_method, has_next) = {
                                let heap = self.get_heap();
                                let inst = heap.get_struct_inst(inst_key)?;
                                let struct_def = heap.get_struct_def(inst.struct_def())?;
                                (
                                    struct_def.name().to_string(),
                                    struct_def.get_method("iter").cloned(),
                                    struct_def.get_method("next").is_some(),
                                )
                            };

                            if let Some(WalrusFunction::Vm(func)) = iter_method {
                                let expected_without_self = func.arity.saturating_sub(1);
                                if expected_without_self != 0 {
                                    return Err(WalrusError::InvalidArgCount {
                                        name: format!("{}::iter", struct_name),
                                        expected: expected_without_self,
                                        got: 0,
                                        span,
                                        src: self.source_ref.source().into(),
                                        filename: self.source_ref.filename().into(),
                                    });
                                }

                                let new_frame = CallFrame {
                                    return_ip: self.ip,
                                    frame_pointer: self.locals.len(),
                                    stack_pointer: self.stack.len(),
                                    instructions: Rc::clone(&func.code),
                                    function_name: format!("{}::iter", struct_name),
                                    return_override: None,
                                    module_binding: func.module_binding.clone(),
                                    awaiting_task: None,
                                };

                                self.call_stack.push(new_frame);
                                self.locals.push(Value::StructInst(inst_key));
                                self.ip = 0;
                                continue;
                            } else if iter_method.is_some() {
                                return Err(WalrusError::StructMethodMustBeVmFunction {
                                    span,
                                    src: self.source_ref.source().into(),
                                    filename: self.source_ref.filename().into(),
                                });
                            } else if has_next {
                                // Iterator object: no separate iter() needed.
                                self.push(Value::StructInst(inst_key));
                            } else {
                                return Err(WalrusError::NotIterable {
                                    type_name: struct_name,
                                    span,
                                    src: self.source_ref.source().into(),
                                    filename: self.source_ref.filename().into(),
                                });
                            }
                        }
                        _ => {
                            if !matches!(
                                value,
                                Value::List(_)
                                    | Value::Tuple(_)
                                    | Value::Dict(_)
                                    | Value::String(_)
                                    | Value::Range(_)
                                    | Value::Iter(_)
                            ) {
                                return Err(WalrusError::NotIterable {
                                    type_name: value.get_type().to_string(),
                                    span,
                                    src: self.source_ref.source().into(),
                                    filename: self.source_ref.filename().into(),
                                });
                            }

                            let iter = self.get_heap_mut().value_to_iter(value)?;
                            self.push(iter);
                        }
                    }
                }
                Opcode::IterNext(offset) => {
                    // JIT PROFILING: This is also a loop header for iterator-based loops
                    if profiling_enabled {
                        let loop_header_ip = self.ip - 1;

                        // Dynamic loop registration for iterator loops
                        if !self.hotspot_detector.is_loop_header(loop_header_ip) {
                            self.hotspot_detector.register_loop(
                                loop_header_ip,
                                loop_header_ip,
                                offset as usize,
                            );
                        }

                        if self.hotspot_detector.record_loop_iteration(loop_header_ip) {
                            debug!("Hot iterator loop detected at IP {}", loop_header_ip);
                        }
                    }

                    let iter =
                        self.stack
                            .last()
                            .copied()
                            .ok_or_else(|| WalrusError::StackUnderflow {
                                op: opcode,
                                span,
                                src: self.source_ref.source().to_string(),
                                filename: self.source_ref.filename().to_string(),
                            })?;

                    let Value::Iter(key) = iter else {
                        return Err(WalrusError::NotIterable {
                            type_name: iter.get_type().to_string(),
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        });
                    };

                    unsafe {
                        let mut ptr = NonNull::from(self.get_heap_mut());
                        let iter = ptr.as_mut().get_mut_iter(key)?;

                        if let Some(value) = iter.next(self.get_heap_mut()) {
                            // Keep iterator on stack and push only the next item.
                            self.push(value);
                        } else {
                            // Iterator exhausted: pop it and jump to loop exit.
                            self.pop_unchecked();
                            self.ip = offset as usize;
                        }
                    }
                }
                Opcode::Call(args) => {
                    let arg_count = args as usize;
                    let func = self.pop(opcode, span)?;

                    if self.stack.len() < arg_count {
                        return Err(WalrusError::StackUnderflow {
                            op: opcode,
                            span,
                            src: self.source_ref.source().to_string(),
                            filename: self.source_ref.filename().to_string(),
                        });
                    }

                    // JIT PROFILING: Track function calls and argument types
                    if profiling_enabled {
                        if let Value::Function(key) = func {
                            if let Ok(func_ref) = self.get_heap().get_function(key) {
                                let name = match func_ref {
                                    WalrusFunction::Vm(f) => f.name.clone(),
                                    WalrusFunction::Rust(f) => f.name.clone(),
                                    _ => String::new(),
                                };
                                if !name.is_empty() {
                                    if self.hotspot_detector.record_function_call(&name) {
                                        debug!("Hot function detected: {}", name);
                                    }
                                    // Track argument types
                                    let args_start = self.stack.len() - arg_count;
                                    for (i, arg) in self.stack[args_start..].iter().enumerate() {
                                        let arg_type = WalrusType::from_value(arg);
                                        self.type_profile.observe(self.ip - 1 + i, arg_type);
                                    }
                                }
                            }
                        }
                    }

                    match func {
                        Value::Function(key) => {
                            let func = self.get_heap().get_function(key)?;

                            match func {
                                WalrusFunction::Vm(func) => {
                                    if arg_count != func.arity {
                                        return Err(WalrusError::InvalidArgCount {
                                            name: func.name.clone(),
                                            expected: func.arity,
                                            got: arg_count,
                                            span,
                                            src: self.source_ref.source().into(),
                                            filename: self.source_ref.filename().into(),
                                        });
                                    }

                                    if func.is_async {
                                        let args = self.pop_n(arg_count, opcode, span)?;
                                        let task = self.create_task_for_function_key(key, args);
                                        self.push(task);
                                        continue;
                                    }

                                    // Create a new call frame instead of a child VM
                                    let new_frame = CallFrame {
                                        return_ip: self.ip,                          // Where to return after this function
                                        frame_pointer: self.locals.len(), // New frame starts at current locals end
                                        stack_pointer: self.stack.len() - arg_count, // Operand stack position for cleanup on return
                                        instructions: Rc::clone(&func.code), // Share the instruction set via Rc
                                        function_name: String::new(),
                                        return_override: None,
                                        module_binding: func.module_binding.clone(),
                                        awaiting_task: None,
                                    };

                                    // Push the new frame
                                    self.call_stack.push(new_frame);

                                    // Move arguments directly from operand stack to locals.
                                    let args_start = self.stack.len() - arg_count;
                                    self.locals.extend(self.stack.drain(args_start..));

                                    // Start execution at the beginning of the new function
                                    self.ip = 0;
                                }
                                WalrusFunction::Rust(func) => {
                                    let func = func.clone();
                                    let args = self.pop_n(arg_count, opcode, span)?;
                                    if args.len() != func.args {
                                        return Err(WalrusError::InvalidArgCount {
                                            name: func.name.clone(),
                                            expected: func.args,
                                            got: args.len(),
                                            span,
                                            src: self.source_ref.source().into(),
                                            filename: self.source_ref.filename().into(),
                                        });
                                    }
                                    let result = func.call(args, self.source_ref, span)?;
                                    self.push(result);
                                }
                                WalrusFunction::Native(native_fn) => {
                                    let native_fn = *native_fn;
                                    let args = self.pop_n(arg_count, opcode, span)?;
                                    let result = self.call_native(native_fn, args, span)?;
                                    self.push(result);
                                }
                                _ => {
                                    // In theory, this should never happen because the compiler
                                    // should not compile a call to a node function (but just in case)
                                    return Err(WalrusError::NodeFunctionNotSupportedInVm {
                                        span,
                                        src: self.source_ref.source().into(),
                                        filename: self.source_ref.filename().into(),
                                    });
                                }
                            }
                        }
                        Value::StructDef(struct_def_key) => {
                            let (struct_name, init_method) = {
                                let struct_def = self.get_heap().get_struct_def(struct_def_key)?;
                                (
                                    struct_def.name().to_string(),
                                    struct_def.get_method("init").cloned(),
                                )
                            };

                            match init_method {
                                Some(WalrusFunction::Vm(init_func)) => {
                                    let expected_without_self = init_func.arity.saturating_sub(1);
                                    if arg_count != expected_without_self {
                                        return Err(WalrusError::InvalidArgCount {
                                            name: format!("{}::init", struct_name),
                                            expected: expected_without_self,
                                            got: arg_count,
                                            span,
                                            src: self.source_ref.source().into(),
                                            filename: self.source_ref.filename().into(),
                                        });
                                    }

                                    let instance = crate::structs::StructInstance::new(
                                        struct_name.clone(),
                                        struct_def_key,
                                    );
                                    let instance_value =
                                        self.get_heap_mut().push(HeapValue::StructInst(instance));

                                    let new_frame = CallFrame {
                                        return_ip: self.ip,
                                        frame_pointer: self.locals.len(),
                                        stack_pointer: self.stack.len() - arg_count,
                                        instructions: Rc::clone(&init_func.code),
                                        function_name: format!("{}::init", struct_name),
                                        return_override: Some(instance_value),
                                        module_binding: init_func.module_binding.clone(),
                                        awaiting_task: None,
                                    };

                                    self.call_stack.push(new_frame);

                                    self.locals.push(instance_value);
                                    let args_start = self.stack.len() - arg_count;
                                    self.locals.extend(self.stack.drain(args_start..));
                                    self.ip = 0;
                                }
                                Some(_) => {
                                    return Err(WalrusError::StructMethodMustBeVmFunction {
                                        span,
                                        src: self.source_ref.source().into(),
                                        filename: self.source_ref.filename().into(),
                                    });
                                }
                                None => {
                                    if arg_count != 0 {
                                        // Preserve stack semantics: consume arguments for this failed call.
                                        self.pop_n(arg_count, opcode, span)?;
                                        return Err(WalrusError::InvalidArgCount {
                                            name: struct_name,
                                            expected: 0,
                                            got: arg_count,
                                            span,
                                            src: self.source_ref.source().into(),
                                            filename: self.source_ref.filename().into(),
                                        });
                                    }

                                    let instance = crate::structs::StructInstance::new(
                                        struct_name,
                                        struct_def_key,
                                    );
                                    let instance_value =
                                        self.get_heap_mut().push(HeapValue::StructInst(instance));
                                    self.push(instance_value);
                                }
                            }
                        }
                        _ => {
                            // Preserve stack semantics: consume arguments for this failed call.
                            self.pop_n(arg_count, opcode, span)?;
                            println!("func: {:?}", func);
                            return Err(WalrusError::NotCallable {
                                value: func.get_type().to_string(),
                                span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            });
                        }
                    }
                }
                Opcode::TailCall(args) => {
                    // Tail Call Optimization: reuse the current frame instead of pushing a new one.
                    // This prevents stack overflow for tail-recursive functions.
                    let arg_count = args as usize;
                    let func = self.pop(opcode, span)?;

                    if self.stack.len() < arg_count {
                        return Err(WalrusError::StackUnderflow {
                            op: opcode,
                            span,
                            src: self.source_ref.source().to_string(),
                            filename: self.source_ref.filename().to_string(),
                        });
                    }

                    match func {
                        Value::Function(key) => {
                            let func = self.get_heap().get_function(key)?;

                            match func {
                                WalrusFunction::Rust(func) => {
                                    let func = func.clone();
                                    let args = self.pop_n(arg_count, opcode, span)?;
                                    // Rust functions don't use call frames, so just call and return
                                    if args.len() != func.args {
                                        return Err(WalrusError::InvalidArgCount {
                                            name: func.name.clone(),
                                            expected: func.args,
                                            got: args.len(),
                                            span,
                                            src: self.source_ref.source().into(),
                                            filename: self.source_ref.filename().into(),
                                        });
                                    }
                                    let result = func.call(args, self.source_ref, span)?;

                                    // For a tail call, we need to return this result
                                    // Pop the current frame and push the result
                                    let frame = self
                                        .call_stack
                                        .pop()
                                        .expect("Call stack should never be empty on tail call");
                                    self.clear_exception_handlers_from_frame(self.call_stack.len());
                                    self.complete_task_on_frame_return(
                                        frame.awaiting_task,
                                        result,
                                    )?;

                                    if self.call_stack.is_empty() {
                                        return Ok(RunSignal::Returned(result));
                                    }

                                    self.locals.truncate(frame.frame_pointer);
                                    self.ip = frame.return_ip;
                                    self.push(result);
                                }
                                WalrusFunction::Native(native_fn) => {
                                    let native_fn = *native_fn;
                                    let args = self.pop_n(arg_count, opcode, span)?;
                                    let result = self.call_native(native_fn, args, span)?;

                                    // For a tail call, we need to return this result
                                    let frame = self
                                        .call_stack
                                        .pop()
                                        .expect("Call stack should never be empty on tail call");
                                    self.clear_exception_handlers_from_frame(self.call_stack.len());
                                    self.complete_task_on_frame_return(
                                        frame.awaiting_task,
                                        result,
                                    )?;

                                    if self.call_stack.is_empty() {
                                        return Ok(RunSignal::Returned(result));
                                    }

                                    self.locals.truncate(frame.frame_pointer);
                                    self.ip = frame.return_ip;
                                    self.push(result);
                                }
                                WalrusFunction::Vm(func) => {
                                    if arg_count != func.arity {
                                        return Err(WalrusError::InvalidArgCount {
                                            name: func.name.clone(),
                                            expected: func.arity,
                                            got: arg_count,
                                            span,
                                            src: self.source_ref.source().into(),
                                            filename: self.source_ref.filename().into(),
                                        });
                                    }

                                    if func.is_async {
                                        let args = self.pop_n(arg_count, opcode, span)?;
                                        let result = self.create_task_for_function_key(key, args);
                                        let frame = self.call_stack.pop().expect(
                                            "Call stack should never be empty on tail call",
                                        );
                                        self.clear_exception_handlers_from_frame(
                                            self.call_stack.len(),
                                        );
                                        self.complete_task_on_frame_return(
                                            frame.awaiting_task,
                                            result,
                                        )?;

                                        if self.call_stack.is_empty() {
                                            return Ok(RunSignal::Returned(result));
                                        }

                                        self.locals.truncate(frame.frame_pointer);
                                        self.ip = frame.return_ip;
                                        self.push(result);
                                        continue;
                                    }

                                    // Clone what we need before mutating self
                                    let new_instructions = Rc::clone(&func.code);
                                    let new_name = String::new();
                                    let new_module_binding = func.module_binding.clone();
                                    let frame_index = self.current_frame_index();
                                    self.clear_exception_handlers_from_frame(frame_index);

                                    // Get the current frame pointer before modifying
                                    let frame_pointer = self.frame_pointer();

                                    // Truncate locals to our frame pointer (clear current frame's locals)
                                    self.locals.truncate(frame_pointer);

                                    // Move the new arguments from operand stack to locals.
                                    let args_start = self.stack.len() - arg_count;
                                    self.locals.extend(self.stack.drain(args_start..));

                                    // Update the current frame in place (reuse it)
                                    // Keep the same return_ip and frame_pointer, just update instructions and name
                                    if let Some(current_frame) = self.call_stack.last_mut() {
                                        current_frame.instructions = new_instructions;
                                        current_frame.function_name = new_name;
                                        current_frame.return_override = None;
                                        current_frame.module_binding = new_module_binding;
                                    }

                                    // Reset IP to start of the new function
                                    self.ip = 0;
                                }
                                _ => {
                                    return Err(WalrusError::NodeFunctionNotSupportedInVm {
                                        span,
                                        src: self.source_ref.source().into(),
                                        filename: self.source_ref.filename().into(),
                                    });
                                }
                            }
                        }
                        Value::StructDef(struct_def_key) => {
                            let (struct_name, init_method) = {
                                let struct_def = self.get_heap().get_struct_def(struct_def_key)?;
                                (
                                    struct_def.name().to_string(),
                                    struct_def.get_method("init").cloned(),
                                )
                            };

                            match init_method {
                                Some(WalrusFunction::Vm(init_func)) => {
                                    let expected_without_self = init_func.arity.saturating_sub(1);
                                    if arg_count != expected_without_self {
                                        return Err(WalrusError::InvalidArgCount {
                                            name: format!("{}::init", struct_name),
                                            expected: expected_without_self,
                                            got: arg_count,
                                            span,
                                            src: self.source_ref.source().into(),
                                            filename: self.source_ref.filename().into(),
                                        });
                                    }

                                    let instance = crate::structs::StructInstance::new(
                                        struct_name.clone(),
                                        struct_def_key,
                                    );
                                    let instance_value =
                                        self.get_heap_mut().push(HeapValue::StructInst(instance));

                                    let new_instructions = Rc::clone(&init_func.code);
                                    let frame_index = self.current_frame_index();
                                    self.clear_exception_handlers_from_frame(frame_index);
                                    let frame_pointer = self.frame_pointer();
                                    self.locals.truncate(frame_pointer);
                                    self.locals.push(instance_value);

                                    let args_start = self.stack.len() - arg_count;
                                    self.locals.extend(self.stack.drain(args_start..));

                                    if let Some(current_frame) = self.call_stack.last_mut() {
                                        current_frame.instructions = new_instructions;
                                        current_frame.function_name =
                                            format!("{}::init", struct_name);
                                        current_frame.return_override = Some(instance_value);
                                        current_frame.module_binding =
                                            init_func.module_binding.clone();
                                    }

                                    self.ip = 0;
                                }
                                Some(_) => {
                                    return Err(WalrusError::StructMethodMustBeVmFunction {
                                        span,
                                        src: self.source_ref.source().into(),
                                        filename: self.source_ref.filename().into(),
                                    });
                                }
                                None => {
                                    if arg_count != 0 {
                                        // Preserve stack semantics: consume arguments for this failed call.
                                        self.pop_n(arg_count, opcode, span)?;
                                        return Err(WalrusError::InvalidArgCount {
                                            name: struct_name,
                                            expected: 0,
                                            got: arg_count,
                                            span,
                                            src: self.source_ref.source().into(),
                                            filename: self.source_ref.filename().into(),
                                        });
                                    }

                                    let instance = crate::structs::StructInstance::new(
                                        struct_name,
                                        struct_def_key,
                                    );
                                    let result =
                                        self.get_heap_mut().push(HeapValue::StructInst(instance));

                                    let frame = self
                                        .call_stack
                                        .pop()
                                        .expect("Call stack should never be empty on tail call");
                                    self.clear_exception_handlers_from_frame(self.call_stack.len());
                                    self.complete_task_on_frame_return(
                                        frame.awaiting_task,
                                        result,
                                    )?;

                                    if self.call_stack.is_empty() {
                                        return Ok(RunSignal::Returned(result));
                                    }

                                    self.locals.truncate(frame.frame_pointer);
                                    self.ip = frame.return_ip;
                                    self.push(result);
                                }
                            }
                        }
                        _ => {
                            // Preserve stack semantics: consume arguments for this failed call.
                            self.pop_n(arg_count, opcode, span)?;
                            return Err(WalrusError::NotCallable {
                                value: func.get_type().to_string(),
                                span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            });
                        }
                    }
                }
                Opcode::PushExceptionHandler(catch_ip) => {
                    let catch_ip = catch_ip as usize;
                    let handler = ExceptionHandler {
                        frame_index: self.current_frame_index(),
                        start_ip: self.ip,
                        end_ip: catch_ip,
                        catch_ip,
                        stack_len: self.stack.len(),
                        locals_len: self.locals.len(),
                    };
                    self.exception_handlers.push(handler);
                }
                Opcode::PopExceptionHandler => {
                    if let Some(handler) = self.exception_handlers.last().copied() {
                        if handler.frame_index == self.current_frame_index() {
                            self.exception_handlers.pop();
                        }
                    }
                }
                Opcode::Throw => {
                    let value = self.pop(opcode, span)?;
                    self.throw_value(value, span)?;
                }
                // Specialized integer arithmetic (hot path - skips type checking)
                // SAFETY: Compiler guarantees stack has operands and both are integers
                Opcode::AddInt => {
                    let b = self.pop_unchecked();
                    let a = self.pop_unchecked();
                    if let (Value::Int(a), Value::Int(b)) = (a, b) {
                        self.push(Value::Int(a + b));
                    } else {
                        // Fallback for safety (shouldn't happen with correct compilation)
                        return Err(WalrusError::TypeMismatch {
                            expected: "int and int".to_string(),
                            found: format!("{} and {}", a.get_type(), b.get_type()),
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        });
                    }
                }
                Opcode::SubtractInt => {
                    let b = self.pop_unchecked();
                    let a = self.pop_unchecked();
                    if let (Value::Int(a), Value::Int(b)) = (a, b) {
                        self.push(Value::Int(a - b));
                    } else {
                        return Err(WalrusError::TypeMismatch {
                            expected: "int and int".to_string(),
                            found: format!("{} and {}", a.get_type(), b.get_type()),
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        });
                    }
                }
                Opcode::LessInt => {
                    let b = self.pop_unchecked();
                    let a = self.pop_unchecked();
                    if let (Value::Int(a), Value::Int(b)) = (a, b) {
                        self.push(Value::Bool(a < b));
                    } else {
                        return Err(WalrusError::TypeMismatch {
                            expected: "int and int".to_string(),
                            found: format!("{} and {}", a.get_type(), b.get_type()),
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        });
                    }
                }
                Opcode::Add => {
                    let b = self.pop_unchecked();
                    let a = self.pop_unchecked();

                    // JIT TYPE PROFILING: Track operand types for arithmetic
                    if profiling_enabled {
                        let type_a = WalrusType::from_value(&a);
                        let type_b = WalrusType::from_value(&b);
                        // Record both operand types at this IP
                        self.type_profile.observe(self.ip - 1, type_a);
                        self.type_profile.observe(self.ip - 1, type_b);
                    }

                    match (a, b) {
                        (Value::Int(a), Value::Int(b)) => {
                            self.push(Value::Int(a + b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Float(FloatOrd(a + b)));
                        }
                        (Value::Int(a), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Float(FloatOrd(a as f64 + b)));
                        }
                        (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                            self.push(Value::Float(FloatOrd(a + b as f64)));
                        }
                        (Value::String(a), Value::String(b)) => {
                            let a = self.get_heap().get_string(a)?;
                            let b = self.get_heap().get_string(b)?;

                            let mut s = String::with_capacity(a.len() + b.len());
                            s.push_str(a);
                            s.push_str(b);

                            let value = self.get_heap_mut().push(HeapValue::String(&s));
                            self.push(value);
                        }
                        (Value::List(a), Value::List(b)) => {
                            let mut a = self.get_heap().get_list(a)?.to_vec();
                            let b = self.get_heap().get_list(b)?;
                            a.extend(b);

                            let value = self.get_heap_mut().push(HeapValue::List(a));
                            self.push(value);
                        }
                        (Value::Dict(a), Value::Dict(b)) => {
                            let mut a = self.get_heap().get_dict(a)?.clone();
                            let b = self.get_heap().get_dict(b)?;
                            a.extend(b);

                            let value = self.get_heap_mut().push(HeapValue::Dict(a));
                            self.push(value);
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Subtract => {
                    let b = self.pop_unchecked();
                    let a = self.pop_unchecked();

                    match (a, b) {
                        (Value::Int(a), Value::Int(b)) => {
                            self.push(Value::Int(a - b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Float(FloatOrd(a - b)));
                        }
                        (Value::Int(a), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Float(FloatOrd(a as f64 - b)));
                        }
                        (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                            self.push(Value::Float(FloatOrd(a - b as f64)));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Multiply => {
                    let b = self.pop_unchecked();
                    let a = self.pop_unchecked();

                    match (a, b) {
                        (Value::Int(a), Value::Int(b)) => {
                            self.push(Value::Int(a * b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Float(FloatOrd(a * b)));
                        }
                        (Value::Int(a), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Float(FloatOrd(a as f64 * b)));
                        }
                        (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                            self.push(Value::Float(FloatOrd(a * b as f64)));
                        }
                        (Value::List(a), Value::Int(b)) | (Value::Int(b), Value::List(a)) => {
                            let a = self.get_heap().get_list(a)?;
                            let mut list = Vec::with_capacity(a.len() * b as usize);

                            for _ in 0..b {
                                list.extend(a);
                            }

                            let value = self.get_heap_mut().push(HeapValue::List(list));
                            self.push(value);
                        }
                        (Value::String(a), Value::Int(b)) | (Value::Int(b), Value::String(a)) => {
                            let a = self.get_heap().get_string(a)?;
                            let mut s = String::with_capacity(a.len() * b as usize);

                            for _ in 0..b {
                                s.push_str(a);
                            }

                            let value = self.get_heap_mut().push(HeapValue::String(&s));
                            self.push(value);
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Divide => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::Int(a), Value::Int(b)) => {
                            if b == 0 {
                                return Err(WalrusError::DivisionByZero {
                                    span,
                                    src: self.source_ref.source().to_string(),
                                    filename: self.source_ref.filename().to_string(),
                                });
                            }
                            self.push(Value::Int(a / b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Float(FloatOrd(a / b)));
                        }
                        (Value::Int(a), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Float(FloatOrd(a as f64 / b)));
                        }
                        (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                            self.push(Value::Float(FloatOrd(a / b as f64)));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Power => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::Int(a), Value::Int(b)) => {
                            if b < 0 {
                                // Convert to float for negative exponents
                                let result = (a as f64).powf(b as f64);
                                self.push(Value::Float(FloatOrd(result)));
                            } else {
                                self.push(Value::Int(a.pow(b as u32)));
                            }
                        }
                        (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Float(FloatOrd(a.powf(b))));
                        }
                        (Value::Int(a), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Float(FloatOrd((a as f64).powf(b))));
                        }
                        (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                            self.push(Value::Float(FloatOrd(a.powf(b as f64))));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Modulo => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::Int(a), Value::Int(b)) => {
                            if b == 0 {
                                return Err(WalrusError::DivisionByZero {
                                    span,
                                    src: self.source_ref.source().to_string(),
                                    filename: self.source_ref.filename().to_string(),
                                });
                            }
                            self.push(Value::Int(a % b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Float(FloatOrd(a % b)));
                        }
                        (Value::Int(a), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Float(FloatOrd(a as f64 % b)));
                        }
                        (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                            self.push(Value::Float(FloatOrd(a % b as f64)));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Negate => {
                    let a = self.pop(opcode, span)?;

                    match a {
                        Value::Int(a) => {
                            self.push(Value::Int(-a));
                        }
                        Value::Float(FloatOrd(a)) => {
                            self.push(Value::Float(FloatOrd(-a)));
                        }
                        _ => return Err(self.construct_err(opcode, a, None, span)),
                    }
                }
                Opcode::Not => {
                    let a = self.pop(opcode, span)?;
                    self.push(Value::Bool(!self.get_heap().is_truthy(a)?));
                }
                Opcode::And => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;
                    self.push(Value::Bool(
                        self.get_heap().is_truthy(a)? && self.get_heap().is_truthy(b)?,
                    ));
                }
                Opcode::Or => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;
                    self.push(Value::Bool(
                        self.get_heap().is_truthy(a)? || self.get_heap().is_truthy(b)?,
                    ));
                }
                Opcode::Equal => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::List(a), Value::List(b)) => {
                            let a = self.get_heap().get_list(a)?;
                            let b = self.get_heap().get_list(b)?;

                            self.push(Value::Bool(a == b));
                        }
                        (Value::Dict(a), Value::Dict(b)) => {
                            let a = self.get_heap().get_dict(a)?;
                            let b = self.get_heap().get_dict(b)?;

                            self.push(Value::Bool(a == b));
                        }
                        (Value::Module(a), Value::Module(b)) => {
                            let a = self.get_heap().get_module(a)?;
                            let b = self.get_heap().get_module(b)?;

                            self.push(Value::Bool(a == b));
                        }
                        (Value::Function(a), Value::Function(b)) => {
                            let a_func = self.get_heap().get_function(a)?;
                            let b_func = self.get_heap().get_function(b)?;

                            self.push(Value::Bool(a_func == b_func));
                        }
                        (Value::Int(a), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Bool(a as f64 == b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                            self.push(Value::Bool(a == b as f64));
                        }
                        _ => self.push(Value::Bool(a == b)),
                    }
                }
                Opcode::NotEqual => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::List(a), Value::List(b)) => {
                            let a = self.get_heap().get_list(a)?;
                            let b = self.get_heap().get_list(b)?;

                            self.push(Value::Bool(a != b));
                        }
                        (Value::Dict(a), Value::Dict(b)) => {
                            let a = self.get_heap().get_dict(a)?;
                            let b = self.get_heap().get_dict(b)?;

                            self.push(Value::Bool(a != b));
                        }
                        (Value::Module(a), Value::Module(b)) => {
                            let a = self.get_heap().get_module(a)?;
                            let b = self.get_heap().get_module(b)?;

                            self.push(Value::Bool(a != b));
                        }
                        (Value::Function(a), Value::Function(b)) => {
                            let a_func = self.get_heap().get_function(a)?;
                            let b_func = self.get_heap().get_function(b)?;

                            self.push(Value::Bool(a_func != b_func));
                        }
                        (Value::Int(a), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Bool(a as f64 != b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                            self.push(Value::Bool(a != b as f64));
                        }
                        _ => self.push(Value::Bool(a != b)),
                    }
                }
                Opcode::Greater => {
                    let b = self.pop_unchecked();
                    let a = self.pop_unchecked();

                    match (a, b) {
                        (Value::Int(a), Value::Int(b)) => {
                            self.push(Value::Bool(a > b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Bool(a > b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                            self.push(Value::Bool(a > b as f64));
                        }
                        (Value::Int(a), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Bool((a as f64) > b));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::GreaterEqual => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::Int(a), Value::Int(b)) => {
                            self.push(Value::Bool(a >= b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Bool(a >= b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                            self.push(Value::Bool(a >= b as f64));
                        }
                        (Value::Int(a), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Bool((a as f64) >= b));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Less => {
                    let b = self.pop_unchecked();
                    let a = self.pop_unchecked();

                    match (a, b) {
                        (Value::Int(a), Value::Int(b)) => {
                            self.push(Value::Bool(a < b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Bool(a < b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                            self.push(Value::Bool(a < b as f64));
                        }
                        (Value::Int(a), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Bool((a as f64) < b));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::LessEqual => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::Int(a), Value::Int(b)) => {
                            self.push(Value::Bool(a <= b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Bool(a <= b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                            self.push(Value::Bool(a <= b as f64));
                        }
                        (Value::Int(a), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Bool((a as f64) <= b));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Index => {
                    let b = self.pop_unchecked();
                    let a = self.pop_unchecked();

                    // Fast path for the hottest case in numeric code: list[int]
                    if let (Value::List(list_key), Value::Int(idx)) = (a, b) {
                        let list = self.get_heap().get_list(list_key)?;
                        if idx >= 0 {
                            let idx_usize = idx as usize;
                            if idx_usize < list.len() {
                                // SAFETY: bounds checked above
                                self.push(unsafe { *list.get_unchecked(idx_usize) });
                                continue;
                            }
                        }

                        let mut normalized = idx;
                        let original = idx;
                        if normalized < 0 {
                            normalized += list.len() as i64;
                        }

                        if normalized < 0 || normalized >= list.len() as i64 {
                            return Err(WalrusError::IndexOutOfBounds {
                                index: original,
                                len: list.len(),
                                span,
                                src: self.source_ref.source().to_string(),
                                filename: self.source_ref.filename().to_string(),
                            });
                        }

                        self.push(list[normalized as usize]);
                        continue;
                    }

                    match (a, b) {
                        (Value::List(_), Value::Int(_)) => unreachable!(),
                        (Value::String(a), Value::Int(b)) => {
                            let a = self.get_heap().get_string(a)?;
                            let char_len = a.chars().count();
                            let original = b;

                            let Some(char_idx) = Self::normalize_index(b, char_len) else {
                                return Err(WalrusError::IndexOutOfBounds {
                                    index: original,
                                    len: char_len,
                                    span,
                                    src: self.source_ref.source().to_string(),
                                    filename: self.source_ref.filename().to_string(),
                                });
                            };

                            let res = a
                                .chars()
                                .nth(char_idx)
                                .map(|ch| ch.to_string())
                                .ok_or_else(|| WalrusError::IndexOutOfBounds {
                                    index: original,
                                    len: char_len,
                                    span,
                                    src: self.source_ref.source().to_string(),
                                    filename: self.source_ref.filename().to_string(),
                                })?;
                            let value = self.get_heap_mut().push(HeapValue::String(&res));

                            self.push(value);
                        }
                        (Value::Dict(a), b) => {
                            let a = self.get_heap().get_dict(a)?;

                            if let Some(value) = a.get(&b) {
                                self.push(*value);
                            } else {
                                // fixme: sometimes this is thrown even when the key exists
                                // this is because it compares the key, not the value, which
                                // means while the key may be different, the contents of the
                                // key may be the same
                                let b_str = b.stringify()?;

                                return Err(WalrusError::KeyNotFound {
                                    key: b_str,
                                    span: span,
                                    src: self.source_ref.source().to_string(),
                                    filename: self.source_ref.filename().to_string(),
                                });
                            }
                        }
                        (Value::Module(a), b) => {
                            let a = self.get_heap().get_module(a)?;

                            if let Some(value) = a.get(&b) {
                                self.push(*value);
                            } else {
                                let b_str = b.stringify()?;

                                return Err(WalrusError::KeyNotFound {
                                    key: b_str,
                                    span,
                                    src: self.source_ref.source().to_string(),
                                    filename: self.source_ref.filename().to_string(),
                                });
                            }
                        }
                        (Value::String(a), Value::Range(range)) => {
                            let a = self.get_heap().get_string(a)?;
                            let a_len = a.chars().count();

                            let start = if range.start < 0 {
                                a_len as i64 + range.start + 1
                            } else {
                                range.start
                            };

                            let end = if range.end < 0 {
                                a_len as i64 + range.end + 1
                            } else {
                                range.end
                            };

                            if start < 0
                                || end < 0
                                || start as usize > a_len
                                || end as usize > a_len
                            {
                                return Err(WalrusError::IndexOutOfBounds {
                                    index: range.start,
                                    len: a_len,
                                    span,
                                    src: self.source_ref.source().to_string(),
                                    filename: self.source_ref.filename().to_string(),
                                });
                            }

                            if start > end {
                                return Err(WalrusError::InvalidRange {
                                    start: range.start,
                                    end: range.end,
                                    span,
                                    src: self.source_ref.source().to_string(),
                                    filename: self.source_ref.filename().to_string(),
                                });
                            }

                            let start = start as usize;
                            let end = end as usize;

                            let start_byte =
                                Self::char_to_byte_offset(a, start).ok_or_else(|| {
                                    WalrusError::IndexOutOfBounds {
                                        index: range.start,
                                        len: a_len,
                                        span,
                                        src: self.source_ref.source().to_string(),
                                        filename: self.source_ref.filename().to_string(),
                                    }
                                })?;
                            let end_byte = Self::char_to_byte_offset(a, end).ok_or_else(|| {
                                WalrusError::IndexOutOfBounds {
                                    index: range.end,
                                    len: a_len,
                                    span,
                                    src: self.source_ref.source().to_string(),
                                    filename: self.source_ref.filename().to_string(),
                                }
                            })?;

                            let res = a[start_byte..end_byte].to_string();
                            let value = self.get_heap_mut().push(HeapValue::String(&res));

                            self.push(value);
                        }
                        (Value::List(a), Value::Range(range)) => {
                            let a = self.get_heap().get_list(a)?;
                            let a_len = a.len();

                            let start = if range.start < 0 {
                                a_len as i64 + range.start + 1
                            } else {
                                range.start
                            };

                            let end = if range.end < 0 {
                                a_len as i64 + range.end + 1
                            } else {
                                range.end
                            };

                            if start < 0
                                || end < 0
                                || start as usize > a_len
                                || end as usize > a_len
                            {
                                return Err(WalrusError::IndexOutOfBounds {
                                    index: range.start,
                                    len: a.len(),
                                    span,
                                    src: self.source_ref.source().to_string(),
                                    filename: self.source_ref.filename().to_string(),
                                });
                            }

                            if start > end {
                                return Err(WalrusError::InvalidRange {
                                    start: range.start,
                                    end: range.end,
                                    span,
                                    src: self.source_ref.source().to_string(),
                                    filename: self.source_ref.filename().to_string(),
                                });
                            }

                            let res = a[start as usize..end as usize].to_vec();
                            let value = self.get_heap_mut().push(HeapValue::List(res));

                            self.push(value);
                        }
                        // maybe add dict range indexing later
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::StoreIndex => {
                    // Stack: [object, index, value]
                    let value = self.pop_unchecked();
                    let index = self.pop_unchecked();
                    let object = self.pop_unchecked();

                    // Fast path for hottest assignment case: list[int] = value
                    if let (Value::List(list_key), Value::Int(idx)) = (object, index) {
                        let list = self.get_heap_mut().get_mut_list(list_key)?;
                        if idx >= 0 {
                            let idx_usize = idx as usize;
                            if idx_usize < list.len() {
                                // SAFETY: bounds checked above
                                unsafe {
                                    *list.get_unchecked_mut(idx_usize) = value;
                                }
                                self.push(Value::Void);
                                continue;
                            }
                        }

                        let mut normalized = idx;
                        let original = idx;
                        if normalized < 0 {
                            normalized += list.len() as i64;
                        }

                        if normalized < 0 || normalized >= list.len() as i64 {
                            return Err(WalrusError::IndexOutOfBounds {
                                index: original,
                                len: list.len(),
                                span,
                                src: self.source_ref.source().to_string(),
                                filename: self.source_ref.filename().to_string(),
                            });
                        }

                        list[normalized as usize] = value;
                        self.push(Value::Void);
                        continue;
                    }

                    match (object, index) {
                        (Value::List(_), Value::Int(_)) => unreachable!(),
                        (Value::Dict(dict_key), key) => {
                            let dict = self.get_heap_mut().get_mut_dict(dict_key)?;
                            dict.insert(key, value);
                            self.push(Value::Void);
                        }
                        (Value::StructInst(inst_key), Value::String(member_key)) => {
                            let member_name = self.get_heap().get_string(member_key)?.to_string();
                            let inst = self.get_heap_mut().get_mut_struct_inst(inst_key)?;
                            inst.set_field(member_name, value);
                            self.push(Value::Void);
                        }
                        _ => {
                            return Err(WalrusError::InvalidIndexType {
                                non_indexable: object.get_type().to_string(),
                                index_type: index.get_type().to_string(),
                                span,
                                src: self.source_ref.source().to_string(),
                                filename: self.source_ref.filename().to_string(),
                            });
                        }
                    }
                }
                Opcode::Print => {
                    let a = self.pop(opcode, span)?;
                    print!("{}", self.stringify_value(a)?);
                }
                Opcode::Println => {
                    let a = self.pop(opcode, span)?;
                    println!("{}", self.stringify_value(a)?);
                }
                Opcode::Str => {
                    let a = self.pop(opcode, span)?;
                    let s = self.stringify_value(a)?;
                    let value = self.get_heap_mut().push(HeapValue::String(&s));
                    self.push(value);
                }
                // Import system - returns a module namespace with native functions
                Opcode::Import => {
                    let module_name = self.pop(opcode, span)?;
                    match module_name {
                        Value::String(name_key) => {
                            let name_str = self.get_heap().get_string(name_key)?;
                            if let Some(functions) = crate::stdlib::get_module_functions(name_str) {
                                // Build a module where keys are function names and values are the functions
                                let mut dict = FxHashMap::default();
                                for native_fn in functions {
                                    let key = self
                                        .get_heap_mut()
                                        .push(HeapValue::String(native_fn.name()));
                                    let func = self.get_heap_mut().push(HeapValue::Function(
                                        WalrusFunction::Native(native_fn),
                                    ));
                                    dict.insert(key, func);
                                }
                                let module = self.get_heap_mut().push(HeapValue::Module(dict));
                                self.push(module);
                            } else {
                                let module = crate::program::load_module_for_vm(
                                    name_str,
                                    self.source_ref.filename(),
                                )?;
                                self.push(module);
                            }
                        }
                        _ => {
                            return Err(WalrusError::TypeMismatch {
                                expected: "string".to_string(),
                                found: module_name.get_type().to_string(),
                                span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            });
                        }
                    }
                }
                Opcode::Await => {
                    let awaited = self.pop(opcode, span)?;
                    let task_key = match awaited {
                        Value::Task(task_key) => task_key,
                        other => {
                            return Err(WalrusError::TypeMismatch {
                                expected: "task".to_string(),
                                found: other.get_type().to_string(),
                                span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            });
                        }
                    };
                    match self.poll_task_resolution(task_key)? {
                        TaskResolution::Ready(value) => {
                            self.push(value);
                        }
                        TaskResolution::Failed(error_value) => {
                            self.throw_value(error_value, span)?;
                            continue 'vm;
                        }
                        TaskResolution::Cancelled => {
                            let cancelled = self.cancelled_task_error_value();
                            self.throw_value(cancelled, span)?;
                            continue 'vm;
                        }
                        TaskResolution::Pending => {
                            self.ip -= 1;
                            self.push(Value::Task(task_key));
                            return Ok(RunSignal::Suspended(task_key));
                        }
                    }
                }
                Opcode::Return => {
                    let mut return_value = self.pop(opcode, span)?;

                    // Pop the current call frame
                    let frame = self
                        .call_stack
                        .pop()
                        .expect("Call stack should never be empty on return");
                    self.clear_exception_handlers_from_frame(self.call_stack.len());

                    if let Some(override_value) = frame.return_override {
                        return_value = override_value;
                    }
                    self.complete_task_on_frame_return(frame.awaiting_task, return_value)?;

                    // If this was the last frame (main), return the value
                    if self.call_stack.is_empty() {
                        return Ok(RunSignal::Returned(return_value));
                    }

                    // Truncate locals back to the frame pointer (cleanup this frame's locals)
                    self.locals.truncate(frame.frame_pointer);

                    // Truncate operand stack back to where it was at call time
                    // This cleans up any leftover values (e.g., iterators from loops)
                    self.stack.truncate(frame.stack_pointer);

                    // Restore the instruction pointer to where we should continue
                    self.ip = frame.return_ip;

                    // Push return value onto operand stack for the caller
                    self.push(return_value);
                }
                // Stack manipulation opcodes
                Opcode::Dup => {
                    let a = self.pop(opcode, span)?;
                    self.push(a);
                    self.push(a);
                }
                Opcode::Swap => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;
                    self.push(b);
                    self.push(a);
                }
                Opcode::Pop2 => {
                    self.pop(opcode, span)?;
                    self.pop(opcode, span)?;
                }
                Opcode::Pop3 => {
                    self.pop(opcode, span)?;
                    self.pop(opcode, span)?;
                    self.pop(opcode, span)?;
                }
                Opcode::Nop => {}
                Opcode::MakeStruct => {
                    // Pop struct definition from stack and create an instance
                    let struct_def_value = self.pop(opcode, span)?;

                    if let Value::StructDef(struct_def_key) = struct_def_value {
                        let struct_def = self.get_heap().get_struct_def(struct_def_key)?;
                        let struct_name = struct_def.name().to_string();

                        // Create a new struct instance
                        let instance =
                            crate::structs::StructInstance::new(struct_name, struct_def_key);
                        let instance_value =
                            self.get_heap_mut().push(HeapValue::StructInst(instance));

                        self.push(instance_value);
                    } else {
                        return Err(WalrusError::TypeMismatch {
                            expected: "struct definition".to_string(),
                            found: struct_def_value.get_type().to_string(),
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        });
                    }
                }
                Opcode::GetMethod => {
                    // Pop member name (string) and object from stack
                    let member_name_value = self.pop(opcode, span)?;
                    let object_value = self.pop(opcode, span)?;

                    match (member_name_value, object_value) {
                        // Struct method access
                        (Value::String(method_name_sym), Value::StructDef(struct_def_key)) => {
                            let method_name = self.get_heap().get_string(method_name_sym)?;
                            let method_clone = {
                                let struct_def = self.get_heap().get_struct_def(struct_def_key)?;
                                if let Some(method) = struct_def.get_method(method_name) {
                                    method.clone()
                                } else {
                                    return Err(WalrusError::MethodNotFound {
                                        type_name: struct_def.name().to_string(),
                                        method: method_name.to_string(),
                                        span,
                                        src: self.source_ref.source().into(),
                                        filename: self.source_ref.filename().into(),
                                    });
                                }
                            };

                            // Push the method as a function value
                            let func_value =
                                self.get_heap_mut().push(HeapValue::Function(method_clone));
                            self.push(func_value);
                        }
                        // Struct instance member access: field first, then method lookup
                        (Value::String(member_name_sym), Value::StructInst(struct_inst_key)) => {
                            enum ResolvedInstanceMember {
                                Field(Value),
                                Method(WalrusFunction),
                            }

                            let member_name =
                                self.get_heap().get_string(member_name_sym)?.to_string();
                            let resolved = {
                                let heap = self.get_heap();
                                let inst = heap.get_struct_inst(struct_inst_key)?;

                                if let Some(value) = inst.get_field(&member_name).copied() {
                                    ResolvedInstanceMember::Field(value)
                                } else {
                                    let struct_def = heap.get_struct_def(inst.struct_def())?;
                                    if let Some(method) = struct_def.get_method(&member_name) {
                                        ResolvedInstanceMember::Method(method.clone())
                                    } else {
                                        return Err(WalrusError::MemberNotFound {
                                            type_name: inst.struct_name().to_string(),
                                            member: member_name,
                                            span,
                                            src: self.source_ref.source().into(),
                                            filename: self.source_ref.filename().into(),
                                        });
                                    }
                                }
                            };

                            match resolved {
                                ResolvedInstanceMember::Field(value) => self.push(value),
                                ResolvedInstanceMember::Method(method) => {
                                    let func_value =
                                        self.get_heap_mut().push(HeapValue::Function(method));
                                    self.push(func_value);
                                }
                            }
                        }
                        (Value::String(member_name_sym), Value::Module(module_key)) => {
                            let module = self.get_heap().get_module(module_key)?;
                            if let Some(&value) = module.get(&Value::String(member_name_sym)) {
                                self.push(value);
                            } else {
                                let member_name = self.get_heap().get_string(member_name_sym)?;
                                return Err(WalrusError::MemberNotFound {
                                    type_name: "module".to_string(),
                                    member: member_name.to_string(),
                                    span,
                                    src: self.source_ref.source().into(),
                                    filename: self.source_ref.filename().into(),
                                });
                            }
                        }
                        (Value::String(member_name_sym), Value::Dict(dict_key)) => {
                            let dict = self.get_heap().get_dict(dict_key)?;
                            if let Some(&value) = dict.get(&Value::String(member_name_sym)) {
                                self.push(value);
                            } else {
                                let member_name = self.get_heap().get_string(member_name_sym)?;
                                return Err(WalrusError::MemberNotFound {
                                    type_name: "dict".to_string(),
                                    member: member_name.to_string(),
                                    span,
                                    src: self.source_ref.source().into(),
                                    filename: self.source_ref.filename().into(),
                                });
                            }
                        }
                        (member, object) => {
                            return Err(WalrusError::InvalidMemberAccessTarget {
                                object_type: object.get_type().to_string(),
                                member_type: member.get_type().to_string(),
                                span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            });
                        }
                    }
                }
                Opcode::CallMethod(arg_count) => {
                    // Stack layout: [object, arg1, arg2, ..., argN, method_name]
                    // Pop method name first
                    let method_name_val = self.pop(opcode, span)?;
                    let method_name_sym = match method_name_val {
                        Value::String(sym) => sym,
                        other => {
                            return Err(WalrusError::TypeMismatch {
                                expected: "string".to_string(),
                                found: other.get_type().to_string(),
                                span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            });
                        }
                    };

                    let arg_count = arg_count as usize;

                    // Fast path for struct VM methods:
                    // avoid creating an args Vec and move arguments directly into locals.
                    if self.stack.len() >= arg_count + 1 {
                        let object_idx = self.stack.len() - arg_count - 1;
                        match self.stack[object_idx] {
                            Value::StructDef(key) => {
                                let method_name =
                                    self.get_heap().get_string(method_name_sym)?.to_string();
                                let (method_arity, method_code, method_binding) = {
                                    let struct_def = self.get_heap().get_struct_def(key)?;
                                    match struct_def.get_method(&method_name) {
                                        Some(WalrusFunction::Vm(func)) => (
                                            func.arity,
                                            Rc::clone(&func.code),
                                            func.module_binding.clone(),
                                        ),
                                        Some(_) => {
                                            return Err(
                                                WalrusError::StructMethodMustBeVmFunction {
                                                    span,
                                                    src: self.source_ref.source().into(),
                                                    filename: self.source_ref.filename().into(),
                                                },
                                            );
                                        }
                                        None => {
                                            return Err(WalrusError::MethodNotFound {
                                                type_name: struct_def.name().to_string(),
                                                method: method_name,
                                                span,
                                                src: self.source_ref.source().into(),
                                                filename: self.source_ref.filename().into(),
                                            });
                                        }
                                    }
                                };

                                if arg_count != method_arity {
                                    return Err(WalrusError::InvalidArgCount {
                                        name: method_name,
                                        expected: method_arity,
                                        got: arg_count,
                                        span,
                                        src: self.source_ref.source().into(),
                                        filename: self.source_ref.filename().into(),
                                    });
                                }

                                let new_frame = CallFrame {
                                    return_ip: self.ip,
                                    frame_pointer: self.locals.len(),
                                    stack_pointer: object_idx,
                                    instructions: method_code,
                                    function_name: String::new(),
                                    return_override: None,
                                    module_binding: method_binding,
                                    awaiting_task: None,
                                };

                                self.call_stack.push(new_frame);

                                let args_start = object_idx + 1;
                                self.locals.extend(self.stack.drain(args_start..));
                                self.stack.truncate(object_idx);

                                self.ip = 0;
                                continue; // Function frame takes control
                            }
                            Value::StructInst(inst_key) => {
                                let method_name =
                                    self.get_heap().get_string(method_name_sym)?.to_string();
                                let (method, type_name) = {
                                    let heap = self.get_heap();
                                    let inst = heap.get_struct_inst(inst_key)?;
                                    let struct_def = heap.get_struct_def(inst.struct_def())?;
                                    (
                                        struct_def.get_method(&method_name).cloned(),
                                        struct_def.name().to_string(),
                                    )
                                };

                                let (method_arity, method_code, method_binding) = match method {
                                    Some(WalrusFunction::Vm(func)) => (
                                        func.arity,
                                        Rc::clone(&func.code),
                                        func.module_binding.clone(),
                                    ),
                                    Some(_) => {
                                        return Err(WalrusError::StructMethodMustBeVmFunction {
                                            span,
                                            src: self.source_ref.source().into(),
                                            filename: self.source_ref.filename().into(),
                                        });
                                    }
                                    None => {
                                        // No method found — check if there's a field holding a callable function
                                        let field_func = {
                                            let inst = self.get_heap().get_struct_inst(inst_key)?;
                                            inst.get_field(&method_name).copied()
                                        };

                                        if let Some(Value::Function(func_key)) = field_func {
                                            // Field holds a function — call it (no self prepended)
                                            let args = self.pop_n(arg_count, opcode, span)?;
                                            let _ = self.pop(opcode, span)?; // pop object

                                            let func =
                                                self.get_heap().get_function(func_key)?.clone();
                                            if let Some(result) =
                                                self.call_exported_function(func, args, span)?
                                            {
                                                self.push(result);
                                            }
                                            continue;
                                        }

                                        return Err(WalrusError::MemberNotFound {
                                            type_name,
                                            member: method_name,
                                            span,
                                            src: self.source_ref.source().into(),
                                            filename: self.source_ref.filename().into(),
                                        });
                                    }
                                };

                                let expected_without_self = method_arity.saturating_sub(1);
                                if arg_count != expected_without_self {
                                    return Err(WalrusError::InvalidArgCount {
                                        name: format!("{}::{}", type_name, method_name),
                                        expected: expected_without_self,
                                        got: arg_count,
                                        span,
                                        src: self.source_ref.source().into(),
                                        filename: self.source_ref.filename().into(),
                                    });
                                }

                                let new_frame = CallFrame {
                                    return_ip: self.ip,
                                    frame_pointer: self.locals.len(),
                                    stack_pointer: object_idx,
                                    instructions: method_code,
                                    function_name: String::new(),
                                    return_override: None,
                                    module_binding: method_binding,
                                    awaiting_task: None,
                                };

                                self.call_stack.push(new_frame);

                                self.locals.push(Value::StructInst(inst_key));
                                let args_start = object_idx + 1;
                                self.locals.extend(self.stack.drain(args_start..));
                                self.stack.truncate(object_idx);

                                self.ip = 0;
                                continue; // Function frame takes control
                            }
                            _ => {}
                        }
                    }

                    // Pop arguments
                    let args = self.pop_n(arg_count, opcode, span)?;

                    // Pop the object
                    let object = self.pop(opcode, span)?;

                    // Borrow source info for errors (allocate only when constructing errors)
                    let src = self.source_ref.source();
                    let filename = self.source_ref.filename();

                    // Dispatch based on object type
                    let result = match object {
                        Value::List(key) => methods::dispatch_list_method(
                            self.get_heap_mut(),
                            key,
                            method_name_sym,
                            args,
                            span,
                            src,
                            filename,
                        )?,
                        Value::String(key) => methods::dispatch_string_method(
                            self.get_heap_mut(),
                            key,
                            method_name_sym,
                            args,
                            span,
                            src,
                            filename,
                        )?,
                        Value::Dict(key) => {
                            let method_key = Value::String(method_name_sym);
                            let dict = self.get_heap().get_dict(key)?;

                            if let Some(func_val) = dict.get(&method_key).copied() {
                                let Value::Function(func_key) = func_val else {
                                    return Err(WalrusError::NotCallable {
                                        value: func_val.get_type().to_string(),
                                        span,
                                        src: self.source_ref.source().into(),
                                        filename: self.source_ref.filename().into(),
                                    });
                                };

                                let function = self.get_heap().get_function(func_key)?.clone();
                                if let Some(result) =
                                    self.call_exported_function(function, args, span)?
                                {
                                    self.push(result);
                                }
                                continue;
                            }

                            methods::dispatch_dict_method(
                                self.get_heap_mut(),
                                key,
                                method_name_sym,
                                args,
                                span,
                                src,
                                filename,
                            )?
                        }
                        Value::Module(key) => {
                            let method_key = Value::String(method_name_sym);
                            let module = self.get_heap().get_module(key)?;
                            let Some(func_val) = module.get(&method_key).copied() else {
                                let method_name = self.get_heap().get_string(method_name_sym)?;
                                return Err(WalrusError::MemberNotFound {
                                    type_name: "module".to_string(),
                                    member: method_name.to_string(),
                                    span,
                                    src: self.source_ref.source().into(),
                                    filename: self.source_ref.filename().into(),
                                });
                            };

                            let Value::Function(func_key) = func_val else {
                                return Err(WalrusError::NotCallable {
                                    value: func_val.get_type().to_string(),
                                    span,
                                    src: self.source_ref.source().into(),
                                    filename: self.source_ref.filename().into(),
                                });
                            };

                            let function = self.get_heap().get_function(func_key)?.clone();
                            if let Some(result) =
                                self.call_exported_function(function, args, span)?
                            {
                                self.push(result);
                            }
                            continue;
                        }
                        Value::Iter(iter_key) => {
                            let method_name =
                                self.get_heap().get_string(method_name_sym)?.to_string();
                            match method_name.as_str() {
                                "next" => {
                                    if !args.is_empty() {
                                        return Err(WalrusError::InvalidArgCount {
                                            name: "next".to_string(),
                                            expected: 0,
                                            got: args.len(),
                                            span,
                                            src: self.source_ref.source().into(),
                                            filename: self.source_ref.filename().into(),
                                        });
                                    }

                                    unsafe {
                                        let mut ptr = NonNull::from(self.get_heap_mut());
                                        let iter = ptr.as_mut().get_mut_iter(iter_key)?;
                                        iter.next(self.get_heap_mut()).unwrap_or(Value::Void)
                                    }
                                }
                                _ => {
                                    return Err(WalrusError::MethodNotFound {
                                        type_name: "iter".to_string(),
                                        method: method_name,
                                        span,
                                        src: self.source_ref.source().into(),
                                        filename: self.source_ref.filename().into(),
                                    });
                                }
                            }
                        }
                        Value::StructDef(key) => {
                            // For struct definitions, look up the method and call it
                            let method_name = self.get_heap().get_string(method_name_sym)?;
                            let method = {
                                let struct_def = self.get_heap().get_struct_def(key)?;
                                if let Some(method) = struct_def.get_method(method_name) {
                                    method.clone()
                                } else {
                                    return Err(WalrusError::MethodNotFound {
                                        type_name: struct_def.name().to_string(),
                                        method: method_name.to_string(),
                                        span,
                                        src: self.source_ref.source().into(),
                                        filename: self.source_ref.filename().into(),
                                    });
                                }
                            };

                            // Call the struct method
                            if let WalrusFunction::Vm(func) = method {
                                if args.len() != func.arity {
                                    return Err(WalrusError::InvalidArgCount {
                                        name: func.name.clone(),
                                        expected: func.arity,
                                        got: args.len(),
                                        span,
                                        src: self.source_ref.source().into(),
                                        filename: self.source_ref.filename().into(),
                                    });
                                }

                                let new_frame = CallFrame {
                                    return_ip: self.ip,
                                    frame_pointer: self.locals.len(),
                                    stack_pointer: self.stack.len(),
                                    instructions: Rc::clone(&func.code),
                                    function_name: String::new(),
                                    return_override: None,
                                    module_binding: func.module_binding.clone(),
                                    awaiting_task: None,
                                };

                                self.call_stack.push(new_frame);

                                for arg in args {
                                    self.locals.push(arg);
                                }

                                self.ip = 0;
                                continue; // Skip pushing result, function handles its own return
                            } else {
                                return Err(WalrusError::StructMethodMustBeVmFunction {
                                    span,
                                    src: self.source_ref.source().into(),
                                    filename: self.source_ref.filename().into(),
                                });
                            }
                        }
                        Value::StructInst(inst_key) => {
                            // For struct instances, look up method on the underlying struct
                            // and inject the instance as implicit `self`.
                            let method_name =
                                self.get_heap().get_string(method_name_sym)?.to_string();
                            let (method, type_name) = {
                                let heap = self.get_heap();
                                let inst = heap.get_struct_inst(inst_key)?;
                                let struct_def = heap.get_struct_def(inst.struct_def())?;
                                (
                                    struct_def.get_method(&method_name).cloned(),
                                    struct_def.name().to_string(),
                                )
                            };

                            if let Some(WalrusFunction::Vm(func)) = method {
                                let expected_without_self = func.arity.saturating_sub(1);
                                if args.len() != expected_without_self {
                                    return Err(WalrusError::InvalidArgCount {
                                        name: format!("{}::{}", type_name, method_name),
                                        expected: expected_without_self,
                                        got: args.len(),
                                        span,
                                        src: self.source_ref.source().into(),
                                        filename: self.source_ref.filename().into(),
                                    });
                                }

                                let new_frame = CallFrame {
                                    return_ip: self.ip,
                                    frame_pointer: self.locals.len(),
                                    stack_pointer: self.stack.len(),
                                    instructions: Rc::clone(&func.code),
                                    function_name: String::new(),
                                    return_override: None,
                                    module_binding: func.module_binding.clone(),
                                    awaiting_task: None,
                                };

                                self.call_stack.push(new_frame);
                                self.locals.push(Value::StructInst(inst_key));
                                for arg in args {
                                    self.locals.push(arg);
                                }

                                self.ip = 0;
                                continue; // Skip pushing result, function handles its own return
                            } else if method.is_some() {
                                return Err(WalrusError::StructMethodMustBeVmFunction {
                                    span,
                                    src: self.source_ref.source().into(),
                                    filename: self.source_ref.filename().into(),
                                });
                            } else {
                                return Err(WalrusError::MethodNotFound {
                                    type_name,
                                    method: method_name,
                                    span,
                                    src: self.source_ref.source().into(),
                                    filename: self.source_ref.filename().into(),
                                });
                            }
                        }
                        _ => {
                            return Err(WalrusError::InvalidMethodReceiver {
                                method: self.get_heap().get_string(method_name_sym)?.to_string(),
                                type_name: object.get_type().to_string(),
                                span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            });
                        }
                    };
                    self.push(result);
                }
            }

            if debug_logging_enabled {
                self.stack_trace();
            }
        }
        // Note: This is unreachable because the loop only exits via return statements
        // (either from Return opcode, errors, or end of main frame)
    }

    fn construct_err(&self, op: Opcode, a: Value, b: Option<Value>, span: Span) -> WalrusError {
        if let Some(b) = b {
            WalrusError::InvalidOperation {
                op,
                left: a.get_type().to_string(),
                right: b.get_type().to_string(),
                span,
                src: self.source_ref.source().to_string(),
                filename: self.source_ref.filename().to_string(),
            }
        } else {
            WalrusError::InvalidUnaryOperation {
                op,
                operand: a.get_type().to_string(),
                span,
                src: self.source_ref.source().to_string(),
                filename: self.source_ref.filename().to_string(),
            }
        }
    }

    /// Profile a loop iteration for hotspot detection.
    /// Returns true if the loop just became hot.
    fn profile_loop_iteration(&mut self, loop_header_ip: usize, exit_ip: usize) -> bool {
        if !self.profiling_enabled {
            return false;
        }

        // Dynamic loop registration: if this is the first time seeing this loop,
        // register it with the hotspot detector
        if !self.hotspot_detector.is_loop_header(loop_header_ip) {
            self.hotspot_detector.register_loop(
                loop_header_ip,
                loop_header_ip, // back edge is also at header for ForRangeNext
                exit_ip,
            );
        }

        let became_hot = self.hotspot_detector.record_loop_iteration(loop_header_ip);
        if became_hot {
            debug!("Hot range loop detected at IP {}", loop_header_ip);
        }
        became_hot
    }

    /// Try to execute a JIT-compiled range loop.
    /// Returns Some(exit_ip) if the loop was handled by JIT, None otherwise.
    #[cfg(feature = "jit")]
    fn try_jit_range_loop(
        &mut self,
        loop_header_ip: usize,
        local_idx: u32,
        jump_target: u32,
    ) -> Option<usize> {
        if !self.jit_enabled {
            return None;
        }

        let jit_compiler = self.jit_compiler.as_ref()?;
        if !jit_compiler.is_compiled(loop_header_ip) {
            return None;
        }

        let fp = self.frame_pointer();
        let loop_var_idx = fp + local_idx as usize;

        // Verify we have integer bounds
        let (current, end) = match (self.locals[loop_var_idx], self.locals[loop_var_idx + 1]) {
            (Value::Int(c), Value::Int(e)) => (c, e),
            _ => return None,
        };

        let compiled = jit_compiler.get_compiled(loop_header_ip)?;

        // Get the initial accumulator value from the correct local
        let initial_acc = compiled
            .accumulator_local
            .and_then(|acc_local| {
                let acc_idx = fp + acc_local as usize;
                if acc_idx < self.locals.len() {
                    match self.locals[acc_idx] {
                        Value::Int(v) => Some(v),
                        _ => None,
                    }
                } else {
                    None
                }
            })
            .unwrap_or(0);

        // Execute the JIT compiled function
        let result = unsafe { compiled.call_int_range_accum(current, end, initial_acc) };

        // Store the result back to the accumulator local
        if let Some(acc_local) = compiled.accumulator_local {
            let acc_idx = fp + acc_local as usize;
            while self.locals.len() <= acc_idx {
                self.locals.push(Value::Void);
            }
            self.locals[acc_idx] = Value::Int(result);
        }

        // Set the loop counter to the end value (loop is complete)
        self.locals[loop_var_idx] = Value::Int(end);

        // Ensure locals vector has space for any locals the loop body would have created
        let min_locals_needed = fp + local_idx as usize + 3;
        while self.locals.len() < min_locals_needed {
            self.locals.push(Value::Void);
        }

        Some(jump_target as usize)
    }

    /// Try to compile a hot range loop.
    #[cfg(feature = "jit")]
    fn try_compile_hot_range_loop(&mut self, loop_header_ip: usize, exit_ip: usize) {
        if !self.jit_enabled {
            return;
        }

        // Check if already compiled or not hot
        let should_compile = self
            .jit_compiler
            .as_ref()
            .map(|jit| {
                !jit.is_compiled(loop_header_ip)
                    && self.hotspot_detector.is_loop_hot(loop_header_ip)
            })
            .unwrap_or(false);

        if !should_compile {
            return;
        }

        let instructions = self.current_frame().instructions.clone();

        if let Some(ref mut jit_compiler) = self.jit_compiler {
            match jit_compiler.compile_int_range_sum_loop(&instructions, loop_header_ip, exit_ip) {
                Ok(_) => {
                    self.hotspot_detector.mark_compiled(loop_header_ip);
                    debug!("JIT: Compiled loop at IP {}", loop_header_ip);
                }
                Err(e) => {
                    debug!(
                        "JIT: Failed to compile loop at IP {}: {}",
                        loop_header_ip, e
                    );
                }
            }
        }
    }

    #[inline(always)]
    fn push(&mut self, value: Value) {
        self.stack.push(value);
    }

    #[inline]
    fn pop(&mut self, op: Opcode, span: Span) -> WalrusResult<Value> {
        self.stack.pop().ok_or_else(|| WalrusError::StackUnderflow {
            op,
            span,
            src: self.source_ref.source().to_string(),
            filename: self.source_ref.filename().to_string(),
        })
    }

    /// Fast path pop - only use when stack is guaranteed to have values
    #[inline(always)]
    fn pop_unchecked(&mut self) -> Value {
        // SAFETY: caller guarantees stack is not empty
        unsafe { self.stack.pop().unwrap_unchecked() }
    }

    fn pop_n(&mut self, n: usize, op: Opcode, span: Span) -> WalrusResult<Vec<Value>> {
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

    fn stack_trace(&self) {
        for (i, frame) in self.stack.iter().enumerate() {
            debug!("| {} | {}: {}", self.function_name(), i, frame);
        }
    }

    /// Run the debugger prompt and return the command
    fn run_debugger_prompt(
        &mut self,
        instructions: &InstructionSet,
    ) -> WalrusResult<debugger::DebuggerCommand> {
        // Build call stack info for debugger
        let call_stack: Vec<debugger::DebugCallFrame> = self
            .call_stack
            .iter()
            .map(|f| debugger::DebugCallFrame {
                function_name: if f.function_name.is_empty() {
                    "<fn>".to_string()
                } else {
                    f.function_name.clone()
                },
                return_ip: f.return_ip,
                frame_pointer: f.frame_pointer,
            })
            .collect();

        let ctx = debugger::DebugContext {
            ip: self.ip,
            stack: &self.stack,
            locals: &self.locals,
            globals: &self.globals,
            call_stack: &call_stack,
            debug_info: instructions.debug_info.as_ref(),
            instructions: &instructions.instructions,
            source: self.source_ref.source(),
        };

        let cmd = if let Some(ref mut dbg) = self.debugger {
            debugger::debug_prompt(dbg, &ctx)
        } else {
            debugger::DebuggerCommand::Continue
        };

        Ok(cmd)
    }
}
