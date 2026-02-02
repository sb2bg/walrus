use std::cell::RefCell;
use std::io;
use std::io::Write;
use std::ptr::NonNull;
use std::rc::Rc;

use float_ord::FloatOrd;
use log::{debug, log_enabled};
use rustc_hash::FxHashMap;

use instruction_set::InstructionSet;

use crate::WalrusResult;
use crate::arenas::HeapValue;
use crate::error::WalrusError;
use crate::function::WalrusFunction;
use crate::iter::ValueIterator;
use crate::range::RangeValue;
use crate::source_ref::SourceRef;
use crate::span::Span;
use crate::value::Value;
use crate::vm::opcode::Opcode;

pub mod compiler;
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
/// - The function's bytecode (via Rc to avoid cloning)
/// - The function name for debugging/stack traces
#[derive(Debug)]
struct CallFrame {
    /// Instruction pointer to return to after this frame completes
    return_ip: usize,
    /// Index into the shared `locals` vector where this frame's variables start
    frame_pointer: usize,
    /// Reference to the function's InstructionSet (shared via Rc to avoid cloning)
    instructions: Rc<InstructionSet>,
    /// Function name for debugging and stack traces
    function_name: String,
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
/// Globals are stored in a shared `Rc<RefCell<Vec<Value>>>` so they persist across
/// all call frames.
///
/// ## Memory Model
/// Heap-allocated values (strings, lists, dicts, functions) are stored in a global
/// arena (`ARENA`) and referenced by keys. The VM never directly holds heap data,
/// only keys that index into the arena.
#[derive(Debug)]
pub struct VM<'a> {
    stack: Vec<Value>,          // Operand stack for expression evaluation
    locals: Vec<Value>,         // Shared across all call frames
    call_stack: Vec<CallFrame>, // Stack of call frames
    ip: usize,                  // Current instruction pointer
    globals: Rc<RefCell<Vec<Value>>>,
    source_ref: SourceRef<'a>,
    paused: bool,
    breakpoints: Vec<usize>,
}

impl<'a> VM<'a> {
    pub fn new(source_ref: SourceRef<'a>, is: InstructionSet) -> Self {
        // Create the initial call frame for the main program
        let main_frame = CallFrame {
            return_ip: 0,
            frame_pointer: 0,
            instructions: Rc::new(is),
            function_name: "<main>".to_string(),
        };

        Self {
            stack: Vec::new(),
            locals: Vec::new(),
            call_stack: vec![main_frame],
            ip: 0,
            globals: Rc::new(RefCell::new(Vec::new())),
            source_ref,
            paused: false,
            breakpoints: Vec::new(),
        }
    }

    /// Get the current call frame (the one at the top of the call stack)
    #[inline(always)]
    fn current_frame(&self) -> &CallFrame {
        // SAFETY: call_stack is never empty during VM execution (always has at least main frame)
        unsafe { self.call_stack.last().unwrap_unchecked() }
    }

    /// Get the current frame pointer (start of current frame's locals)
    #[inline(always)]
    fn frame_pointer(&self) -> usize {
        self.current_frame().frame_pointer
    }

    /// Get the current function name
    #[inline(always)]
    fn function_name(&self) -> &str {
        &self.current_frame().function_name
    }

    /// Helper to access heap - uses global ARENA
    #[inline]
    fn get_heap(&self) -> &crate::arenas::ValueHolder {
        unsafe { &*std::ptr::addr_of!(crate::arenas::ARENA) }
    }

    /// Helper to access heap mutably - uses global ARENA
    #[inline]
    fn get_heap_mut(&mut self) -> &mut crate::arenas::ValueHolder {
        unsafe { &mut *std::ptr::addr_of_mut!(crate::arenas::ARENA) }
    }

    /// Collect all root values that the GC needs to trace from
    fn collect_roots(&self) -> Vec<Value> {
        let mut roots = Vec::with_capacity(self.stack.len() + self.locals.len() + 64);

        // Stack values are roots
        roots.extend(self.stack.iter().copied());

        // Local variables are roots
        roots.extend(self.locals.iter().copied());

        // Global variables are roots
        roots.extend(self.globals.borrow().iter().copied());

        // Constants in all call frames are roots (they might reference heap objects)
        for frame in &self.call_stack {
            roots.extend(frame.instructions.constants.iter().copied());
        }

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
                trace.push_str(&format!("  {}: {}\n", i, frame.function_name));
            }
        } else {
            // Show first N frames
            for (i, frame) in self.call_stack.iter().take(MAX_FRAMES_TOP).enumerate() {
                trace.push_str(&format!("  {}: {}\n", i, frame.function_name));
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
                trace.push_str(&format!("  {}: {}\n", actual_index, frame.function_name));
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
        use crate::function::NativeFunction;

        // Check arity
        if args.len() != native_fn.arity() {
            return Err(WalrusError::InvalidArgCount {
                name: native_fn.name().to_string(),
                expected: native_fn.arity(),
                got: args.len(),
                span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            });
        }

        match native_fn {
            NativeFunction::FileOpen => {
                let path = self.value_to_string(args[0], span)?;
                let mode = self.value_to_string(args[1], span)?;
                crate::stdlib::file_open(&path, &mode, span)
            }
            NativeFunction::FileRead => {
                let handle = self.value_to_int(args[0], span)?;
                let content = crate::stdlib::file_read(handle, span)?;
                let value = self.get_heap_mut().push(HeapValue::String(&content));
                Ok(value)
            }
            NativeFunction::FileReadLine => {
                let handle = self.value_to_int(args[0], span)?;
                match crate::stdlib::file_read_line(handle, span)? {
                    Some(line) => {
                        let value = self.get_heap_mut().push(HeapValue::String(&line));
                        Ok(value)
                    }
                    None => Ok(Value::Void),
                }
            }
            NativeFunction::FileWrite => {
                let handle = self.value_to_int(args[0], span)?;
                let content = self.value_to_string(args[1], span)?;
                let bytes = crate::stdlib::file_write(handle, &content, span)?;
                Ok(Value::Int(bytes))
            }
            NativeFunction::FileClose => {
                let handle = self.value_to_int(args[0], span)?;
                crate::stdlib::file_close(handle, span)?;
                Ok(Value::Void)
            }
            NativeFunction::FileExists => {
                let path = self.value_to_string(args[0], span)?;
                Ok(Value::Bool(crate::stdlib::file_exists(&path)))
            }
            NativeFunction::ReadFile => {
                let path = self.value_to_string(args[0], span)?;
                let content = crate::stdlib::read_file(&path, span)?;
                let value = self.get_heap_mut().push(HeapValue::String(&content));
                Ok(value)
            }
            NativeFunction::WriteFile => {
                let path = self.value_to_string(args[0], span)?;
                let content = self.value_to_string(args[1], span)?;
                crate::stdlib::write_file(&path, &content, span)?;
                Ok(Value::Void)
            }
            NativeFunction::EnvGet => {
                let name = self.value_to_string(args[0], span)?;
                match crate::stdlib::env_get(&name) {
                    Some(value) => {
                        let v = self.get_heap_mut().push(HeapValue::String(&value));
                        Ok(v)
                    }
                    None => Ok(Value::Void),
                }
            }
            NativeFunction::Args => {
                let args = crate::stdlib::args();
                let mut list = Vec::with_capacity(args.len());
                for arg in args {
                    let s = self.get_heap_mut().push(HeapValue::String(&arg));
                    list.push(s);
                }
                let list_val = self.get_heap_mut().push(HeapValue::List(list));
                Ok(list_val)
            }
            NativeFunction::Cwd => {
                match crate::stdlib::cwd() {
                    Some(path) => {
                        let v = self.get_heap_mut().push(HeapValue::String(&path));
                        Ok(v)
                    }
                    None => Ok(Value::Void),
                }
            }
            NativeFunction::Exit => {
                let code = self.value_to_int(args[0], span)?;
                std::process::exit(code as i32);
            }
        }
    }

    /// Helper to extract string from Value
    fn value_to_string(&self, value: Value, span: Span) -> WalrusResult<String> {
        match value {
            Value::String(key) => Ok(self.get_heap().get_string(key)?.to_string()),
            _ => Err(WalrusError::Exception {
                message: format!("Expected string, got {}", value.get_type()),
                span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            }),
        }
    }

    /// Helper to extract int from Value
    fn value_to_int(&self, value: Value, span: Span) -> WalrusResult<i64> {
        match value {
            Value::Int(n) => Ok(n),
            _ => Err(WalrusError::Exception {
                message: format!("Expected int, got {}", value.get_type()),
                span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            }),
        }
    }

    /// Run the VM and add stack trace information to any errors
    pub fn run(&mut self) -> WalrusResult<Value> {
        match self.run_inner() {
            Ok(value) => Ok(value),
            Err(err) => {
                // Add stack trace to the error message
                let stack_trace = self.format_stack_trace();
                if stack_trace.is_empty() {
                    Err(err)
                } else {
                    // Wrap the error with stack trace info
                    Err(WalrusError::GenericError {
                        message: format!("{}{}", err, stack_trace),
                    })
                }
            }
        }
    }

    fn run_inner(&mut self) -> WalrusResult<Value> {
        loop {
            // Check if garbage collection is needed
            self.maybe_collect_garbage();

            // Get current frame's instruction set
            let instructions = Rc::clone(&self.current_frame().instructions);

            // Check if we've reached the end of the current frame's instructions
            if self.ip >= instructions.instructions.len() {
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

            instructions.disassemble_single(self.ip, self.function_name());

            if self.paused || self.breakpoints.contains(&self.ip) {
                self.debug_prompt()?;
            }

            let instruction = instructions.get(self.ip);
            let opcode = instruction.opcode();
            let span = instruction.span();

            self.ip += 1;

            match opcode {
                Opcode::LoadConst(index) => {
                    self.push(instructions.get_constant(index));
                }
                Opcode::LoadConst0 => {
                    self.push(instructions.get_constant(0));
                }
                Opcode::LoadConst1 => {
                    self.push(instructions.get_constant(1));
                }
                Opcode::Load(index) => {
                    let fp = self.frame_pointer();
                    self.push(self.locals[fp + index as usize]);
                }
                Opcode::LoadLocal0 => {
                    let fp = self.frame_pointer();
                    self.push(self.locals[fp]);
                }
                Opcode::LoadLocal1 => {
                    let fp = self.frame_pointer();
                    self.push(self.locals[fp + 1]);
                }
                Opcode::LoadLocal2 => {
                    let fp = self.frame_pointer();
                    self.push(self.locals[fp + 2]);
                }
                Opcode::LoadLocal3 => {
                    let fp = self.frame_pointer();
                    self.push(self.locals[fp + 3]);
                }
                // Specialized increment/decrement for loop counters (hot path)
                Opcode::IncrementLocal(index) => {
                    let fp = self.frame_pointer();
                    let idx = fp + index as usize;
                    if let Value::Int(v) = self.locals[idx] {
                        self.locals[idx] = Value::Int(v + 1);
                    } else {
                        return Err(WalrusError::Exception {
                            message: "IncrementLocal requires integer".to_string(),
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        });
                    }
                }
                Opcode::DecrementLocal(index) => {
                    let fp = self.frame_pointer();
                    let idx = fp + index as usize;
                    if let Value::Int(v) = self.locals[idx] {
                        self.locals[idx] = Value::Int(v - 1);
                    } else {
                        return Err(WalrusError::Exception {
                            message: "DecrementLocal requires integer".to_string(),
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
                    // Check if current < end, if so push current and increment, else jump
                    let fp = self.frame_pointer();
                    let idx = fp + local_idx as usize;
                    if let (Value::Int(current), Value::Int(end)) =
                        (self.locals[idx], self.locals[idx + 1])
                    {
                        if current < end {
                            self.push(Value::Int(current));
                            self.locals[idx] = Value::Int(current + 1);
                        } else {
                            self.ip = jump_target as usize;
                        }
                    } else {
                        return Err(WalrusError::Exception {
                            message: "ForRangeNext requires integer range bounds".to_string(),
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        });
                    }
                }
                Opcode::LoadGlobal(index) => {
                    let value = {
                        let globals = self.globals.borrow();
                        globals[index as usize]
                    };
                    self.push(value);
                }
                Opcode::LoadGlobal0 => {
                    let value = {
                        let globals = self.globals.borrow();
                        globals[0]
                    };
                    self.push(value);
                }
                Opcode::LoadGlobal1 => {
                    let value = {
                        let globals = self.globals.borrow();
                        globals[1]
                    };
                    self.push(value);
                }
                Opcode::LoadGlobal2 => {
                    let value = {
                        let globals = self.globals.borrow();
                        globals[2]
                    };
                    self.push(value);
                }
                Opcode::LoadGlobal3 => {
                    let value = {
                        let globals = self.globals.borrow();
                        globals[3]
                    };
                    self.push(value);
                }
                Opcode::Store => {
                    let value = self.pop(opcode, span)?;
                    self.locals.push(value);
                }
                Opcode::StoreAt(index) => {
                    let value = self.pop(opcode, span)?;
                    let fp = self.frame_pointer();
                    let abs_index = fp + index as usize;

                    if abs_index == self.locals.len() {
                        self.locals.push(value);
                    } else {
                        self.locals[abs_index] = value;
                    }
                }
                Opcode::StoreGlobal(index) => {
                    let value = self.pop(opcode, span)?;
                    let index = index as usize;
                    let mut globals = self.globals.borrow_mut();

                    if index == globals.len() {
                        globals.push(value);
                    } else {
                        globals[index] = value;
                    }
                }
                Opcode::Reassign(index) => {
                    let value = self.pop(opcode, span)?;
                    let fp = self.frame_pointer();
                    self.locals[fp + index as usize] = value;
                }
                Opcode::ReassignGlobal(index) => {
                    let value = self.pop(opcode, span)?;
                    let mut globals = self.globals.borrow_mut();
                    globals[index as usize] = value;
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
                    self.pop(opcode, span)?;
                }
                Opcode::PopLocal(num) => {
                    for _ in 0..num {
                        self.locals.pop();
                    }
                }
                Opcode::JumpIfFalse(offset) => {
                    let value = self.pop(opcode, span)?;

                    if let Value::Bool(false) = value {
                        self.ip = offset as usize;
                    }
                }
                Opcode::Jump(offset) => {
                    self.ip = offset as usize;
                }
                Opcode::GetIter => {
                    let value = self.pop(opcode, span)?;
                    let iter = self.get_heap_mut().value_to_iter(value)?;
                    self.push(iter);
                }
                Opcode::IterNext(offset) => {
                    let iter = self.pop(opcode, span)?;

                    match iter {
                        Value::Iter(key) => unsafe {
                            let mut ptr = NonNull::from(self.get_heap_mut());
                            let iter = ptr.as_mut().get_mut_iter(key)?;

                            if let Some(value) = iter.next(self.get_heap_mut()) {
                                // fixme: if another value gets pushed on the stack, this cause a non iterable error
                                self.push(Value::Iter(key));
                                self.push(value);
                            } else {
                                self.ip = offset as usize;
                            }
                        },
                        value => {
                            return Err(WalrusError::NotIterable {
                                type_name: value.get_type().to_string(),
                                span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            });
                        }
                    }
                }
                Opcode::Call(args) => {
                    let arg_count = args as usize;
                    let func = self.pop(opcode, span)?;
                    let args = self.pop_n(arg_count, opcode, span)?;

                    match func {
                        Value::Function(key) => {
                            let func = self.get_heap().get_function(key)?;

                            match func {
                                WalrusFunction::Rust(func) => {
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
                                    let result = self.call_native(*native_fn, args, span)?;
                                    self.push(result);
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

                                    // Create a new call frame instead of a child VM
                                    let new_frame = CallFrame {
                                        return_ip: self.ip,                  // Where to return after this function
                                        frame_pointer: self.locals.len(), // New frame starts at current locals end
                                        instructions: Rc::clone(&func.code), // Share the instruction set via Rc
                                        function_name: format!("fn<{}>", func.name),
                                    };

                                    // Push the new frame
                                    self.call_stack.push(new_frame);

                                    // Push arguments as locals for the new frame
                                    for arg in args {
                                        self.locals.push(arg);
                                    }

                                    // Start execution at the beginning of the new function
                                    self.ip = 0;
                                }
                                _ => {
                                    // In theory, this should never happen because the compiler
                                    // should not compile a call to a node function (but just in case)
                                    return Err(WalrusError::Exception {
                                        message: "Cannot call a node function from the VM"
                                            .to_string(),
                                        span,
                                        src: self.source_ref.source().into(),
                                        filename: self.source_ref.filename().into(),
                                    });
                                }
                            }
                        }
                        _ => {
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
                    let args = self.pop_n(arg_count, opcode, span)?;

                    match func {
                        Value::Function(key) => {
                            let func = self.get_heap().get_function(key)?;

                            match func {
                                WalrusFunction::Rust(func) => {
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

                                    if self.call_stack.is_empty() {
                                        return Ok(result);
                                    }

                                    self.locals.truncate(frame.frame_pointer);
                                    self.ip = frame.return_ip;
                                    self.push(result);
                                }
                                WalrusFunction::Native(native_fn) => {
                                    let result = self.call_native(*native_fn, args, span)?;
                                    
                                    // For a tail call, we need to return this result
                                    let frame = self
                                        .call_stack
                                        .pop()
                                        .expect("Call stack should never be empty on tail call");

                                    if self.call_stack.is_empty() {
                                        return Ok(result);
                                    }

                                    self.locals.truncate(frame.frame_pointer);
                                    self.ip = frame.return_ip;
                                    self.push(result);
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

                                    // Clone what we need before mutating self
                                    let new_instructions = Rc::clone(&func.code);
                                    let new_name = format!("fn<{}>", func.name);

                                    // Get the current frame pointer before modifying
                                    let frame_pointer = self.frame_pointer();

                                    // Truncate locals to our frame pointer (clear current frame's locals)
                                    self.locals.truncate(frame_pointer);

                                    // Push the new arguments
                                    for arg in args {
                                        self.locals.push(arg);
                                    }

                                    // Update the current frame in place (reuse it)
                                    // Keep the same return_ip and frame_pointer, just update instructions and name
                                    if let Some(current_frame) = self.call_stack.last_mut() {
                                        current_frame.instructions = new_instructions;
                                        current_frame.function_name = new_name;
                                    }

                                    // Reset IP to start of the new function
                                    self.ip = 0;
                                }
                                _ => {
                                    return Err(WalrusError::Exception {
                                        message: "Cannot call a node function from the VM"
                                            .to_string(),
                                        span,
                                        src: self.source_ref.source().into(),
                                        filename: self.source_ref.filename().into(),
                                    });
                                }
                            }
                        }
                        _ => {
                            return Err(WalrusError::NotCallable {
                                value: func.get_type().to_string(),
                                span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            });
                        }
                    }
                }
                // Specialized integer arithmetic (hot path - skips type checking)
                Opcode::AddInt => {
                    // SAFETY: Compiler guarantees both operands are integers
                    let b = self.stack.pop().unwrap();
                    let a = self.stack.pop().unwrap();
                    if let (Value::Int(a), Value::Int(b)) = (a, b) {
                        self.push(Value::Int(a + b));
                    } else {
                        // Fallback for safety (shouldn't happen with correct compilation)
                        return Err(WalrusError::Exception {
                            message: "AddInt requires integers".to_string(),
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        });
                    }
                }
                Opcode::SubtractInt => {
                    let b = self.stack.pop().unwrap();
                    let a = self.stack.pop().unwrap();
                    if let (Value::Int(a), Value::Int(b)) = (a, b) {
                        self.push(Value::Int(a - b));
                    } else {
                        return Err(WalrusError::Exception {
                            message: "SubtractInt requires integers".to_string(),
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        });
                    }
                }
                Opcode::LessInt => {
                    let b = self.stack.pop().unwrap();
                    let a = self.stack.pop().unwrap();
                    if let (Value::Int(a), Value::Int(b)) = (a, b) {
                        self.push(Value::Bool(a < b));
                    } else {
                        return Err(WalrusError::Exception {
                            message: "LessInt requires integers".to_string(),
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        });
                    }
                }
                Opcode::Add => {
                    let b = self.pop_unchecked();
                    let a = self.pop_unchecked();

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

                    match a {
                        Value::Bool(a) => {
                            self.push(Value::Bool(!a));
                        }
                        _ => return Err(self.construct_err(opcode, a, None, span)),
                    }
                }
                Opcode::And => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::Bool(a), Value::Bool(b)) => {
                            self.push(Value::Bool(a && b));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Or => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::Bool(a), Value::Bool(b)) => {
                            self.push(Value::Bool(a || b));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
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
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

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
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::List(a), Value::Int(b)) => {
                            let a = self.get_heap().get_list(a)?;
                            let mut b = b;
                            let original = b;

                            // todo: merge code with other index ops
                            if b < 0 {
                                b += a.len() as i64;
                            }

                            if b < 0 || b >= a.len() as i64 {
                                return Err(WalrusError::IndexOutOfBounds {
                                    index: original,
                                    len: a.len(),
                                    span,
                                    src: self.source_ref.source().to_string(),
                                    filename: self.source_ref.filename().to_string(),
                                });
                            }

                            self.push(a[b as usize]);
                        }
                        (Value::String(a), Value::Int(b)) => {
                            let a = self.get_heap().get_string(a)?;
                            let mut b = b;
                            let original = b;

                            if b < 0 {
                                b += a.len() as i64;
                            }

                            if b < 0 || b >= a.len() as i64 {
                                return Err(WalrusError::IndexOutOfBounds {
                                    index: original,
                                    len: a.len(),
                                    span,
                                    src: self.source_ref.source().to_string(),
                                    filename: self.source_ref.filename().to_string(),
                                });
                            }

                            let b = b as usize;
                            let res = a[b..b + 1].to_string();
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
                        (Value::String(a), Value::Range(range)) => {
                            let a = self.get_heap().get_string(a)?;
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

                            let res = a[start as usize..end as usize].to_string();
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
                    let value = self.pop(opcode, span)?;
                    let index = self.pop(opcode, span)?;
                    let object = self.pop(opcode, span)?;

                    match (object, index) {
                        (Value::List(list_key), Value::Int(idx)) => {
                            let list = self.get_heap_mut().get_mut_list(list_key)?;
                            let mut idx = idx;
                            let original = idx;

                            // Handle negative indices
                            if idx < 0 {
                                idx += list.len() as i64;
                            }

                            if idx < 0 || idx >= list.len() as i64 {
                                return Err(WalrusError::IndexOutOfBounds {
                                    index: original,
                                    len: list.len(),
                                    span,
                                    src: self.source_ref.source().to_string(),
                                    filename: self.source_ref.filename().to_string(),
                                });
                            }

                            list[idx as usize] = value;
                            self.push(Value::Void);
                        }
                        (Value::Dict(dict_key), key) => {
                            let dict = self.get_heap_mut().get_mut_dict(dict_key)?;
                            dict.insert(key, value);
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
                Opcode::Len => {
                    let a = self.pop(opcode, span)?;

                    match a {
                        Value::String(key) => {
                            let s = self.get_heap().get_string(key)?;
                            self.push(Value::Int(s.len() as i64));
                        }
                        Value::List(key) => {
                            let list = self.get_heap().get_list(key)?;
                            self.push(Value::Int(list.len() as i64));
                        }
                        Value::Dict(key) => {
                            let dict = self.get_heap().get_dict(key)?;
                            self.push(Value::Int(dict.len() as i64));
                        }
                        _ => {
                            return Err(WalrusError::NoLength {
                                type_name: a.get_type().to_string(),
                                span,
                                src: self.source_ref.source().to_string(),
                                filename: self.source_ref.filename().to_string(),
                            });
                        }
                    }
                }
                Opcode::Str => {
                    let a = self.pop(opcode, span)?;
                    let s = self.stringify_value(a)?;
                    let value = self.get_heap_mut().push(HeapValue::String(&s));
                    self.push(value);
                }
                Opcode::Type => {
                    let a = self.pop(opcode, span)?;
                    let type_name = a.get_type();
                    let value = self.get_heap_mut().push(HeapValue::String(type_name));
                    self.push(value);
                }
                Opcode::Gc => {
                    // Trigger garbage collection and return stats as a dict
                    let roots = self.collect_roots();
                    let result = self.get_heap_mut().force_collect(&roots);

                    // Build result dict
                    let mut dict = FxHashMap::default();
                    let heap = self.get_heap_mut();

                    let key_freed = heap.push(HeapValue::String("objects_freed"));
                    let key_before = heap.push(HeapValue::String("objects_before"));
                    let key_after = heap.push(HeapValue::String("objects_after"));
                    let key_collections = heap.push(HeapValue::String("total_collections"));

                    dict.insert(key_freed, Value::Int(result.objects_freed as i64));
                    dict.insert(key_before, Value::Int(result.objects_before as i64));
                    dict.insert(key_after, Value::Int(result.objects_after as i64));
                    dict.insert(key_collections, Value::Int(result.collections_total as i64));

                    let result_dict = self.get_heap_mut().push(HeapValue::Dict(dict));
                    self.push(result_dict);
                }
                Opcode::HeapStats => {
                    // Get heap statistics as a dict
                    let stats = self.get_heap().heap_stats();
                    let gc_info = self.get_heap().gc_stats();

                    let mut dict = FxHashMap::default();
                    let heap = self.get_heap_mut();

                    // Object counts
                    let key_lists = heap.push(HeapValue::String("lists"));
                    let key_tuples = heap.push(HeapValue::String("tuples"));
                    let key_dicts = heap.push(HeapValue::String("dicts"));
                    let key_functions = heap.push(HeapValue::String("functions"));
                    let key_iterators = heap.push(HeapValue::String("iterators"));
                    let key_struct_defs = heap.push(HeapValue::String("struct_defs"));
                    let key_struct_insts = heap.push(HeapValue::String("struct_instances"));
                    let key_total = heap.push(HeapValue::String("total_objects"));

                    // GC info
                    let key_alloc_count = heap.push(HeapValue::String("allocation_count"));
                    let key_bytes = heap.push(HeapValue::String("bytes_allocated"));
                    let key_bytes_freed = heap.push(HeapValue::String("total_bytes_freed"));
                    let key_collections = heap.push(HeapValue::String("total_collections"));
                    let key_threshold = heap.push(HeapValue::String("allocation_threshold"));
                    let key_mem_threshold = heap.push(HeapValue::String("memory_threshold"));

                    dict.insert(key_lists, Value::Int(stats.lists as i64));
                    dict.insert(key_tuples, Value::Int(stats.tuples as i64));
                    dict.insert(key_dicts, Value::Int(stats.dicts as i64));
                    dict.insert(key_functions, Value::Int(stats.functions as i64));
                    dict.insert(key_iterators, Value::Int(stats.iterators as i64));
                    dict.insert(key_struct_defs, Value::Int(stats.struct_defs as i64));
                    dict.insert(key_struct_insts, Value::Int(stats.struct_instances as i64));
                    dict.insert(key_total, Value::Int(stats.total_objects() as i64));

                    dict.insert(key_alloc_count, Value::Int(gc_info.allocation_count as i64));
                    dict.insert(key_bytes, Value::Int(gc_info.bytes_allocated as i64));
                    dict.insert(
                        key_bytes_freed,
                        Value::Int(gc_info.total_bytes_freed as i64),
                    );
                    dict.insert(
                        key_collections,
                        Value::Int(gc_info.total_collections as i64),
                    );
                    dict.insert(
                        key_threshold,
                        Value::Int(gc_info.allocation_threshold as i64),
                    );
                    dict.insert(
                        key_mem_threshold,
                        Value::Int(gc_info.memory_threshold as i64),
                    );

                    let result_dict = self.get_heap_mut().push(HeapValue::Dict(dict));
                    self.push(result_dict);
                }
                Opcode::GcConfig => {
                    // Set GC allocation threshold
                    let threshold = self.pop(opcode, span)?;
                    match threshold {
                        Value::Int(n) if n > 0 => {
                            let old = crate::gc::set_allocation_threshold(n as usize);
                            self.push(Value::Int(old as i64));
                        }
                        _ => {
                            return Err(WalrusError::GenericError {
                                message: "__gc_threshold__ requires a positive integer".to_string(),
                            });
                        }
                    }
                }
                // Import system - returns a module dict with native functions
                Opcode::Import => {
                    let module_name = self.pop(opcode, span)?;
                    match module_name {
                        Value::String(name_key) => {
                            let name_str = self.get_heap().get_string(name_key)?;
                            if let Some(functions) = crate::stdlib::get_module_functions(name_str) {
                                // Build a dict where keys are function names and values are the functions
                                let mut dict = FxHashMap::default();
                                for native_fn in functions {
                                    let key = self.get_heap_mut().push(HeapValue::String(native_fn.name()));
                                    let func = self.get_heap_mut().push(HeapValue::Function(
                                        WalrusFunction::Native(native_fn)
                                    ));
                                    dict.insert(key, func);
                                }
                                let module = self.get_heap_mut().push(HeapValue::Dict(dict));
                                self.push(module);
                            } else {
                                return Err(WalrusError::Exception {
                                    message: format!("Unknown module: '{}'. Available: std/io, std/sys", name_str),
                                    span,
                                    src: self.source_ref.source().into(),
                                    filename: self.source_ref.filename().into(),
                                });
                            }
                        }
                        _ => {
                            return Err(WalrusError::Exception {
                                message: "import requires a string module name".to_string(),
                                span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            });
                        }
                    }
                }
                Opcode::Return => {
                    let return_value = self.pop(opcode, span)?;

                    // Pop the current call frame
                    let frame = self
                        .call_stack
                        .pop()
                        .expect("Call stack should never be empty on return");

                    // If this was the last frame (main), return the value
                    if self.call_stack.is_empty() {
                        return Ok(return_value);
                    }

                    // Truncate locals back to the frame pointer (cleanup this frame's locals)
                    self.locals.truncate(frame.frame_pointer);

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
                        let instance = crate::structs::StructInstance::new(struct_name);
                        let instance_value =
                            self.get_heap_mut().push(HeapValue::StructInst(instance));

                        self.push(instance_value);
                    } else {
                        return Err(WalrusError::Exception {
                            message: format!(
                                "Expected struct definition, got {}",
                                struct_def_value.get_type()
                            ),
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        });
                    }
                }
                Opcode::GetMethod => {
                    // Pop method name (string) and struct definition from stack
                    let method_name_value = self.pop(opcode, span)?;
                    let struct_def_value = self.pop(opcode, span)?;

                    if let (Value::String(method_name_sym), Value::StructDef(struct_def_key)) =
                        (method_name_value, struct_def_value)
                    {
                        let method_name = self.get_heap().get_string(method_name_sym)?.to_string();
                        let method_clone = {
                            let struct_def = self.get_heap().get_struct_def(struct_def_key)?;
                            if let Some(method) = struct_def.get_method(&method_name) {
                                method.clone()
                            } else {
                                return Err(WalrusError::Exception {
                                    message: format!(
                                        "Method '{}' not found on struct '{}'",
                                        method_name,
                                        struct_def.name()
                                    ),
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
                    } else {
                        return Err(WalrusError::Exception {
                            message: "GetMethod expects string method name and struct definition"
                                .to_string(),
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        });
                    }
                }
                Opcode::CallMethod(arg_count) => {
                    // Stack layout: [object, arg1, arg2, ..., argN, method_name]
                    // Pop method name first
                    let method_name_val = self.pop(opcode, span)?;
                    let method_name = match method_name_val {
                        Value::String(sym) => self.get_heap().get_string(sym)?.to_string(),
                        _ => {
                            return Err(WalrusError::Exception {
                                message: "Method name must be a string".to_string(),
                                span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            });
                        }
                    };

                    // Pop arguments
                    let args = self.pop_n(arg_count as usize, opcode, span)?;

                    // Pop the object
                    let object = self.pop(opcode, span)?;

                    // Get source info for error messages
                    let src = self.source_ref.source().to_string();
                    let filename = self.source_ref.filename().to_string();

                    // Dispatch based on object type
                    let result = match object {
                        Value::List(key) => methods::dispatch_list_method(
                            self.get_heap_mut(),
                            key,
                            &method_name,
                            args,
                            span,
                            &src,
                            &filename,
                        )?,
                        Value::String(key) => methods::dispatch_string_method(
                            self.get_heap_mut(),
                            key,
                            &method_name,
                            args,
                            span,
                            &src,
                            &filename,
                        )?,
                        Value::Dict(key) => {
                            // First, check if this dict contains a native function with this name
                            let method_key = self
                                .get_heap_mut()
                                .push(HeapValue::String(&method_name));
                            
                            let dict = self.get_heap().get_dict(key)?;
                            
                            if let Some(func_val) = dict.get(&method_key).copied() {
                                if let Value::Function(func_key) = func_val {
                                    let func = self.get_heap().get_function(func_key)?.clone();
                                    if let WalrusFunction::Native(native) = func {
                                        // Call the native function and push result
                                        let result = self.call_native(native, args, span)?;
                                        self.push(result);
                                        continue; // Skip the push at the end
                                    }
                                }
                            }

                            // Otherwise, use regular dict method dispatch
                            methods::dispatch_dict_method(
                                self.get_heap_mut(),
                                key,
                                &method_name,
                                args,
                                span,
                                &src,
                                &filename,
                            )?
                        },
                        Value::StructDef(key) => {
                            // For struct definitions, look up the method and call it
                            let method = {
                                let struct_def = self.get_heap().get_struct_def(key)?;
                                if let Some(method) = struct_def.get_method(&method_name) {
                                    method.clone()
                                } else {
                                    return Err(WalrusError::Exception {
                                        message: format!(
                                            "Struct '{}' has no method '{}'",
                                            struct_def.name(),
                                            method_name
                                        ),
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
                                    instructions: Rc::clone(&func.code),
                                    function_name: format!("fn<{}>", func.name),
                                };

                                self.call_stack.push(new_frame);

                                for arg in args {
                                    self.locals.push(arg);
                                }

                                self.ip = 0;
                                continue; // Skip pushing result, function handles its own return
                            } else {
                                return Err(WalrusError::Exception {
                                    message: "Struct methods must be VM functions".to_string(),
                                    span,
                                    src: self.source_ref.source().into(),
                                    filename: self.source_ref.filename().into(),
                                });
                            }
                        }
                        _ => {
                            return Err(WalrusError::Exception {
                                message: format!(
                                    "Cannot call method '{}' on type '{}'",
                                    method_name,
                                    object.get_type()
                                ),
                                span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            });
                        }
                    };
                    self.push(result);
                }
            }

            self.stack_trace();
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
        let mut values = Vec::with_capacity(n);

        for _ in 0..n {
            values.push(self.pop(op, span)?);
        }

        values.reverse();
        Ok(values)
    }

    fn stack_trace(&self) {
        if !log_enabled!(log::Level::Debug) {
            return;
        }

        for (i, frame) in self.stack.iter().enumerate() {
            debug!("| {} | {}: {}", self.function_name(), i, frame);
        }
    }

    fn debug_prompt(&mut self) -> WalrusResult<()> {
        loop {
            print!("(debug) ");
            io::stdout().flush().expect("Failed to flush stdout");

            let mut input = String::new();
            io::stdin()
                .read_line(&mut input)
                .expect("Failed to read line");

            match input.trim() {
                "s" | "step" => {
                    self.paused = true;
                    break;
                }
                "c" | "continue" => {
                    self.paused = false;
                    break;
                }
                "p" | "print" => self.print_debug_info(),
                "b" | "breakpoint" => {
                    print!("Enter breakpoint line number: ");
                    io::stdout().flush().expect("Failed to flush stdout");
                    let mut line = String::new();
                    io::stdin()
                        .read_line(&mut line)
                        .expect("Failed to read line");
                    if let Ok(line_num) = line.trim().parse::<usize>() {
                        self.breakpoints.push(line_num);
                        debug!("Breakpoint set at line {}", line_num);
                    } else {
                        debug!("Invalid line number");
                    }
                }
                "q" | "quit" => {
                    return Err(WalrusError::UnknownError {
                        message: "Debugger quit".to_string(),
                    });
                }
                _ => debug!(
                    "Unknown command. Available commands: step (s), continue (c), print (p), breakpoint (b), quit (q)"
                ),
            }
        }

        Ok(())
    }

    fn print_current_instruction(&self) {
        let instruction = self.current_frame().instructions.get(self.ip);
        debug!(
            "Executing {} -> {}",
            instruction.opcode(),
            &self.source_ref.source()[instruction.span().0..instruction.span().1],
        );
    }

    fn print_debug_info(&self) {
        self.print_current_instruction();
        debug!("Current instruction pointer -> {}", self.ip);
        debug!("Stack ->");
        for (i, value) in self.stack.iter().enumerate() {
            debug!("  {}: {:?}", i, value);
        }
        debug!("Locals ->");
        for (i, value) in self.locals.iter().enumerate() {
            debug!("  {}: {:?}", i, value);
        }
    }
}
