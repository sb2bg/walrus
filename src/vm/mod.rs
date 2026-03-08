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
use crate::arenas::{DictKey, FuncKey, HeapValue};
use crate::error::WalrusError;
use crate::function::{NativeFunction, VmModuleBinding, WalrusFunction};
use crate::iter::ValueIterator;
use crate::jit::{HotSpotDetector, TypeProfile, WalrusType};
use crate::range::RangeValue;
use crate::source_ref::SourceRef;
use crate::span::Span;
use crate::value::{AsyncTask, IoChannel, IoResult, Value};
use crate::vm::opcode::Opcode;

pub mod compiler;
mod debug_runtime;
pub mod debugger;
pub mod instruction_set;
mod io_pool;
pub mod methods;
mod modules;
pub mod opcode;
pub mod optimize;
mod scheduler;
mod state;
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

#[derive(Debug, Clone)]
enum CachedGlobalCall {
    Vm {
        arity: usize,
        is_async: bool,
        code: Rc<InstructionSet>,
        module_binding: Option<Rc<VmModuleBinding>>,
    },
    Native {
        function: NativeFunction,
    },
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
    global_call_cache: Vec<Option<CachedGlobalCall>>,
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
    #[inline(always)]
    fn copy_stack_tail_to_locals(&mut self, start: usize) {
        self.locals.extend_from_slice(&self.stack[start..]);
        self.stack.truncate(start);
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
                Opcode::LoadLocal4 => {
                    let fp = self.frame_pointer();
                    // SAFETY: compiler emits LoadLocal4 only when local 4 exists.
                    self.push(unsafe { *self.locals.get_unchecked(fp + 4) });
                }
                Opcode::LoadLocal5 => {
                    let fp = self.frame_pointer();
                    // SAFETY: compiler emits LoadLocal5 only when local 5 exists.
                    self.push(unsafe { *self.locals.get_unchecked(fp + 5) });
                }
                Opcode::LoadLocal6 => {
                    let fp = self.frame_pointer();
                    // SAFETY: compiler emits LoadLocal6 only when local 6 exists.
                    self.push(unsafe { *self.locals.get_unchecked(fp + 6) });
                }
                Opcode::LoadLocal7 => {
                    let fp = self.frame_pointer();
                    // SAFETY: compiler emits LoadLocal7 only when local 7 exists.
                    self.push(unsafe { *self.locals.get_unchecked(fp + 7) });
                }
                Opcode::LoadLocal8 => {
                    let fp = self.frame_pointer();
                    self.push(unsafe { *self.locals.get_unchecked(fp + 8) });
                }
                Opcode::LoadLocal9 => {
                    let fp = self.frame_pointer();
                    self.push(unsafe { *self.locals.get_unchecked(fp + 9) });
                }
                Opcode::LoadLocal10 => {
                    let fp = self.frame_pointer();
                    self.push(unsafe { *self.locals.get_unchecked(fp + 10) });
                }
                Opcode::LoadLocal11 => {
                    let fp = self.frame_pointer();
                    self.push(unsafe { *self.locals.get_unchecked(fp + 11) });
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

                    // Standard execution
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
                Opcode::ForRangeNextDiscard(jump_target, local_idx) => {
                    let loop_header_ip = self.ip - 1;
                    let exit_ip = jump_target as usize;

                    if profiling_enabled {
                        self.profile_loop_iteration(loop_header_ip, exit_ip);

                        #[cfg(feature = "jit")]
                        if let Some(jit_exit) =
                            self.try_jit_range_loop(loop_header_ip, local_idx, jump_target)
                        {
                            self.ip = jit_exit;
                            continue;
                        }

                        #[cfg(feature = "jit")]
                        self.try_compile_hot_range_loop(loop_header_ip, exit_ip);
                    }

                    let fp = self.frame_pointer();
                    let idx = fp + local_idx as usize;
                    let current_value = unsafe { *self.locals.get_unchecked(idx) };
                    let end_value = unsafe { *self.locals.get_unchecked(idx + 1) };
                    if let (Value::Int(current), Value::Int(end)) = (current_value, end_value) {
                        if current < end {
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
                    let value = self.load_global_value_fast(index as usize, span)?;
                    self.push(value);
                }
                Opcode::LoadGlobal0 => {
                    let value = self.load_global_value_fast(0, span)?;
                    self.push(value);
                }
                Opcode::LoadGlobal1 => {
                    let value = self.load_global_value_fast(1, span)?;
                    self.push(value);
                }
                Opcode::LoadGlobal2 => {
                    let value = self.load_global_value_fast(2, span)?;
                    self.push(value);
                }
                Opcode::LoadGlobal3 => {
                    let value = self.load_global_value_fast(3, span)?;
                    self.push(value);
                }
                Opcode::Store => {
                    // SAFETY: valid bytecode guarantees stack has a value to store.
                    let value = self.pop_unchecked();
                    self.locals.push(value);
                }
                Opcode::StoreLocal0 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    if fp == self.locals.len() {
                        self.locals.push(value);
                    } else {
                        unsafe {
                            *self.locals.get_unchecked_mut(fp) = value;
                        }
                    }
                }
                Opcode::StoreLocal1 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    let idx = fp + 1;
                    if idx == self.locals.len() {
                        self.locals.push(value);
                    } else {
                        unsafe {
                            *self.locals.get_unchecked_mut(idx) = value;
                        }
                    }
                }
                Opcode::StoreLocal2 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    let idx = fp + 2;
                    if idx == self.locals.len() {
                        self.locals.push(value);
                    } else {
                        unsafe {
                            *self.locals.get_unchecked_mut(idx) = value;
                        }
                    }
                }
                Opcode::StoreLocal3 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    let idx = fp + 3;
                    if idx == self.locals.len() {
                        self.locals.push(value);
                    } else {
                        unsafe {
                            *self.locals.get_unchecked_mut(idx) = value;
                        }
                    }
                }
                Opcode::StoreLocal4 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    let idx = fp + 4;
                    if idx == self.locals.len() {
                        self.locals.push(value);
                    } else {
                        unsafe {
                            *self.locals.get_unchecked_mut(idx) = value;
                        }
                    }
                }
                Opcode::StoreLocal5 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    let idx = fp + 5;
                    if idx == self.locals.len() {
                        self.locals.push(value);
                    } else {
                        unsafe {
                            *self.locals.get_unchecked_mut(idx) = value;
                        }
                    }
                }
                Opcode::StoreLocal6 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    let idx = fp + 6;
                    if idx == self.locals.len() {
                        self.locals.push(value);
                    } else {
                        unsafe {
                            *self.locals.get_unchecked_mut(idx) = value;
                        }
                    }
                }
                Opcode::StoreLocal7 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    let idx = fp + 7;
                    if idx == self.locals.len() {
                        self.locals.push(value);
                    } else {
                        unsafe {
                            *self.locals.get_unchecked_mut(idx) = value;
                        }
                    }
                }
                Opcode::StoreLocal8 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    let idx = fp + 8;
                    if idx == self.locals.len() {
                        self.locals.push(value);
                    } else {
                        unsafe {
                            *self.locals.get_unchecked_mut(idx) = value;
                        }
                    }
                }
                Opcode::StoreLocal9 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    let idx = fp + 9;
                    if idx == self.locals.len() {
                        self.locals.push(value);
                    } else {
                        unsafe {
                            *self.locals.get_unchecked_mut(idx) = value;
                        }
                    }
                }
                Opcode::StoreLocal10 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    let idx = fp + 10;
                    if idx == self.locals.len() {
                        self.locals.push(value);
                    } else {
                        unsafe {
                            *self.locals.get_unchecked_mut(idx) = value;
                        }
                    }
                }
                Opcode::StoreLocal11 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    let idx = fp + 11;
                    if idx == self.locals.len() {
                        self.locals.push(value);
                    } else {
                        unsafe {
                            *self.locals.get_unchecked_mut(idx) = value;
                        }
                    }
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
                    self.store_global_value_fast(index as usize, value, span)?;
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
                Opcode::ReassignLocal0 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    unsafe {
                        *self.locals.get_unchecked_mut(fp) = value;
                    }
                }
                Opcode::ReassignLocal1 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    unsafe {
                        *self.locals.get_unchecked_mut(fp + 1) = value;
                    }
                }
                Opcode::ReassignLocal2 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    unsafe {
                        *self.locals.get_unchecked_mut(fp + 2) = value;
                    }
                }
                Opcode::ReassignLocal3 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    unsafe {
                        *self.locals.get_unchecked_mut(fp + 3) = value;
                    }
                }
                Opcode::ReassignLocal4 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    unsafe {
                        *self.locals.get_unchecked_mut(fp + 4) = value;
                    }
                }
                Opcode::ReassignLocal5 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    unsafe {
                        *self.locals.get_unchecked_mut(fp + 5) = value;
                    }
                }
                Opcode::ReassignLocal6 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    unsafe {
                        *self.locals.get_unchecked_mut(fp + 6) = value;
                    }
                }
                Opcode::ReassignLocal7 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    unsafe {
                        *self.locals.get_unchecked_mut(fp + 7) = value;
                    }
                }
                Opcode::ReassignLocal8 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    unsafe {
                        *self.locals.get_unchecked_mut(fp + 8) = value;
                    }
                }
                Opcode::ReassignLocal9 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    unsafe {
                        *self.locals.get_unchecked_mut(fp + 9) = value;
                    }
                }
                Opcode::ReassignLocal10 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    unsafe {
                        *self.locals.get_unchecked_mut(fp + 10) = value;
                    }
                }
                Opcode::ReassignLocal11 => {
                    let value = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    unsafe {
                        *self.locals.get_unchecked_mut(fp + 11) = value;
                    }
                }
                Opcode::AddAssignLocal(index) => {
                    let rhs = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    let idx = fp + index as usize;
                    let current = unsafe { *self.locals.get_unchecked(idx) };
                    let result = self.add_values(current, rhs, Opcode::Add, span)?;
                    unsafe {
                        *self.locals.get_unchecked_mut(idx) = result;
                    }
                }
                Opcode::AddAssignGlobal(index) => {
                    let rhs = self.pop_unchecked();
                    let current = self.load_global_value_fast(index as usize, span)?;
                    let result = self.add_values(current, rhs, Opcode::Add, span)?;
                    self.store_global_value_fast(index as usize, result, span)?;
                }
                Opcode::AddAssignLocalInt(index) => {
                    let rhs = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    let idx = fp + index as usize;
                    let current = unsafe { *self.locals.get_unchecked(idx) };
                    match (current, rhs) {
                        (Value::Int(current), Value::Int(rhs)) => unsafe {
                            *self.locals.get_unchecked_mut(idx) = Value::Int(current + rhs);
                        },
                        (current, rhs) => {
                            return Err(self.construct_err(opcode, current, Some(rhs), span));
                        }
                    }
                }
                Opcode::AddAssignGlobalInt(index) => {
                    let rhs = self.pop_unchecked();
                    let current = self.load_global_value_fast(index as usize, span)?;
                    match (current, rhs) {
                        (Value::Int(current), Value::Int(rhs)) => {
                            self.store_global_value_fast(
                                index as usize,
                                Value::Int(current + rhs),
                                span,
                            )?;
                        }
                        (current, rhs) => {
                            return Err(self.construct_err(opcode, current, Some(rhs), span));
                        }
                    }
                }
                Opcode::ReassignGlobal(index) => {
                    // SAFETY: valid bytecode guarantees stack has a value to assign.
                    let value = self.pop_unchecked();
                    self.store_global_value_fast(index as usize, value, span)?;
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
                                let iter_key = self.get_heap_mut().push_ident("iter");
                                let next_key = self.get_heap_mut().push_ident("next");
                                let heap = self.get_heap();
                                let inst = heap.get_struct_inst(inst_key)?;
                                let struct_def = heap.get_struct_def(inst.struct_def())?;
                                (
                                    struct_def.name().to_string(),
                                    struct_def.get_method(iter_key).cloned(),
                                    struct_def.get_method(next_key).is_some(),
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
                Opcode::CallGlobal(global_index, args) => {
                    let arg_count = args as usize;
                    let global_index = global_index as usize;

                    if self.stack.len() < arg_count {
                        return Err(WalrusError::StackUnderflow {
                            op: opcode,
                            span,
                            src: self.source_ref.source().to_string(),
                            filename: self.source_ref.filename().to_string(),
                        });
                    }

                    if !profiling_enabled && self.current_frame().module_binding.is_none() {
                        enum FastGlobalCall {
                            Vm(Rc<InstructionSet>, Option<Rc<VmModuleBinding>>),
                            Native(NativeFunction),
                            None,
                        }

                        let fast_call = match self
                            .global_call_cache
                            .get(global_index)
                            .and_then(|entry| entry.as_ref())
                        {
                            Some(CachedGlobalCall::Vm {
                                arity,
                                is_async: false,
                                code,
                                module_binding,
                                ..
                            }) if arg_count == *arity => {
                                FastGlobalCall::Vm(Rc::clone(code), module_binding.clone())
                            }
                            Some(CachedGlobalCall::Native { function }) => {
                                FastGlobalCall::Native(*function)
                            }
                            _ => FastGlobalCall::None,
                        };

                        match fast_call {
                            FastGlobalCall::Vm(code, module_binding) => {
                                let new_frame = CallFrame {
                                    return_ip: self.ip,
                                    frame_pointer: self.locals.len(),
                                    stack_pointer: self.stack.len() - arg_count,
                                    instructions: code,
                                    function_name: String::new(),
                                    return_override: None,
                                    module_binding,
                                    awaiting_task: None,
                                };

                                self.call_stack.push(new_frame);

                                let args_start = self.stack.len() - arg_count;
                                self.copy_stack_tail_to_locals(args_start);

                                self.ip = 0;
                                continue;
                            }
                            FastGlobalCall::Native(function) => {
                                let args = self.pop_n(arg_count, opcode, span)?;
                                let result = self.call_native(function, args, span)?;
                                self.push(result);
                                continue;
                            }
                            FastGlobalCall::None => {}
                        }
                    }

                    let func = self.load_global_value_fast(global_index, span)?;

                    // JIT PROFILING: Track function calls and argument types
                    if profiling_enabled {
                        if let Value::Function(key) = func {
                            if let Ok(func_ref) = self.get_heap().get_function(key) {
                                let name = match func_ref {
                                    WalrusFunction::Vm(f) => f.name.clone(),
                                    WalrusFunction::Native(f) => f.name().to_string(),
                                };
                                if self.hotspot_detector.record_function_call(&name) {
                                    debug!("Hot function detected: {}", name);
                                }
                                let args_start = self.stack.len() - arg_count;
                                for (i, arg) in self.stack[args_start..].iter().enumerate() {
                                    let arg_type = WalrusType::from_value(arg);
                                    self.type_profile.observe(self.ip - 1 + i, arg_type);
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

                                    let new_frame = CallFrame {
                                        return_ip: self.ip,
                                        frame_pointer: self.locals.len(),
                                        stack_pointer: self.stack.len() - arg_count,
                                        instructions: Rc::clone(&func.code),
                                        function_name: String::new(),
                                        return_override: None,
                                        module_binding: func.module_binding.clone(),
                                        awaiting_task: None,
                                    };

                                    self.call_stack.push(new_frame);

                                    let args_start = self.stack.len() - arg_count;
                                    self.copy_stack_tail_to_locals(args_start);

                                    self.ip = 0;
                                }
                                WalrusFunction::Native(native_fn) => {
                                    let native_fn = *native_fn;
                                    let args = self.pop_n(arg_count, opcode, span)?;
                                    let result = self.call_native(native_fn, args, span)?;
                                    self.push(result);
                                }
                            }
                        }
                        Value::StructDef(struct_def_key) => {
                            let (struct_name, init_method) = {
                                let init_key = self.get_heap_mut().push_ident("init");
                                let struct_def = self.get_heap().get_struct_def(struct_def_key)?;
                                (
                                    struct_def.name().to_string(),
                                    struct_def.get_method(init_key).cloned(),
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
                                    self.copy_stack_tail_to_locals(args_start);
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
                Opcode::CallGlobal1(global_index) => {
                    let global_index = global_index as usize;

                    if self.stack.is_empty() {
                        return Err(WalrusError::StackUnderflow {
                            op: opcode,
                            span,
                            src: self.source_ref.source().to_string(),
                            filename: self.source_ref.filename().to_string(),
                        });
                    }

                    if !profiling_enabled && self.current_frame().module_binding.is_none() {
                        enum FastGlobalCall1 {
                            Vm(Rc<InstructionSet>, Option<Rc<VmModuleBinding>>),
                            Native(NativeFunction),
                            None,
                        }

                        let fast_call = match self
                            .global_call_cache
                            .get(global_index)
                            .and_then(|entry| entry.as_ref())
                        {
                            Some(CachedGlobalCall::Vm {
                                arity: 1,
                                is_async: false,
                                code,
                                module_binding,
                            }) => FastGlobalCall1::Vm(Rc::clone(code), module_binding.clone()),
                            Some(CachedGlobalCall::Native { function }) => {
                                FastGlobalCall1::Native(*function)
                            }
                            _ => FastGlobalCall1::None,
                        };

                        match fast_call {
                            FastGlobalCall1::Vm(code, module_binding) => {
                                let arg = self.pop_unchecked();
                                let new_frame = CallFrame {
                                    return_ip: self.ip,
                                    frame_pointer: self.locals.len(),
                                    stack_pointer: self.stack.len(),
                                    instructions: code,
                                    function_name: String::new(),
                                    return_override: None,
                                    module_binding,
                                    awaiting_task: None,
                                };

                                self.call_stack.push(new_frame);
                                self.locals.push(arg);
                                self.ip = 0;
                                continue;
                            }
                            FastGlobalCall1::Native(function) => {
                                let arg = self.pop_unchecked();
                                let result = self.call_native(function, vec![arg], span)?;
                                self.push(result);
                                continue;
                            }
                            FastGlobalCall1::None => {}
                        }
                    }

                    let func = self.load_global_value_fast(global_index, span)?;

                    if profiling_enabled {
                        if let Value::Function(key) = func {
                            if let Ok(func_ref) = self.get_heap().get_function(key) {
                                let name = match func_ref {
                                    WalrusFunction::Vm(f) => f.name.clone(),
                                    WalrusFunction::Native(f) => f.name().to_string(),
                                };
                                if self.hotspot_detector.record_function_call(&name) {
                                    debug!("Hot function detected: {}", name);
                                }
                                let arg_type = WalrusType::from_value(
                                    self.stack.last().expect("checked stack len above"),
                                );
                                self.type_profile.observe(self.ip - 1, arg_type);
                            }
                        }
                    }

                    match func {
                        Value::Function(key) => {
                            let func = self.get_heap().get_function(key)?.clone();

                            match func {
                                WalrusFunction::Vm(func) => {
                                    if func.arity != 1 {
                                        return Err(WalrusError::InvalidArgCount {
                                            name: func.name.clone(),
                                            expected: func.arity,
                                            got: 1,
                                            span,
                                            src: self.source_ref.source().into(),
                                            filename: self.source_ref.filename().into(),
                                        });
                                    }

                                    if func.is_async {
                                        let arg = self.pop_unchecked();
                                        let task =
                                            self.create_task_for_function_key(key, vec![arg]);
                                        self.push(task);
                                        continue;
                                    }

                                    let arg = self.pop_unchecked();
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
                                    self.locals.push(arg);
                                    self.ip = 0;
                                }
                                WalrusFunction::Native(native_fn) => {
                                    let arg = self.pop_unchecked();
                                    let result = self.call_native(native_fn, vec![arg], span)?;
                                    self.push(result);
                                }
                            }
                        }
                        Value::StructDef(struct_def_key) => {
                            let (struct_name, init_method) = {
                                let init_key = self.get_heap_mut().push_ident("init");
                                let struct_def = self.get_heap().get_struct_def(struct_def_key)?;
                                (
                                    struct_def.name().to_string(),
                                    struct_def.get_method(init_key).cloned(),
                                )
                            };

                            match init_method {
                                Some(WalrusFunction::Vm(init_func)) => {
                                    let expected_without_self = init_func.arity.saturating_sub(1);
                                    if expected_without_self != 1 {
                                        return Err(WalrusError::InvalidArgCount {
                                            name: format!("{}::init", struct_name),
                                            expected: expected_without_self,
                                            got: 1,
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
                                    let arg = self.pop_unchecked();

                                    let new_frame = CallFrame {
                                        return_ip: self.ip,
                                        frame_pointer: self.locals.len(),
                                        stack_pointer: self.stack.len(),
                                        instructions: Rc::clone(&init_func.code),
                                        function_name: format!("{}::init", struct_name),
                                        return_override: Some(instance_value),
                                        module_binding: init_func.module_binding.clone(),
                                        awaiting_task: None,
                                    };

                                    self.call_stack.push(new_frame);
                                    self.locals.push(instance_value);
                                    self.locals.push(arg);
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
                                    self.pop_unchecked();
                                    return Err(WalrusError::InvalidArgCount {
                                        name: struct_name,
                                        expected: 0,
                                        got: 1,
                                        span,
                                        src: self.source_ref.source().into(),
                                        filename: self.source_ref.filename().into(),
                                    });
                                }
                            }
                        }
                        _ => {
                            self.pop_unchecked();
                            return Err(WalrusError::NotCallable {
                                value: func.get_type().to_string(),
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
                                    WalrusFunction::Native(f) => f.name().to_string(),
                                };
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
                                    self.copy_stack_tail_to_locals(args_start);

                                    // Start execution at the beginning of the new function
                                    self.ip = 0;
                                }
                                WalrusFunction::Native(native_fn) => {
                                    let native_fn = *native_fn;
                                    let args = self.pop_n(arg_count, opcode, span)?;
                                    let result = self.call_native(native_fn, args, span)?;
                                    self.push(result);
                                }
                            }
                        }
                        Value::StructDef(struct_def_key) => {
                            let (struct_name, init_method) = {
                                let init_key = self.get_heap_mut().push_ident("init");
                                let struct_def = self.get_heap().get_struct_def(struct_def_key)?;
                                (
                                    struct_def.name().to_string(),
                                    struct_def.get_method(init_key).cloned(),
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
                                    self.copy_stack_tail_to_locals(args_start);
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
                                    self.copy_stack_tail_to_locals(args_start);

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
                            }
                        }
                        Value::StructDef(struct_def_key) => {
                            let (struct_name, init_method) = {
                                let init_key = self.get_heap_mut().push_ident("init");
                                let struct_def = self.get_heap().get_struct_def(struct_def_key)?;
                                (
                                    struct_def.name().to_string(),
                                    struct_def.get_method(init_key).cloned(),
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
                                    self.copy_stack_tail_to_locals(args_start);

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
                Opcode::AddInt1 => {
                    let a = self.pop_unchecked();
                    if let Value::Int(a) = a {
                        self.push(Value::Int(a + 1));
                    } else {
                        return Err(WalrusError::TypeMismatch {
                            expected: "int".to_string(),
                            found: a.get_type().to_string(),
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
                Opcode::SubtractInt1 => {
                    let a = self.pop_unchecked();
                    if let Value::Int(a) = a {
                        self.push(Value::Int(a - 1));
                    } else {
                        return Err(WalrusError::TypeMismatch {
                            expected: "int".to_string(),
                            found: a.get_type().to_string(),
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        });
                    }
                }
                Opcode::SubtractInt2 => {
                    let a = self.pop_unchecked();
                    if let Value::Int(a) = a {
                        self.push(Value::Int(a - 2));
                    } else {
                        return Err(WalrusError::TypeMismatch {
                            expected: "int".to_string(),
                            found: a.get_type().to_string(),
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        });
                    }
                }
                Opcode::MultiplyInt => {
                    let b = self.pop_unchecked();
                    let a = self.pop_unchecked();
                    if let (Value::Int(a), Value::Int(b)) = (a, b) {
                        self.push(Value::Int(a * b));
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
                Opcode::DivideInt => {
                    let b = self.pop_unchecked();
                    let a = self.pop_unchecked();
                    if let (Value::Int(a), Value::Int(b)) = (a, b) {
                        if b == 0 {
                            return Err(WalrusError::DivisionByZero {
                                span,
                                src: self.source_ref.source().to_string(),
                                filename: self.source_ref.filename().to_string(),
                            });
                        }
                        self.push(Value::Int(a / b));
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
                Opcode::ModuloInt => {
                    let b = self.pop_unchecked();
                    let a = self.pop_unchecked();
                    if let (Value::Int(a), Value::Int(b)) = (a, b) {
                        if b == 0 {
                            return Err(WalrusError::DivisionByZero {
                                span,
                                src: self.source_ref.source().to_string(),
                                filename: self.source_ref.filename().to_string(),
                            });
                        }
                        self.push(Value::Int(a % b));
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
                Opcode::LessEqualInt => {
                    let b = self.pop_unchecked();
                    let a = self.pop_unchecked();
                    if let (Value::Int(a), Value::Int(b)) = (a, b) {
                        self.push(Value::Bool(a <= b));
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
                Opcode::LessEqualInt1 => {
                    let a = self.pop_unchecked();
                    if let Value::Int(a) = a {
                        self.push(Value::Bool(a <= 1));
                    } else {
                        return Err(WalrusError::TypeMismatch {
                            expected: "int".to_string(),
                            found: a.get_type().to_string(),
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

                    let result = self.add_values(a, b, opcode, span)?;
                    self.push(result);
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

                            let value = self.get_heap_mut().push_string_owned(s);
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
                    self.index_value(a, b, opcode, span)?;
                }
                Opcode::IndexConst(index) => {
                    let object = self.pop_unchecked();
                    let key = self.current_frame().instructions.get_constant(index);
                    self.index_value(object, key, opcode, span)?;
                }
                Opcode::IndexLocal(local_idx) => {
                    let index = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    let object = unsafe { *self.locals.get_unchecked(fp + local_idx as usize) };
                    self.index_value(object, index, opcode, span)?;
                }
                Opcode::IndexLocalConst(local_idx, const_idx) => {
                    let fp = self.frame_pointer();
                    let object = unsafe { *self.locals.get_unchecked(fp + local_idx as usize) };
                    let key = self.current_frame().instructions.get_constant(const_idx);
                    self.index_value(object, key, opcode, span)?;
                }
                Opcode::StoreIndex => {
                    let value = self.pop_unchecked();
                    let index = self.pop_unchecked();
                    let object = self.pop_unchecked();
                    self.store_index_value(object, index, value, span)?;
                }
                Opcode::StoreIndexLocal(local_idx) => {
                    let value = self.pop_unchecked();
                    let index = self.pop_unchecked();
                    let fp = self.frame_pointer();
                    let object = unsafe { *self.locals.get_unchecked(fp + local_idx as usize) };
                    self.store_index_value(object, index, value, span)?;
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
                    let value = self.get_heap_mut().push_string_owned(s);
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
                Opcode::ListPush => {
                    let value = self.pop_unchecked();
                    let object = self.pop_unchecked();
                    match object {
                        Value::List(list_key) => {
                            self.get_heap_mut().get_mut_list(list_key)?.push(value);
                            self.push(Value::Void);
                        }
                        other => {
                            return Err(WalrusError::InvalidMethodReceiver {
                                method: "push".to_string(),
                                type_name: other.get_type().to_string(),
                                span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            });
                        }
                    }
                }
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
                            let method_clone = {
                                let struct_def = self.get_heap().get_struct_def(struct_def_key)?;
                                if let Some(method) = struct_def.get_method(method_name_sym) {
                                    method.clone()
                                } else {
                                    let method_name =
                                        self.get_heap().get_string(method_name_sym)?;
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
                                    if let Some(method) = struct_def.get_method(member_name_sym) {
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
                                let (method_arity, method_code, method_binding) = {
                                    let struct_def = self.get_heap().get_struct_def(key)?;
                                    match struct_def.get_method(method_name_sym) {
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
                                            let method_name =
                                                self.get_heap().get_string(method_name_sym)?;
                                            return Err(WalrusError::MethodNotFound {
                                                type_name: struct_def.name().to_string(),
                                                method: method_name.to_string(),
                                                span,
                                                src: self.source_ref.source().into(),
                                                filename: self.source_ref.filename().into(),
                                            });
                                        }
                                    }
                                };

                                if arg_count != method_arity {
                                    let method_name =
                                        self.get_heap().get_string(method_name_sym)?.to_string();
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
                                self.copy_stack_tail_to_locals(args_start);
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
                                        struct_def.get_method(method_name_sym).cloned(),
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
                                self.copy_stack_tail_to_locals(args_start);
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
                                if let Some(method) = struct_def.get_method(method_name_sym) {
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
                                    struct_def.get_method(method_name_sym).cloned(),
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

    fn add_values(
        &mut self,
        a: Value,
        b: Value,
        opcode: Opcode,
        span: Span,
    ) -> WalrusResult<Value> {
        match (a, b) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                Ok(Value::Float(FloatOrd(a + b)))
            }
            (Value::Int(a), Value::Float(FloatOrd(b))) => Ok(Value::Float(FloatOrd(a as f64 + b))),
            (Value::Float(FloatOrd(a)), Value::Int(b)) => Ok(Value::Float(FloatOrd(a + b as f64))),
            (Value::String(a), Value::String(b)) => {
                let a = self.get_heap().get_string(a)?;
                let b = self.get_heap().get_string(b)?;

                let mut s = String::with_capacity(a.len() + b.len());
                s.push_str(a);
                s.push_str(b);

                Ok(self.get_heap_mut().push_string_owned(s))
            }
            (Value::List(a), Value::List(b)) => {
                let mut a = self.get_heap().get_list(a)?.to_vec();
                let b = self.get_heap().get_list(b)?;
                a.extend(b);

                Ok(self.get_heap_mut().push(HeapValue::List(a)))
            }
            (Value::Dict(a), Value::Dict(b)) => {
                let mut a = self.get_heap().get_dict(a)?.clone();
                let b = self.get_heap().get_dict(b)?;
                a.extend(b);

                Ok(self.get_heap_mut().push(HeapValue::Dict(a)))
            }
            _ => Err(self.construct_err(opcode, a, Some(b), span)),
        }
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

    fn index_value(
        &mut self,
        object: Value,
        index: Value,
        opcode: Opcode,
        span: Span,
    ) -> WalrusResult<()> {
        // Fast path for the hottest case in numeric code: list[int]
        if let (Value::List(list_key), Value::Int(idx)) = (object, index) {
            let list = self.get_heap().get_list(list_key)?;
            if idx >= 0 {
                let idx_usize = idx as usize;
                if idx_usize < list.len() {
                    // SAFETY: bounds checked above
                    self.push(unsafe { *list.get_unchecked(idx_usize) });
                    return Ok(());
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
            return Ok(());
        }

        match (object, index) {
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
                let value = self.get_heap_mut().push_string_owned(res);

                self.push(value);
            }
            (Value::Dict(a), b) => {
                let a = self.get_heap().get_dict(a)?;

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

                if start < 0 || end < 0 || start as usize > a_len || end as usize > a_len {
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

                let start_byte = Self::char_to_byte_offset(a, start).ok_or_else(|| {
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
                let value = self.get_heap_mut().push_string_owned(res);

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

                if start < 0 || end < 0 || start as usize > a_len || end as usize > a_len {
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
            (a, b) => return Err(self.construct_err(opcode, a, Some(b), span)),
        }

        Ok(())
    }

    fn store_index_value(
        &mut self,
        object: Value,
        index: Value,
        value: Value,
        span: Span,
    ) -> WalrusResult<()> {
        if let (Value::List(list_key), Value::Int(idx)) = (object, index) {
            let list = self.get_heap_mut().get_mut_list(list_key)?;
            if idx >= 0 {
                let idx_usize = idx as usize;
                if idx_usize < list.len() {
                    unsafe {
                        *list.get_unchecked_mut(idx_usize) = value;
                    }
                    self.push(Value::Void);
                    return Ok(());
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
            return Ok(());
        }

        match (object, index) {
            (Value::List(_), Value::Int(_)) => unreachable!(),
            (Value::Dict(dict_key), key) => {
                let dict = self.get_heap_mut().get_mut_dict(dict_key)?;
                dict.insert(key, value);
                self.push(Value::Void);
                Ok(())
            }
            (Value::StructInst(inst_key), Value::String(member_key)) => {
                let member_name = self.get_heap().get_string(member_key)?.to_string();
                let inst = self.get_heap_mut().get_mut_struct_inst(inst_key)?;
                inst.set_field(member_name, value);
                self.push(Value::Void);
                Ok(())
            }
            _ => Err(WalrusError::InvalidIndexType {
                non_indexable: object.get_type().to_string(),
                index_type: index.get_type().to_string(),
                span,
                src: self.source_ref.source().to_string(),
                filename: self.source_ref.filename().to_string(),
            }),
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
}
