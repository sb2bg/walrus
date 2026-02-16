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
use crate::jit::{HotSpotDetector, TypeProfile, WalrusType};
use crate::range::RangeValue;
use crate::source_ref::SourceRef;
use crate::span::Span;
use crate::value::Value;
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
#[derive(Debug)]
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
    ip: usize,                  // Current instruction pointer
    gc_poll_counter: u32,       // Throttle GC checks to avoid per-instruction overhead
    globals: Vec<Value>,
    source_ref: SourceRef<'a>,
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
        };

        Self {
            stack: Vec::new(),
            locals: Vec::new(),
            call_stack: vec![main_frame],
            ip: 0,
            gc_poll_counter: 0,
            globals: Vec::new(),
            source_ref,
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

    /// Helper to access heap - uses thread-local ARENA
    ///
    /// # Safety
    /// Uses unsafe get_arena_ptr() for performance. This is safe because:
    /// - The VM is single-threaded
    /// - We only access the arena within VM methods (no escaping references)
    /// - No concurrent borrows occur within VM execution
    #[inline]
    fn get_heap(&self) -> &crate::arenas::ValueHolder {
        unsafe { &*crate::arenas::get_arena_ptr() }
    }

    /// Helper to access heap mutably - uses thread-local ARENA
    ///
    /// # Safety
    /// See get_heap() for safety rationale.
    #[inline]
    fn get_heap_mut(&mut self) -> &mut crate::arenas::ValueHolder {
        unsafe { &mut *crate::arenas::get_arena_ptr() }
    }

    /// Collect all root values that the GC needs to trace from
    fn collect_roots(&self) -> Vec<Value> {
        let mut roots = Vec::with_capacity(self.stack.len() + self.locals.len() + 64);

        // Stack values are roots
        roots.extend(self.stack.iter().copied());

        // Local variables are roots
        roots.extend(self.locals.iter().copied());

        // Global variables are roots
        roots.extend(self.globals.iter().copied());

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
            NativeFunction::Cwd => match crate::stdlib::cwd() {
                Some(path) => {
                    let v = self.get_heap_mut().push(HeapValue::String(&path));
                    Ok(v)
                }
                None => Ok(Value::Void),
            },
            NativeFunction::MathPi => Ok(Value::Float(FloatOrd(std::f64::consts::PI))),
            NativeFunction::MathE => Ok(Value::Float(FloatOrd(std::f64::consts::E))),
            NativeFunction::MathTau => Ok(Value::Float(FloatOrd(std::f64::consts::TAU))),
            NativeFunction::MathInf => Ok(Value::Float(FloatOrd(f64::INFINITY))),
            NativeFunction::MathNaN => Ok(Value::Float(FloatOrd(f64::NAN))),
            NativeFunction::MathAbs => match args[0] {
                Value::Int(n) => {
                    let abs = n.checked_abs().ok_or_else(|| WalrusError::GenericError {
                        message: "math.abs: overflow for i64::MIN".to_string(),
                    })?;
                    Ok(Value::Int(abs))
                }
                Value::Float(FloatOrd(n)) => Ok(Value::Float(FloatOrd(n.abs()))),
                other => Err(WalrusError::TypeMismatch {
                    expected: "number".to_string(),
                    found: other.get_type().to_string(),
                    span,
                    src: self.source_ref.source().into(),
                    filename: self.source_ref.filename().into(),
                }),
            },
            NativeFunction::MathSign => match args[0] {
                Value::Int(n) => Ok(Value::Int(n.signum())),
                Value::Float(FloatOrd(n)) => {
                    if n.is_nan() {
                        return Err(WalrusError::GenericError {
                            message: "math.sign: cannot determine sign of NaN".to_string(),
                        });
                    }
                    let sign = if n > 0.0 {
                        1
                    } else if n < 0.0 {
                        -1
                    } else {
                        0
                    };
                    Ok(Value::Int(sign))
                }
                other => Err(WalrusError::TypeMismatch {
                    expected: "number".to_string(),
                    found: other.get_type().to_string(),
                    span,
                    src: self.source_ref.source().into(),
                    filename: self.source_ref.filename().into(),
                }),
            },
            NativeFunction::MathMin => match (args[0], args[1]) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.min(b))),
                (a, b) => {
                    let a = self.value_to_number(a, span)?;
                    let b = self.value_to_number(b, span)?;
                    Ok(Value::Float(FloatOrd(a.min(b))))
                }
            },
            NativeFunction::MathMax => match (args[0], args[1]) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.max(b))),
                (a, b) => {
                    let a = self.value_to_number(a, span)?;
                    let b = self.value_to_number(b, span)?;
                    Ok(Value::Float(FloatOrd(a.max(b))))
                }
            },
            NativeFunction::MathClamp => match (args[0], args[1], args[2]) {
                (Value::Int(v), Value::Int(min), Value::Int(max)) => {
                    if min > max {
                        return Err(WalrusError::GenericError {
                            message: format!(
                                "math.clamp: min ({min}) cannot be greater than max ({max})"
                            ),
                        });
                    }
                    Ok(Value::Int(v.clamp(min, max)))
                }
                (v, min, max) => {
                    let v = self.value_to_number(v, span)?;
                    let min = self.value_to_number(min, span)?;
                    let max = self.value_to_number(max, span)?;
                    if v.is_nan() || min.is_nan() || max.is_nan() {
                        return Err(WalrusError::GenericError {
                            message: "math.clamp: NaN is not supported".to_string(),
                        });
                    }
                    if min > max {
                        return Err(WalrusError::GenericError {
                            message: format!(
                                "math.clamp: min ({min}) cannot be greater than max ({max})"
                            ),
                        });
                    }
                    let clamped = if v < min {
                        min
                    } else if v > max {
                        max
                    } else {
                        v
                    };
                    Ok(Value::Float(FloatOrd(clamped)))
                }
            },
            NativeFunction::MathFloor => match args[0] {
                Value::Int(n) => Ok(Value::Int(n)),
                value => Ok(Value::Float(FloatOrd(
                    self.value_to_number(value, span)?.floor(),
                ))),
            },
            NativeFunction::MathCeil => match args[0] {
                Value::Int(n) => Ok(Value::Int(n)),
                value => Ok(Value::Float(FloatOrd(
                    self.value_to_number(value, span)?.ceil(),
                ))),
            },
            NativeFunction::MathRound => match args[0] {
                Value::Int(n) => Ok(Value::Int(n)),
                value => Ok(Value::Float(FloatOrd(
                    self.value_to_number(value, span)?.round(),
                ))),
            },
            NativeFunction::MathTrunc => match args[0] {
                Value::Int(n) => Ok(Value::Int(n)),
                value => Ok(Value::Float(FloatOrd(
                    self.value_to_number(value, span)?.trunc(),
                ))),
            },
            NativeFunction::MathFract => {
                let value = self.value_to_number(args[0], span)?;
                Ok(Value::Float(FloatOrd(value.fract())))
            }
            NativeFunction::MathSqrt => {
                let value = self.value_to_number(args[0], span)?;
                if value < 0.0 {
                    return Err(WalrusError::GenericError {
                        message: format!("math.sqrt: domain error for value {value}"),
                    });
                }
                Ok(Value::Float(FloatOrd(value.sqrt())))
            }
            NativeFunction::MathCbrt => {
                let value = self.value_to_number(args[0], span)?;
                Ok(Value::Float(FloatOrd(value.cbrt())))
            }
            NativeFunction::MathPow => {
                let base = self.value_to_number(args[0], span)?;
                let exponent = self.value_to_number(args[1], span)?;
                let result = base.powf(exponent);
                if !result.is_finite() {
                    return Err(WalrusError::GenericError {
                        message: "math.pow: result is not finite".to_string(),
                    });
                }
                Ok(Value::Float(FloatOrd(result)))
            }
            NativeFunction::MathHypot => {
                let x = self.value_to_number(args[0], span)?;
                let y = self.value_to_number(args[1], span)?;
                let result = x.hypot(y);
                if !result.is_finite() {
                    return Err(WalrusError::GenericError {
                        message: "math.hypot: result is not finite".to_string(),
                    });
                }
                Ok(Value::Float(FloatOrd(result)))
            }
            NativeFunction::MathSin => {
                let value = self.value_to_number(args[0], span)?;
                Ok(Value::Float(FloatOrd(value.sin())))
            }
            NativeFunction::MathCos => {
                let value = self.value_to_number(args[0], span)?;
                Ok(Value::Float(FloatOrd(value.cos())))
            }
            NativeFunction::MathTan => {
                let value = self.value_to_number(args[0], span)?;
                Ok(Value::Float(FloatOrd(value.tan())))
            }
            NativeFunction::MathAsin => {
                let value = self.value_to_number(args[0], span)?;
                if !(-1.0..=1.0).contains(&value) {
                    return Err(WalrusError::GenericError {
                        message: format!("math.asin: domain error for value {value}"),
                    });
                }
                Ok(Value::Float(FloatOrd(value.asin())))
            }
            NativeFunction::MathAcos => {
                let value = self.value_to_number(args[0], span)?;
                if !(-1.0..=1.0).contains(&value) {
                    return Err(WalrusError::GenericError {
                        message: format!("math.acos: domain error for value {value}"),
                    });
                }
                Ok(Value::Float(FloatOrd(value.acos())))
            }
            NativeFunction::MathAtan => {
                let value = self.value_to_number(args[0], span)?;
                Ok(Value::Float(FloatOrd(value.atan())))
            }
            NativeFunction::MathAtan2 => {
                let y = self.value_to_number(args[0], span)?;
                let x = self.value_to_number(args[1], span)?;
                Ok(Value::Float(FloatOrd(y.atan2(x))))
            }
            NativeFunction::MathExp => {
                let value = self.value_to_number(args[0], span)?;
                let result = value.exp();
                if !result.is_finite() {
                    return Err(WalrusError::GenericError {
                        message: "math.exp: result is not finite".to_string(),
                    });
                }
                Ok(Value::Float(FloatOrd(result)))
            }
            NativeFunction::MathLn => {
                let value = self.value_to_number(args[0], span)?;
                if value <= 0.0 {
                    return Err(WalrusError::GenericError {
                        message: format!("math.ln: domain error for value {value}"),
                    });
                }
                Ok(Value::Float(FloatOrd(value.ln())))
            }
            NativeFunction::MathLog2 => {
                let value = self.value_to_number(args[0], span)?;
                if value <= 0.0 {
                    return Err(WalrusError::GenericError {
                        message: format!("math.log2: domain error for value {value}"),
                    });
                }
                Ok(Value::Float(FloatOrd(value.log2())))
            }
            NativeFunction::MathLog10 => {
                let value = self.value_to_number(args[0], span)?;
                if value <= 0.0 {
                    return Err(WalrusError::GenericError {
                        message: format!("math.log10: domain error for value {value}"),
                    });
                }
                Ok(Value::Float(FloatOrd(value.log10())))
            }
            NativeFunction::MathLog => {
                let value = self.value_to_number(args[0], span)?;
                let base = self.value_to_number(args[1], span)?;
                if value <= 0.0 {
                    return Err(WalrusError::GenericError {
                        message: format!("math.log: domain error for value {value}"),
                    });
                }
                if base <= 0.0 || (base - 1.0).abs() < f64::EPSILON {
                    return Err(WalrusError::GenericError {
                        message: format!("math.log: invalid base {base}"),
                    });
                }
                Ok(Value::Float(FloatOrd(value.log(base))))
            }
            NativeFunction::MathLerp => {
                let a = self.value_to_number(args[0], span)?;
                let b = self.value_to_number(args[1], span)?;
                let t = self.value_to_number(args[2], span)?;
                let result = a + (b - a) * t;
                if !result.is_finite() {
                    return Err(WalrusError::GenericError {
                        message: "math.lerp: result is not finite".to_string(),
                    });
                }
                Ok(Value::Float(FloatOrd(result)))
            }
            NativeFunction::MathDegrees => {
                let value = self.value_to_number(args[0], span)?;
                Ok(Value::Float(FloatOrd(value.to_degrees())))
            }
            NativeFunction::MathRadians => {
                let value = self.value_to_number(args[0], span)?;
                Ok(Value::Float(FloatOrd(value.to_radians())))
            }
            NativeFunction::MathIsFinite => {
                let value = self.value_to_number(args[0], span)?;
                Ok(Value::Bool(value.is_finite()))
            }
            NativeFunction::MathIsNaN => {
                let value = self.value_to_number(args[0], span)?;
                Ok(Value::Bool(value.is_nan()))
            }
            NativeFunction::MathIsInf => {
                let value = self.value_to_number(args[0], span)?;
                Ok(Value::Bool(value.is_infinite()))
            }
            NativeFunction::MathSeed => {
                let seed = self.value_to_int(args[0], span)?;
                crate::stdlib::math_seed(seed);
                Ok(Value::Void)
            }
            NativeFunction::MathRandFloat => {
                Ok(Value::Float(FloatOrd(crate::stdlib::math_rand_float())))
            }
            NativeFunction::MathRandBool => Ok(Value::Bool(crate::stdlib::math_rand_bool())),
            NativeFunction::MathRandInt => {
                let min = self.value_to_int(args[0], span)?;
                let max = self.value_to_int(args[1], span)?;
                Ok(Value::Int(crate::stdlib::math_rand_int(min, max, span)?))
            }
            NativeFunction::MathRandRange => {
                let min = self.value_to_number(args[0], span)?;
                let max = self.value_to_number(args[1], span)?;
                Ok(Value::Float(FloatOrd(crate::stdlib::math_rand_range(
                    min, max, span,
                )?)))
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
    fn value_to_int(&self, value: Value, span: Span) -> WalrusResult<i64> {
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
    fn value_to_number(&self, value: Value, span: Span) -> WalrusResult<f64> {
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
                    Err(WalrusError::RuntimeErrorWithStackTrace {
                        error: err.to_string(),
                        stack_trace,
                    })
                }
            }
        }
    }

    fn run_inner(&mut self) -> WalrusResult<Value> {
        let debug_logging_enabled = log_enabled!(log::Level::Debug);
        let profiling_enabled = self.profiling_enabled;

        loop {
            // Poll GC periodically instead of every instruction.
            self.gc_poll_counter = self.gc_poll_counter.wrapping_add(1);
            if self.gc_poll_counter & 0xFF == 0 {
                self.maybe_collect_garbage();
            }

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
                    // SAFETY: compiler guarantees global indices are valid.
                    let value = unsafe { *self.globals.get_unchecked(index as usize) };
                    self.push(value);
                }
                Opcode::LoadGlobal0 => {
                    // SAFETY: compiler emits LoadGlobal0 only when global 0 exists.
                    let value = unsafe { *self.globals.get_unchecked(0) };
                    self.push(value);
                }
                Opcode::LoadGlobal1 => {
                    // SAFETY: compiler emits LoadGlobal1 only when global 1 exists.
                    let value = unsafe { *self.globals.get_unchecked(1) };
                    self.push(value);
                }
                Opcode::LoadGlobal2 => {
                    // SAFETY: compiler emits LoadGlobal2 only when global 2 exists.
                    let value = unsafe { *self.globals.get_unchecked(2) };
                    self.push(value);
                }
                Opcode::LoadGlobal3 => {
                    // SAFETY: compiler emits LoadGlobal3 only when global 3 exists.
                    let value = unsafe { *self.globals.get_unchecked(3) };
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
                    let index = index as usize;

                    if index == self.globals.len() {
                        self.globals.push(value);
                    } else {
                        // SAFETY: compiler guarantees global index is valid when not appending.
                        unsafe {
                            *self.globals.get_unchecked_mut(index) = value;
                        }
                    }
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
                    // SAFETY: compiler guarantees reassigned global exists.
                    unsafe {
                        *self.globals.get_unchecked_mut(index as usize) = value;
                    }
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

                    if let Value::Bool(false) = value {
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

                                    // Create a new call frame instead of a child VM
                                    let new_frame = CallFrame {
                                        return_ip: self.ip,                          // Where to return after this function
                                        frame_pointer: self.locals.len(), // New frame starts at current locals end
                                        stack_pointer: self.stack.len() - arg_count, // Operand stack position for cleanup on return
                                        instructions: Rc::clone(&func.code), // Share the instruction set via Rc
                                        function_name: String::new(),
                                        return_override: None,
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

                                    if self.call_stack.is_empty() {
                                        return Ok(result);
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

                                    if self.call_stack.is_empty() {
                                        return Ok(result);
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

                                    // Clone what we need before mutating self
                                    let new_instructions = Rc::clone(&func.code);
                                    let new_name = String::new();

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

                                    if self.call_stack.is_empty() {
                                        return Ok(result);
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
                            return Err(WalrusError::InvalidGcThresholdArg {
                                span,
                                src: self.source_ref.source().to_string(),
                                filename: self.source_ref.filename().to_string(),
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
                                    let key = self
                                        .get_heap_mut()
                                        .push(HeapValue::String(native_fn.name()));
                                    let func = self.get_heap_mut().push(HeapValue::Function(
                                        WalrusFunction::Native(native_fn),
                                    ));
                                    dict.insert(key, func);
                                }
                                let module = self.get_heap_mut().push(HeapValue::Dict(dict));
                                self.push(module);
                            } else {
                                return Err(WalrusError::ModuleNotFound {
                                    module: name_str.to_string(),
                                    span,
                                    src: self.source_ref.source().into(),
                                    filename: self.source_ref.filename().into(),
                                });
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
                Opcode::Return => {
                    let mut return_value = self.pop(opcode, span)?;

                    // Pop the current call frame
                    let frame = self
                        .call_stack
                        .pop()
                        .expect("Call stack should never be empty on return");

                    if let Some(override_value) = frame.return_override {
                        return_value = override_value;
                    }

                    // If this was the last frame (main), return the value
                    if self.call_stack.is_empty() {
                        return Ok(return_value);
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
                        // Dict/Module member access (for imported modules)
                        (Value::String(member_name_sym), Value::Dict(dict_key)) => {
                            let dict = self.get_heap().get_dict(dict_key)?;
                            // Look up by the string key
                            if let Some(&value) = dict.get(&Value::String(member_name_sym)) {
                                self.push(value);
                            } else {
                                let member_name = self.get_heap().get_string(member_name_sym)?;
                                return Err(WalrusError::MemberNotFound {
                                    type_name: "module/dict".to_string(),
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
                                let (method_arity, method_code) = {
                                    let struct_def = self.get_heap().get_struct_def(key)?;
                                    match struct_def.get_method(&method_name) {
                                        Some(WalrusFunction::Vm(func)) => {
                                            (func.arity, Rc::clone(&func.code))
                                        }
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

                                let (method_arity, method_code) = match method {
                                    Some(WalrusFunction::Vm(func)) => {
                                        (func.arity, Rc::clone(&func.code))
                                    }
                                    Some(_) => {
                                        return Err(WalrusError::StructMethodMustBeVmFunction {
                                            span,
                                            src: self.source_ref.source().into(),
                                            filename: self.source_ref.filename().into(),
                                        });
                                    }
                                    None => {
                                        return Err(WalrusError::MethodNotFound {
                                            type_name,
                                            method: method_name,
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
                            // First, check if this dict contains a native function with this name
                            let method_key = Value::String(method_name_sym);

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
                                method_name_sym,
                                args,
                                span,
                                src,
                                filename,
                            )?
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
