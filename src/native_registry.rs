use float_ord::FloatOrd;

use crate::arenas::HeapValue;
use crate::error::WalrusError;
use crate::function::NativeFunction;
use crate::span::Span;
use crate::value::Value;
use crate::vm::VM;
use crate::WalrusResult;

pub type NativeHandler = for<'a> fn(&mut VM<'a>, &[Value], Span) -> WalrusResult<Value>;

#[derive(Clone, Copy)]
pub struct NativeSpec {
    pub id: NativeFunction,
    pub module: &'static str,
    pub name: &'static str,
    pub arity: usize,
    pub params: &'static [&'static str],
    pub docs: &'static str,
    pub handler: NativeHandler,
}

macro_rules! count_params {
    () => {
        0usize
    };
    ($head:literal $(, $tail:literal)*) => {
        1usize + count_params!($($tail),*)
    };
}

macro_rules! native_spec {
    (
        $id:ident,
        $module:literal,
        $name:literal,
        [$($param:literal),* $(,)?],
        $docs:literal,
        $handler:path
    ) => {
        NativeSpec {
            id: NativeFunction::$id,
            module: $module,
            name: $name,
            arity: count_params!($($param),*),
            params: &[$($param),*],
            docs: $docs,
            handler: $handler,
        }
    };
}

pub static NATIVE_SPECS: &[NativeSpec] = &[
    native_spec!(
        FileOpen,
        "std/io",
        "file_open",
        ["path", "mode"],
        "Open a file and return a handle.",
        native_file_open
    ),
    native_spec!(
        FileRead,
        "std/io",
        "file_read",
        ["handle"],
        "Read entire contents from an open file handle.",
        native_file_read
    ),
    native_spec!(
        FileReadLine,
        "std/io",
        "file_read_line",
        ["handle"],
        "Read one line from an open file handle.",
        native_file_read_line
    ),
    native_spec!(
        FileWrite,
        "std/io",
        "file_write",
        ["handle", "content"],
        "Write string content to an open file handle.",
        native_file_write
    ),
    native_spec!(
        FileClose,
        "std/io",
        "file_close",
        ["handle"],
        "Close an open file handle.",
        native_file_close
    ),
    native_spec!(
        FileExists,
        "std/io",
        "file_exists",
        ["path"],
        "Return true if the path exists.",
        native_file_exists
    ),
    native_spec!(
        ReadFile,
        "std/io",
        "read_file",
        ["path"],
        "Read a file into a string.",
        native_read_file
    ),
    native_spec!(
        WriteFile,
        "std/io",
        "write_file",
        ["path", "content"],
        "Write a string to a file.",
        native_write_file
    ),
    native_spec!(
        EnvGet,
        "std/sys",
        "env_get",
        ["name"],
        "Get an environment variable by name.",
        native_env_get
    ),
    native_spec!(
        Args,
        "std/sys",
        "args",
        [],
        "Get command-line arguments.",
        native_args
    ),
    native_spec!(
        Cwd,
        "std/sys",
        "cwd",
        [],
        "Get the current working directory.",
        native_cwd
    ),
    native_spec!(
        Exit,
        "std/sys",
        "exit",
        ["code"],
        "Exit the process with a status code.",
        native_exit
    ),
    native_spec!(MathPi, "std/math", "pi", [], "Return pi.", native_math_pi),
    native_spec!(
        MathE,
        "std/math",
        "e",
        [],
        "Return Euler's number.",
        native_math_e
    ),
    native_spec!(
        MathTau,
        "std/math",
        "tau",
        [],
        "Return tau (2*pi).",
        native_math_tau
    ),
    native_spec!(
        MathInf,
        "std/math",
        "inf",
        [],
        "Return positive infinity.",
        native_math_inf
    ),
    native_spec!(
        MathNaN,
        "std/math",
        "nan",
        [],
        "Return NaN.",
        native_math_nan
    ),
    native_spec!(
        MathAbs,
        "std/math",
        "abs",
        ["x"],
        "Absolute value.",
        native_math_abs
    ),
    native_spec!(
        MathSign,
        "std/math",
        "sign",
        ["x"],
        "Sign as -1, 0, or 1.",
        native_math_sign
    ),
    native_spec!(
        MathMin,
        "std/math",
        "min",
        ["a", "b"],
        "Minimum of two numbers.",
        native_math_min
    ),
    native_spec!(
        MathMax,
        "std/math",
        "max",
        ["a", "b"],
        "Maximum of two numbers.",
        native_math_max
    ),
    native_spec!(
        MathClamp,
        "std/math",
        "clamp",
        ["x", "lo", "hi"],
        "Clamp to [lo, hi].",
        native_math_clamp
    ),
    native_spec!(
        MathFloor,
        "std/math",
        "floor",
        ["x"],
        "Round down to integer.",
        native_math_floor
    ),
    native_spec!(
        MathCeil,
        "std/math",
        "ceil",
        ["x"],
        "Round up to integer.",
        native_math_ceil
    ),
    native_spec!(
        MathRound,
        "std/math",
        "round",
        ["x"],
        "Round to nearest integer.",
        native_math_round
    ),
    native_spec!(
        MathTrunc,
        "std/math",
        "trunc",
        ["x"],
        "Truncate fractional component.",
        native_math_trunc
    ),
    native_spec!(
        MathFract,
        "std/math",
        "fract",
        ["x"],
        "Fractional component.",
        native_math_fract
    ),
    native_spec!(
        MathSqrt,
        "std/math",
        "sqrt",
        ["x"],
        "Square root (x >= 0).",
        native_math_sqrt
    ),
    native_spec!(
        MathCbrt,
        "std/math",
        "cbrt",
        ["x"],
        "Cube root.",
        native_math_cbrt
    ),
    native_spec!(
        MathPow,
        "std/math",
        "pow",
        ["x", "y"],
        "Raise x to power y.",
        native_math_pow
    ),
    native_spec!(
        MathHypot,
        "std/math",
        "hypot",
        ["x", "y"],
        "Euclidean norm sqrt(x*x+y*y).",
        native_math_hypot
    ),
    native_spec!(
        MathSin,
        "std/math",
        "sin",
        ["x"],
        "Sine in radians.",
        native_math_sin
    ),
    native_spec!(
        MathCos,
        "std/math",
        "cos",
        ["x"],
        "Cosine in radians.",
        native_math_cos
    ),
    native_spec!(
        MathTan,
        "std/math",
        "tan",
        ["x"],
        "Tangent in radians.",
        native_math_tan
    ),
    native_spec!(
        MathAsin,
        "std/math",
        "asin",
        ["x"],
        "Inverse sine for x in [-1,1].",
        native_math_asin
    ),
    native_spec!(
        MathAcos,
        "std/math",
        "acos",
        ["x"],
        "Inverse cosine for x in [-1,1].",
        native_math_acos
    ),
    native_spec!(
        MathAtan,
        "std/math",
        "atan",
        ["x"],
        "Inverse tangent.",
        native_math_atan
    ),
    native_spec!(
        MathAtan2,
        "std/math",
        "atan2",
        ["y", "x"],
        "Quadrant-aware inverse tangent.",
        native_math_atan2
    ),
    native_spec!(MathExp, "std/math", "exp", ["x"], "e^x.", native_math_exp),
    native_spec!(
        MathLn,
        "std/math",
        "ln",
        ["x"],
        "Natural log for x > 0.",
        native_math_ln
    ),
    native_spec!(
        MathLog2,
        "std/math",
        "log2",
        ["x"],
        "Base-2 log for x > 0.",
        native_math_log2
    ),
    native_spec!(
        MathLog10,
        "std/math",
        "log10",
        ["x"],
        "Base-10 log for x > 0.",
        native_math_log10
    ),
    native_spec!(
        MathLog,
        "std/math",
        "log",
        ["x", "base"],
        "Log in a custom base.",
        native_math_log
    ),
    native_spec!(
        MathLerp,
        "std/math",
        "lerp",
        ["a", "b", "t"],
        "Linear interpolation between a and b.",
        native_math_lerp
    ),
    native_spec!(
        MathDegrees,
        "std/math",
        "degrees",
        ["r"],
        "Radians to degrees.",
        native_math_degrees
    ),
    native_spec!(
        MathRadians,
        "std/math",
        "radians",
        ["d"],
        "Degrees to radians.",
        native_math_radians
    ),
    native_spec!(
        MathIsFinite,
        "std/math",
        "is_finite",
        ["x"],
        "True if finite.",
        native_math_is_finite
    ),
    native_spec!(
        MathIsNaN,
        "std/math",
        "is_nan",
        ["x"],
        "True if NaN.",
        native_math_is_nan
    ),
    native_spec!(
        MathIsInf,
        "std/math",
        "is_inf",
        ["x"],
        "True if infinite.",
        native_math_is_inf
    ),
    native_spec!(
        MathSeed,
        "std/math",
        "seed",
        ["n"],
        "Seed RNG state.",
        native_math_seed
    ),
    native_spec!(
        MathRandFloat,
        "std/math",
        "rand_float",
        [],
        "Random float in [0.0,1.0).",
        native_math_rand_float
    ),
    native_spec!(
        MathRandBool,
        "std/math",
        "rand_bool",
        [],
        "Random boolean.",
        native_math_rand_bool
    ),
    native_spec!(
        MathRandInt,
        "std/math",
        "rand_int",
        ["a", "b"],
        "Random integer in [a,b].",
        native_math_rand_int
    ),
    native_spec!(
        MathRandRange,
        "std/math",
        "rand_range",
        ["a", "b"],
        "Random float in [a,b).",
        native_math_rand_range
    ),
];

pub fn native_spec(function: NativeFunction) -> &'static NativeSpec {
    NATIVE_SPECS
        .iter()
        .find(|spec| spec.id == function)
        .expect("native function not registered")
}

pub fn module_functions(module: &str) -> Option<Vec<NativeFunction>> {
    let functions: Vec<NativeFunction> = NATIVE_SPECS
        .iter()
        .filter(|spec| spec.module == module)
        .map(|spec| spec.id)
        .collect();

    if functions.is_empty() {
        None
    } else {
        Some(functions)
    }
}

pub fn dispatch_native(
    vm: &mut VM<'_>,
    native_fn: NativeFunction,
    args: Vec<Value>,
    span: Span,
) -> WalrusResult<Value> {
    let spec = native_spec(native_fn);

    if args.len() != spec.arity {
        return Err(WalrusError::InvalidArgCount {
            name: spec.name.to_string(),
            expected: spec.arity,
            got: args.len(),
            span,
            src: vm.source_ref().source().into(),
            filename: vm.source_ref().filename().into(),
        });
    }

    (spec.handler)(vm, &args, span)
}

fn native_file_open(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let path = vm.value_to_string(args[0], span)?;
    let mode = vm.value_to_string(args[1], span)?;
    crate::stdlib::file_open(&path, &mode, span)
}

fn native_file_read(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let handle = vm.value_to_int(args[0], span)?;
    let content = crate::stdlib::file_read(handle, span)?;
    Ok(vm.get_heap_mut().push(HeapValue::String(&content)))
}

fn native_file_read_line(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let handle = vm.value_to_int(args[0], span)?;
    match crate::stdlib::file_read_line(handle, span)? {
        Some(line) => Ok(vm.get_heap_mut().push(HeapValue::String(&line))),
        None => Ok(Value::Void),
    }
}

fn native_file_write(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let handle = vm.value_to_int(args[0], span)?;
    let content = vm.value_to_string(args[1], span)?;
    let bytes = crate::stdlib::file_write(handle, &content, span)?;
    Ok(Value::Int(bytes))
}

fn native_file_close(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let handle = vm.value_to_int(args[0], span)?;
    crate::stdlib::file_close(handle, span)?;
    Ok(Value::Void)
}

fn native_file_exists(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let path = vm.value_to_string(args[0], span)?;
    Ok(Value::Bool(crate::stdlib::file_exists(&path)))
}

fn native_read_file(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let path = vm.value_to_string(args[0], span)?;
    let content = crate::stdlib::read_file(&path, span)?;
    Ok(vm.get_heap_mut().push(HeapValue::String(&content)))
}

fn native_write_file(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let path = vm.value_to_string(args[0], span)?;
    let content = vm.value_to_string(args[1], span)?;
    crate::stdlib::write_file(&path, &content, span)?;
    Ok(Value::Void)
}

fn native_env_get(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let name = vm.value_to_string(args[0], span)?;
    match crate::stdlib::env_get(&name) {
        Some(value) => Ok(vm.get_heap_mut().push(HeapValue::String(&value))),
        None => Ok(Value::Void),
    }
}

fn native_args(vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    let cli_args = crate::stdlib::args();
    let mut list = Vec::with_capacity(cli_args.len());
    for arg in cli_args {
        let s = vm.get_heap_mut().push(HeapValue::String(&arg));
        list.push(s);
    }
    Ok(vm.get_heap_mut().push(HeapValue::List(list)))
}

fn native_cwd(vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    match crate::stdlib::cwd() {
        Some(path) => Ok(vm.get_heap_mut().push(HeapValue::String(&path))),
        None => Ok(Value::Void),
    }
}

fn native_exit(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let code = vm.value_to_int(args[0], span)?;
    std::process::exit(code as i32);
}

fn native_math_pi(_vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    Ok(Value::Float(FloatOrd(std::f64::consts::PI)))
}

fn native_math_e(_vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    Ok(Value::Float(FloatOrd(std::f64::consts::E)))
}

fn native_math_tau(_vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    Ok(Value::Float(FloatOrd(std::f64::consts::TAU)))
}

fn native_math_inf(_vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    Ok(Value::Float(FloatOrd(f64::INFINITY)))
}

fn native_math_nan(_vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    Ok(Value::Float(FloatOrd(f64::NAN)))
}

fn native_math_abs(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    match args[0] {
        Value::Int(n) => {
            let abs = n.checked_abs().ok_or_else(|| WalrusError::GenericError {
                message: "math.abs: overflow for i64::MIN".to_string(),
            })?;
            Ok(Value::Int(abs))
        }
        _ => {
            let n = vm.value_to_number(args[0], span)?;
            Ok(Value::Float(FloatOrd(n.abs())))
        }
    }
}

fn native_math_sign(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    match args[0] {
        Value::Int(n) => Ok(Value::Int(n.signum())),
        _ => {
            let n = vm.value_to_number(args[0], span)?;
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
    }
}

fn native_math_min(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    match (args[0], args[1]) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.min(b))),
        _ => {
            let a = vm.value_to_number(args[0], span)?;
            let b = vm.value_to_number(args[1], span)?;
            Ok(Value::Float(FloatOrd(a.min(b))))
        }
    }
}

fn native_math_max(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    match (args[0], args[1]) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.max(b))),
        _ => {
            let a = vm.value_to_number(args[0], span)?;
            let b = vm.value_to_number(args[1], span)?;
            Ok(Value::Float(FloatOrd(a.max(b))))
        }
    }
}

fn native_math_clamp(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    match (args[0], args[1], args[2]) {
        (Value::Int(v), Value::Int(min), Value::Int(max)) => {
            if min > max {
                return Err(WalrusError::GenericError {
                    message: format!("math.clamp: min ({min}) cannot be greater than max ({max})"),
                });
            }
            Ok(Value::Int(v.clamp(min, max)))
        }
        _ => {
            let v = vm.value_to_number(args[0], span)?;
            let min = vm.value_to_number(args[1], span)?;
            let max = vm.value_to_number(args[2], span)?;
            if v.is_nan() || min.is_nan() || max.is_nan() {
                return Err(WalrusError::GenericError {
                    message: "math.clamp: NaN is not supported".to_string(),
                });
            }
            if min > max {
                return Err(WalrusError::GenericError {
                    message: format!("math.clamp: min ({min}) cannot be greater than max ({max})"),
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
    }
}

fn native_math_floor(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    match args[0] {
        Value::Int(n) => Ok(Value::Int(n)),
        _ => Ok(Value::Float(FloatOrd(
            vm.value_to_number(args[0], span)?.floor(),
        ))),
    }
}

fn native_math_ceil(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    match args[0] {
        Value::Int(n) => Ok(Value::Int(n)),
        _ => Ok(Value::Float(FloatOrd(
            vm.value_to_number(args[0], span)?.ceil(),
        ))),
    }
}

fn native_math_round(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    match args[0] {
        Value::Int(n) => Ok(Value::Int(n)),
        _ => Ok(Value::Float(FloatOrd(
            vm.value_to_number(args[0], span)?.round(),
        ))),
    }
}

fn native_math_trunc(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    match args[0] {
        Value::Int(n) => Ok(Value::Int(n)),
        _ => Ok(Value::Float(FloatOrd(
            vm.value_to_number(args[0], span)?.trunc(),
        ))),
    }
}

fn native_math_fract(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    Ok(Value::Float(FloatOrd(value.fract())))
}

fn native_math_sqrt(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    if value < 0.0 {
        return Err(WalrusError::GenericError {
            message: format!("math.sqrt: domain error for value {value}"),
        });
    }
    Ok(Value::Float(FloatOrd(value.sqrt())))
}

fn native_math_cbrt(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    Ok(Value::Float(FloatOrd(value.cbrt())))
}

fn native_math_pow(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let base = vm.value_to_number(args[0], span)?;
    let exponent = vm.value_to_number(args[1], span)?;
    let result = base.powf(exponent);
    if !result.is_finite() {
        return Err(WalrusError::GenericError {
            message: "math.pow: result is not finite".to_string(),
        });
    }
    Ok(Value::Float(FloatOrd(result)))
}

fn native_math_hypot(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let x = vm.value_to_number(args[0], span)?;
    let y = vm.value_to_number(args[1], span)?;
    let result = x.hypot(y);
    if !result.is_finite() {
        return Err(WalrusError::GenericError {
            message: "math.hypot: result is not finite".to_string(),
        });
    }
    Ok(Value::Float(FloatOrd(result)))
}

fn native_math_sin(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    Ok(Value::Float(FloatOrd(value.sin())))
}

fn native_math_cos(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    Ok(Value::Float(FloatOrd(value.cos())))
}

fn native_math_tan(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    Ok(Value::Float(FloatOrd(value.tan())))
}

fn native_math_asin(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    if !(-1.0..=1.0).contains(&value) {
        return Err(WalrusError::GenericError {
            message: format!("math.asin: domain error for value {value}"),
        });
    }
    Ok(Value::Float(FloatOrd(value.asin())))
}

fn native_math_acos(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    if !(-1.0..=1.0).contains(&value) {
        return Err(WalrusError::GenericError {
            message: format!("math.acos: domain error for value {value}"),
        });
    }
    Ok(Value::Float(FloatOrd(value.acos())))
}

fn native_math_atan(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    Ok(Value::Float(FloatOrd(value.atan())))
}

fn native_math_atan2(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let y = vm.value_to_number(args[0], span)?;
    let x = vm.value_to_number(args[1], span)?;
    Ok(Value::Float(FloatOrd(y.atan2(x))))
}

fn native_math_exp(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    let result = value.exp();
    if !result.is_finite() {
        return Err(WalrusError::GenericError {
            message: "math.exp: result is not finite".to_string(),
        });
    }
    Ok(Value::Float(FloatOrd(result)))
}

fn native_math_ln(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    if value <= 0.0 {
        return Err(WalrusError::GenericError {
            message: format!("math.ln: domain error for value {value}"),
        });
    }
    Ok(Value::Float(FloatOrd(value.ln())))
}

fn native_math_log2(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    if value <= 0.0 {
        return Err(WalrusError::GenericError {
            message: format!("math.log2: domain error for value {value}"),
        });
    }
    Ok(Value::Float(FloatOrd(value.log2())))
}

fn native_math_log10(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    if value <= 0.0 {
        return Err(WalrusError::GenericError {
            message: format!("math.log10: domain error for value {value}"),
        });
    }
    Ok(Value::Float(FloatOrd(value.log10())))
}

fn native_math_log(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    let base = vm.value_to_number(args[1], span)?;
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

fn native_math_lerp(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let a = vm.value_to_number(args[0], span)?;
    let b = vm.value_to_number(args[1], span)?;
    let t = vm.value_to_number(args[2], span)?;
    let result = a + (b - a) * t;
    if !result.is_finite() {
        return Err(WalrusError::GenericError {
            message: "math.lerp: result is not finite".to_string(),
        });
    }
    Ok(Value::Float(FloatOrd(result)))
}

fn native_math_degrees(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    Ok(Value::Float(FloatOrd(value.to_degrees())))
}

fn native_math_radians(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    Ok(Value::Float(FloatOrd(value.to_radians())))
}

fn native_math_is_finite(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    Ok(Value::Bool(value.is_finite()))
}

fn native_math_is_nan(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    Ok(Value::Bool(value.is_nan()))
}

fn native_math_is_inf(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    Ok(Value::Bool(value.is_infinite()))
}

fn native_math_seed(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let seed = vm.value_to_int(args[0], span)?;
    crate::stdlib::math_seed(seed);
    Ok(Value::Void)
}

fn native_math_rand_float(_vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    Ok(Value::Float(FloatOrd(crate::stdlib::math_rand_float())))
}

fn native_math_rand_bool(_vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    Ok(Value::Bool(crate::stdlib::math_rand_bool()))
}

fn native_math_rand_int(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let min = vm.value_to_int(args[0], span)?;
    let max = vm.value_to_int(args[1], span)?;
    Ok(Value::Int(crate::stdlib::math_rand_int(min, max, span)?))
}

fn native_math_rand_range(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let min = vm.value_to_number(args[0], span)?;
    let max = vm.value_to_number(args[1], span)?;
    Ok(Value::Float(FloatOrd(crate::stdlib::math_rand_range(
        min, max, span,
    )?)))
}
