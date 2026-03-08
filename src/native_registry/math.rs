use float_ord::FloatOrd;

use crate::WalrusResult;
use crate::error::WalrusError;
use crate::function::NativeFunction;
use crate::span::Span;
use crate::value::Value;
use crate::vm::VM;

use super::{NativeSpec, define_native_spec};

pub static SPECS: &[NativeSpec] = &[
    define_native_spec(
        NativeFunction::MathPi,
        "std/math",
        "pi",
        &[],
        "Return pi.",
        native_math_pi,
    ),
    define_native_spec(
        NativeFunction::MathE,
        "std/math",
        "e",
        &[],
        "Return Euler's number.",
        native_math_e,
    ),
    define_native_spec(
        NativeFunction::MathTau,
        "std/math",
        "tau",
        &[],
        "Return tau (2*pi).",
        native_math_tau,
    ),
    define_native_spec(
        NativeFunction::MathInf,
        "std/math",
        "inf",
        &[],
        "Return positive infinity.",
        native_math_inf,
    ),
    define_native_spec(
        NativeFunction::MathNaN,
        "std/math",
        "nan",
        &[],
        "Return NaN.",
        native_math_nan,
    ),
    define_native_spec(
        NativeFunction::MathAbs,
        "std/math",
        "abs",
        &["x"],
        "Absolute value.",
        native_math_abs,
    ),
    define_native_spec(
        NativeFunction::MathSign,
        "std/math",
        "sign",
        &["x"],
        "Sign as -1, 0, or 1.",
        native_math_sign,
    ),
    define_native_spec(
        NativeFunction::MathMin,
        "std/math",
        "min",
        &["a", "b"],
        "Minimum of two numbers.",
        native_math_min,
    ),
    define_native_spec(
        NativeFunction::MathMax,
        "std/math",
        "max",
        &["a", "b"],
        "Maximum of two numbers.",
        native_math_max,
    ),
    define_native_spec(
        NativeFunction::MathClamp,
        "std/math",
        "clamp",
        &["x", "lo", "hi"],
        "Clamp to [lo, hi].",
        native_math_clamp,
    ),
    define_native_spec(
        NativeFunction::MathFloor,
        "std/math",
        "floor",
        &["x"],
        "Round down to integer.",
        native_math_floor,
    ),
    define_native_spec(
        NativeFunction::MathCeil,
        "std/math",
        "ceil",
        &["x"],
        "Round up to integer.",
        native_math_ceil,
    ),
    define_native_spec(
        NativeFunction::MathRound,
        "std/math",
        "round",
        &["x"],
        "Round to nearest integer.",
        native_math_round,
    ),
    define_native_spec(
        NativeFunction::MathTrunc,
        "std/math",
        "trunc",
        &["x"],
        "Truncate fractional component.",
        native_math_trunc,
    ),
    define_native_spec(
        NativeFunction::MathFract,
        "std/math",
        "fract",
        &["x"],
        "Fractional component.",
        native_math_fract,
    ),
    define_native_spec(
        NativeFunction::MathSqrt,
        "std/math",
        "sqrt",
        &["x"],
        "Square root (x >= 0).",
        native_math_sqrt,
    ),
    define_native_spec(
        NativeFunction::MathCbrt,
        "std/math",
        "cbrt",
        &["x"],
        "Cube root.",
        native_math_cbrt,
    ),
    define_native_spec(
        NativeFunction::MathPow,
        "std/math",
        "pow",
        &["x", "y"],
        "Raise x to power y.",
        native_math_pow,
    ),
    define_native_spec(
        NativeFunction::MathHypot,
        "std/math",
        "hypot",
        &["x", "y"],
        "Euclidean norm sqrt(x*x+y*y).",
        native_math_hypot,
    ),
    define_native_spec(
        NativeFunction::MathSin,
        "std/math",
        "sin",
        &["x"],
        "Sine in radians.",
        native_math_sin,
    ),
    define_native_spec(
        NativeFunction::MathCos,
        "std/math",
        "cos",
        &["x"],
        "Cosine in radians.",
        native_math_cos,
    ),
    define_native_spec(
        NativeFunction::MathTan,
        "std/math",
        "tan",
        &["x"],
        "Tangent in radians.",
        native_math_tan,
    ),
    define_native_spec(
        NativeFunction::MathAsin,
        "std/math",
        "asin",
        &["x"],
        "Inverse sine for x in [-1,1].",
        native_math_asin,
    ),
    define_native_spec(
        NativeFunction::MathAcos,
        "std/math",
        "acos",
        &["x"],
        "Inverse cosine for x in [-1,1].",
        native_math_acos,
    ),
    define_native_spec(
        NativeFunction::MathAtan,
        "std/math",
        "atan",
        &["x"],
        "Inverse tangent.",
        native_math_atan,
    ),
    define_native_spec(
        NativeFunction::MathAtan2,
        "std/math",
        "atan2",
        &["y", "x"],
        "Quadrant-aware inverse tangent.",
        native_math_atan2,
    ),
    define_native_spec(
        NativeFunction::MathExp,
        "std/math",
        "exp",
        &["x"],
        "e^x.",
        native_math_exp,
    ),
    define_native_spec(
        NativeFunction::MathLn,
        "std/math",
        "ln",
        &["x"],
        "Natural log for x > 0.",
        native_math_ln,
    ),
    define_native_spec(
        NativeFunction::MathLog2,
        "std/math",
        "log2",
        &["x"],
        "Base-2 log for x > 0.",
        native_math_log2,
    ),
    define_native_spec(
        NativeFunction::MathLog10,
        "std/math",
        "log10",
        &["x"],
        "Base-10 log for x > 0.",
        native_math_log10,
    ),
    define_native_spec(
        NativeFunction::MathLog,
        "std/math",
        "log",
        &["x", "base"],
        "Log in a custom base.",
        native_math_log,
    ),
    define_native_spec(
        NativeFunction::MathLerp,
        "std/math",
        "lerp",
        &["a", "b", "t"],
        "Linear interpolation between a and b.",
        native_math_lerp,
    ),
    define_native_spec(
        NativeFunction::MathDegrees,
        "std/math",
        "degrees",
        &["r"],
        "Radians to degrees.",
        native_math_degrees,
    ),
    define_native_spec(
        NativeFunction::MathRadians,
        "std/math",
        "radians",
        &["d"],
        "Degrees to radians.",
        native_math_radians,
    ),
    define_native_spec(
        NativeFunction::MathIsFinite,
        "std/math",
        "is_finite",
        &["x"],
        "True if finite.",
        native_math_is_finite,
    ),
    define_native_spec(
        NativeFunction::MathIsNaN,
        "std/math",
        "is_nan",
        &["x"],
        "True if NaN.",
        native_math_is_nan,
    ),
    define_native_spec(
        NativeFunction::MathIsInf,
        "std/math",
        "is_inf",
        &["x"],
        "True if infinite.",
        native_math_is_inf,
    ),
    define_native_spec(
        NativeFunction::MathSeed,
        "std/math",
        "seed",
        &["n"],
        "Seed RNG state.",
        native_math_seed,
    ),
    define_native_spec(
        NativeFunction::MathRandFloat,
        "std/math",
        "rand_float",
        &[],
        "Random float in [0.0,1.0).",
        native_math_rand_float,
    ),
    define_native_spec(
        NativeFunction::MathRandBool,
        "std/math",
        "rand_bool",
        &[],
        "Random boolean.",
        native_math_rand_bool,
    ),
    define_native_spec(
        NativeFunction::MathRandInt,
        "std/math",
        "rand_int",
        &["a", "b"],
        "Random integer in [a,b].",
        native_math_rand_int,
    ),
    define_native_spec(
        NativeFunction::MathRandRange,
        "std/math",
        "rand_range",
        &["a", "b"],
        "Random float in [a,b).",
        native_math_rand_range,
    ),
];
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
