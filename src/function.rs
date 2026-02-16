use rustc_hash::FxHashMap;

use crate::ast::Node;
use crate::interpreter::InterpreterResult;
use crate::source_ref::SourceRef;
use crate::span::Span;
use crate::value::Value;
use crate::vm::instruction_set::InstructionSet;
use std::fmt::Display;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct RustFunction {
    pub name: String,
    pub args: usize,
    func: fn(Vec<Value>, SourceRef, span: Span) -> InterpreterResult,
}

impl RustFunction {
    pub fn new(
        name: String,
        args: usize,
        func: fn(Vec<Value>, SourceRef, Span) -> InterpreterResult,
    ) -> Self {
        Self { name, args, func }
    }

    pub fn call(&self, args: Vec<Value>, source_ref: SourceRef, span: Span) -> InterpreterResult {
        (self.func)(args, source_ref, span)
    }
}

#[derive(Debug, Clone)]
pub struct NodeFunction {
    pub name: String,
    pub args: Vec<String>,
    pub body: Node,
    /// Captured variables from the enclosing scope (for closures)
    pub captures: FxHashMap<String, Value>,
}

impl NodeFunction {
    pub fn new(name: String, args: Vec<String>, node: Node) -> Self {
        Self {
            name,
            args,
            body: node,
            captures: FxHashMap::default(),
        }
    }

    pub fn new_with_captures(
        name: String,
        args: Vec<String>,
        node: Node,
        captures: FxHashMap<String, Value>,
    ) -> Self {
        Self {
            name,
            args,
            body: node,
            captures,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VmFunction {
    pub name: String,
    pub arity: usize,
    pub code: Rc<InstructionSet>,
}

impl VmFunction {
    pub fn new(name: String, arity: usize, code: InstructionSet) -> Self {
        Self {
            name,
            arity,
            code: Rc::new(code),
        }
    }
}

#[derive(Debug, Clone)]
pub enum WalrusFunction {
    Rust(RustFunction),
    TreeWalk(NodeFunction),
    Vm(VmFunction),
    Native(NativeFunction),
}

/// Native function ID for stdlib functions callable from VM
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeFunction {
    // File I/O
    FileOpen,
    FileRead,
    FileReadLine,
    FileWrite,
    FileClose,
    FileExists,
    ReadFile,
    WriteFile,
    // System
    EnvGet,
    Args,
    Cwd,
    Exit,
    // Math
    MathPi,
    MathE,
    MathTau,
    MathInf,
    MathNaN,
    MathAbs,
    MathSign,
    MathMin,
    MathMax,
    MathClamp,
    MathFloor,
    MathCeil,
    MathRound,
    MathTrunc,
    MathFract,
    MathSqrt,
    MathCbrt,
    MathPow,
    MathHypot,
    MathSin,
    MathCos,
    MathTan,
    MathAsin,
    MathAcos,
    MathAtan,
    MathAtan2,
    MathExp,
    MathLn,
    MathLog2,
    MathLog10,
    MathLog,
    MathLerp,
    MathDegrees,
    MathRadians,
    MathIsFinite,
    MathIsNaN,
    MathIsInf,
    MathSeed,
    MathRandFloat,
    MathRandBool,
    MathRandInt,
    MathRandRange,
}

impl NativeFunction {
    pub fn name(&self) -> &'static str {
        match self {
            NativeFunction::FileOpen => "file_open",
            NativeFunction::FileRead => "file_read",
            NativeFunction::FileReadLine => "file_read_line",
            NativeFunction::FileWrite => "file_write",
            NativeFunction::FileClose => "file_close",
            NativeFunction::FileExists => "file_exists",
            NativeFunction::ReadFile => "read_file",
            NativeFunction::WriteFile => "write_file",
            NativeFunction::EnvGet => "env_get",
            NativeFunction::Args => "args",
            NativeFunction::Cwd => "cwd",
            NativeFunction::Exit => "exit",
            NativeFunction::MathPi => "pi",
            NativeFunction::MathE => "e",
            NativeFunction::MathTau => "tau",
            NativeFunction::MathInf => "inf",
            NativeFunction::MathNaN => "nan",
            NativeFunction::MathAbs => "abs",
            NativeFunction::MathSign => "sign",
            NativeFunction::MathMin => "min",
            NativeFunction::MathMax => "max",
            NativeFunction::MathClamp => "clamp",
            NativeFunction::MathFloor => "floor",
            NativeFunction::MathCeil => "ceil",
            NativeFunction::MathRound => "round",
            NativeFunction::MathTrunc => "trunc",
            NativeFunction::MathFract => "fract",
            NativeFunction::MathSqrt => "sqrt",
            NativeFunction::MathCbrt => "cbrt",
            NativeFunction::MathPow => "pow",
            NativeFunction::MathHypot => "hypot",
            NativeFunction::MathSin => "sin",
            NativeFunction::MathCos => "cos",
            NativeFunction::MathTan => "tan",
            NativeFunction::MathAsin => "asin",
            NativeFunction::MathAcos => "acos",
            NativeFunction::MathAtan => "atan",
            NativeFunction::MathAtan2 => "atan2",
            NativeFunction::MathExp => "exp",
            NativeFunction::MathLn => "ln",
            NativeFunction::MathLog2 => "log2",
            NativeFunction::MathLog10 => "log10",
            NativeFunction::MathLog => "log",
            NativeFunction::MathLerp => "lerp",
            NativeFunction::MathDegrees => "degrees",
            NativeFunction::MathRadians => "radians",
            NativeFunction::MathIsFinite => "is_finite",
            NativeFunction::MathIsNaN => "is_nan",
            NativeFunction::MathIsInf => "is_inf",
            NativeFunction::MathSeed => "seed",
            NativeFunction::MathRandFloat => "rand_float",
            NativeFunction::MathRandBool => "rand_bool",
            NativeFunction::MathRandInt => "rand_int",
            NativeFunction::MathRandRange => "rand_range",
        }
    }

    pub fn arity(&self) -> usize {
        match self {
            NativeFunction::FileOpen => 2,
            NativeFunction::FileRead => 1,
            NativeFunction::FileReadLine => 1,
            NativeFunction::FileWrite => 2,
            NativeFunction::FileClose => 1,
            NativeFunction::FileExists => 1,
            NativeFunction::ReadFile => 1,
            NativeFunction::WriteFile => 2,
            NativeFunction::EnvGet => 1,
            NativeFunction::Args => 0,
            NativeFunction::Cwd => 0,
            NativeFunction::Exit => 1,
            NativeFunction::MathPi => 0,
            NativeFunction::MathE => 0,
            NativeFunction::MathTau => 0,
            NativeFunction::MathInf => 0,
            NativeFunction::MathNaN => 0,
            NativeFunction::MathAbs => 1,
            NativeFunction::MathSign => 1,
            NativeFunction::MathMin => 2,
            NativeFunction::MathMax => 2,
            NativeFunction::MathClamp => 3,
            NativeFunction::MathFloor => 1,
            NativeFunction::MathCeil => 1,
            NativeFunction::MathRound => 1,
            NativeFunction::MathTrunc => 1,
            NativeFunction::MathFract => 1,
            NativeFunction::MathSqrt => 1,
            NativeFunction::MathCbrt => 1,
            NativeFunction::MathPow => 2,
            NativeFunction::MathHypot => 2,
            NativeFunction::MathSin => 1,
            NativeFunction::MathCos => 1,
            NativeFunction::MathTan => 1,
            NativeFunction::MathAsin => 1,
            NativeFunction::MathAcos => 1,
            NativeFunction::MathAtan => 1,
            NativeFunction::MathAtan2 => 2,
            NativeFunction::MathExp => 1,
            NativeFunction::MathLn => 1,
            NativeFunction::MathLog2 => 1,
            NativeFunction::MathLog10 => 1,
            NativeFunction::MathLog => 2,
            NativeFunction::MathLerp => 3,
            NativeFunction::MathDegrees => 1,
            NativeFunction::MathRadians => 1,
            NativeFunction::MathIsFinite => 1,
            NativeFunction::MathIsNaN => 1,
            NativeFunction::MathIsInf => 1,
            NativeFunction::MathSeed => 1,
            NativeFunction::MathRandFloat => 0,
            NativeFunction::MathRandBool => 0,
            NativeFunction::MathRandInt => 2,
            NativeFunction::MathRandRange => 2,
        }
    }
}

impl Display for WalrusFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                WalrusFunction::Rust(func) => format!("<builtin_function({})>", func.name),
                WalrusFunction::TreeWalk(func) => format!("<function({})>", func.name),
                WalrusFunction::Vm(func) => format!("<function({})>", func.name),
                WalrusFunction::Native(func) => format!("<native_function({})>", func.name()),
            }
        )
    }
}

impl PartialEq for WalrusFunction {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (WalrusFunction::Rust(f1), WalrusFunction::Rust(f2)) => cmp(f1, f2),
            (WalrusFunction::TreeWalk(f1), WalrusFunction::TreeWalk(f2)) => cmp(f1, f2),
            (WalrusFunction::Vm(f1), WalrusFunction::Vm(f2)) => cmp(f1, f2),
            (WalrusFunction::Native(f1), WalrusFunction::Native(f2)) => f1 == f2,
            _ => false,
        }
    }
}

fn cmp<T>(a1: &T, a2: &T) -> bool {
    std::ptr::eq(a1, a2)
}
