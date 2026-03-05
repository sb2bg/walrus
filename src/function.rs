use rustc_hash::FxHashMap;

use crate::arenas::DictKey;
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
    pub is_async: bool,
    /// Captured variables from the enclosing scope (for closures)
    pub captures: FxHashMap<String, Value>,
}

impl NodeFunction {
    pub fn new(name: String, args: Vec<String>, node: Node) -> Self {
        Self::new_with_async(name, args, node, false)
    }

    pub fn new_async(name: String, args: Vec<String>, node: Node) -> Self {
        Self::new_with_async(name, args, node, true)
    }

    fn new_with_async(name: String, args: Vec<String>, node: Node, is_async: bool) -> Self {
        Self {
            name,
            args,
            body: node,
            is_async,
            captures: FxHashMap::default(),
        }
    }

    pub fn new_with_captures(
        name: String,
        args: Vec<String>,
        node: Node,
        is_async: bool,
        captures: FxHashMap<String, Value>,
    ) -> Self {
        Self {
            name,
            args,
            body: node,
            is_async,
            captures,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VmModuleBinding {
    pub module_key: DictKey,
    pub global_names: Rc<Vec<String>>,
    pub global_values: Rc<Vec<Value>>,
    pub source: Rc<String>,
    pub filename: Rc<String>,
}

#[derive(Debug, Clone)]
pub struct VmFunction {
    pub name: String,
    pub arity: usize,
    pub is_async: bool,
    pub code: Rc<InstructionSet>,
    pub module_binding: Option<Rc<VmModuleBinding>>,
}

impl VmFunction {
    pub fn new(name: String, arity: usize, code: InstructionSet) -> Self {
        Self::new_with_async(name, arity, code, false)
    }

    pub fn new_async(name: String, arity: usize, code: InstructionSet) -> Self {
        Self::new_with_async(name, arity, code, true)
    }

    fn new_with_async(name: String, arity: usize, code: InstructionSet, is_async: bool) -> Self {
        Self {
            name,
            arity,
            is_async,
            code: Rc::new(code),
            module_binding: None,
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
    // Core
    CoreLen,
    CoreStr,
    CoreType,
    CoreInput,
    CoreGc,
    CoreHeapStats,
    CoreGcThreshold,
    // Async primitives
    AsyncSpawn,
    AsyncSleep,
    AsyncTimeout,
    AsyncGather,
    AsyncCancel,
    AsyncCancelled,
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
    // Networking
    NetTcpBind,
    NetTcpAccept,
    NetTcpConnect,
    NetTcpLocalPort,
    NetTcpRead,
    NetTcpReadLine,
    NetTcpWrite,
    NetTcpClose,
    NetTcpCloseListener,
    // HTTP
    HttpParseRequestLine,
    HttpParseQuery,
    HttpNormalizePath,
    HttpMatchRoute,
    HttpStatusText,
    HttpResponse,
    HttpResponseWithHeaders,
    HttpReadRequest,
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
        crate::native_registry::native_spec(*self).name
    }

    pub fn arity(&self) -> usize {
        crate::native_registry::native_spec(*self).arity
    }

    pub fn module(&self) -> &'static str {
        crate::native_registry::native_spec(*self).module
    }

    pub fn params(&self) -> &'static [&'static str] {
        crate::native_registry::native_spec(*self).params
    }

    pub fn docs(&self) -> &'static str {
        crate::native_registry::native_spec(*self).docs
    }
}

impl Display for WalrusFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                WalrusFunction::Rust(func) => format!("<builtin_function({})>", func.name),
                WalrusFunction::TreeWalk(func) => {
                    if func.is_async {
                        format!("<async_function({})>", func.name)
                    } else {
                        format!("<function({})>", func.name)
                    }
                }
                WalrusFunction::Vm(func) => {
                    if func.is_async {
                        format!("<async_function({})>", func.name)
                    } else {
                        format!("<function({})>", func.name)
                    }
                }
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
