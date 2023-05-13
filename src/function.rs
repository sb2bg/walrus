use crate::ast::Node;
use crate::interpreter::{Interpreter, InterpreterResult};
use crate::source_ref::SourceRef;
use crate::span::Span;
use crate::value::Value;
use crate::vm::instruction_set::InstructionSet;

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

pub struct NodeFunction {
    pub name: String,
    pub args: Vec<String>,
    pub body: Node,
}

impl NodeFunction {
    pub fn new(name: String, args: Vec<String>, node: Node) -> Self {
        Self {
            name,
            args,
            body: node,
        }
    }
}

pub struct VmFunction {
    pub name: String,
    pub arity: usize,
    chunk: InstructionSet,
}

impl VmFunction {
    pub fn new(name: String, arity: usize, chunk: InstructionSet) -> Self {
        Self { name, arity, chunk }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn arity(&self) -> usize {
        self.arity
    }

    pub fn chunk(&self) -> &InstructionSet {
        &self.chunk
    }

    pub fn into_chunk(self) -> InstructionSet {
        self.chunk
    }
}

pub enum WalrusFunction {
    Rust(RustFunction),
    TreeWalk(NodeFunction),
    Vm(VmFunction),
}

impl ToString for WalrusFunction {
    fn to_string(&self) -> String {
        match self {
            WalrusFunction::Rust(func) => format!("<builtin_function({})>", func.name),
            WalrusFunction::TreeWalk(func) => format!("<function({})>", func.name),
            WalrusFunction::Vm(func) => format!("<function({})>", func.name),
        }
    }
}

impl PartialEq for WalrusFunction {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (WalrusFunction::Rust(f1), WalrusFunction::Rust(f2)) => cmp(f1, f2),
            (WalrusFunction::TreeWalk(f1), WalrusFunction::TreeWalk(f2)) => cmp(f1, f2),
            (WalrusFunction::Vm(f1), WalrusFunction::Vm(f2)) => cmp(f1, f2),
            _ => false,
        }
    }
}

fn cmp<T>(a1: &T, a2: &T) -> bool {
    a1 as *const _ == a2 as *const _ // todo: is this correct?
}
