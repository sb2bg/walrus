use crate::interpreter::InterpreterResult;
use crate::source_ref::SourceRef;
use crate::span::Span;
use crate::value::Value;

pub struct RustFunction {
    name: String,
    args: Option<usize>,
    func: fn(Vec<Value>, SourceRef, span: Span) -> InterpreterResult,
}

impl RustFunction {
    pub fn new(
        name: String,
        args: Option<usize>,
        func: fn(Vec<Value>, SourceRef, Span) -> InterpreterResult,
    ) -> Self {
        Self { name, args, func }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn args(&self) -> Option<usize> {
        self.args
    }

    pub fn call(&self, args: Vec<Value>, source_ref: SourceRef, span: Span) -> InterpreterResult {
        (self.func)(args, source_ref, span)
    }
}
