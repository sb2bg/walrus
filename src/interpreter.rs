use crate::ast::{Node, Op};
use crate::error::GlassError;
use crate::scope::Scope;

pub struct Interpreter {
    scope: Scope,
}

pub type InterpreterResult = Result<(), GlassError>;

impl Interpreter {
    pub fn new() -> Self {
        Self {
            scope: Scope::new(),
        }
    }

    pub fn interpret(&self, ast: Node) -> InterpreterResult {
        self.visit(ast)
    }

    fn visit(&self, node: Node) -> InterpreterResult {
        todo!()
    }

    fn visit_bin_op(&self, op: Op, left: Node, right: Node) -> InterpreterResult {
        let left = self.visit(left)?;
        let right = self.visit(right)?;

        match op {
            _ => todo!(),
        }
    }

    fn visit_unary_op(&self, op: Op, right: Node) -> InterpreterResult {
        todo!()
    }

    fn visit_num(&self, num: f64) -> InterpreterResult {
        todo!()
    }

    fn visit_bool(&self, bool: bool) -> InterpreterResult {
        todo!()
    }

    fn visit_str(&self, string: String) -> InterpreterResult {
        todo!()
    }
}
