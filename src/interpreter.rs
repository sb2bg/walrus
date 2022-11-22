use crate::ast::{Node, Op};
use crate::error::GlassError;
use crate::scope::Scope;
use crate::span::Span;

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
        match node {
            Node::Statement(nodes) => self.visit_statements(nodes),
            Node::BinOp(left, op, right, span) => self.visit_bin_op(*left, op, *right, span),
            Node::UnaryOp(op, right, span) => self.visit_unary_op(op, *right, span),
            Node::Int(num, span) => self.visit_int(num, span),
            Node::Float(num, span) => self.visit_float(num, span),
            Node::Bool(boolean, span) => self.visit_bool(boolean, span),
            Node::String(string, span) => self.visit_str(string, span),
            node => Err(GlassError::UnknownError {
                message: format!("Unknown node: {:?}", node),
            }),
        }
    }

    fn visit_statements(&self, nodes: Vec<Box<Node>>) -> InterpreterResult {
        for node in nodes {
            self.visit(*node)?;
        }

        Ok(())
    }

    fn visit_bin_op(&self, left: Node, op: Op, right: Node, span: Span) -> InterpreterResult {
        let left = self.visit(left)?;
        let right = self.visit(right)?;

        match op {
            Op::Add => Ok(()),
            _ => Err(GlassError::UnknownError {
                message: "Unimplemented binary operation".into(),
            }),
        }
    }

    fn visit_unary_op(&self, op: Op, right: Node, span: Span) -> InterpreterResult {
        todo!()
    }

    fn visit_int(&self, num: i64, span: Span) -> InterpreterResult {
        Ok(())
    }

    fn visit_float(&self, num: f64, span: Span) -> InterpreterResult {
        todo!()
    }

    fn visit_bool(&self, bool: bool, span: Span) -> InterpreterResult {
        todo!()
    }

    fn visit_str(&self, string: String, span: Span) -> InterpreterResult {
        todo!()
    }
}
