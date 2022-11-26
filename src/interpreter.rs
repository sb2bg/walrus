use crate::ast::{Node, Op};
use crate::error::GlassError;
use crate::scope::Scope;
use crate::source_ref::SourceRef;
use crate::span::Span;
use crate::value::{Value, ValueKind};
use float_ord::FloatOrd;
use std::collections::BTreeMap;

pub struct Interpreter<'a> {
    scope: Scope<'a>,
    source_ref: SourceRef<'a>,
}

pub type InterpreterResult<'a> = Result<Value<'a>, GlassError>;

impl<'a> Interpreter<'a> {
    pub fn new(source_ref: SourceRef<'a>) -> Self {
        Self {
            scope: Scope::new(),
            source_ref,
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
            Node::Int(num, span) => self.make_val(ValueKind::Int(num), span),
            Node::Float(num, span) => self.make_val(ValueKind::Float(FloatOrd(num)), span),
            Node::Bool(boolean, span) => self.make_val(ValueKind::Bool(boolean), span),
            Node::String(string, span) => self.make_val(ValueKind::String(string), span),
            Node::Dict(dict, span) => self.visit_dict(dict, span),
            Node::List(list, span) => self.visit_list(list, span),
            node => Err(GlassError::UnknownError {
                message: format!("Unknown node: {:?}", node),
            }),
        }
    }

    fn make_val(&self, value: ValueKind, span: Span) -> InterpreterResult {
        Ok(Value::new(&self.source_ref, span, value))
    }

    fn visit_statements(&self, nodes: Vec<Box<Node>>) -> InterpreterResult {
        for node in nodes {
            println!("{:?}", self.visit(*node)?);
        }

        self.make_val(ValueKind::Void, Span(0, 0)) // todo: return real span (add span to statements)
    }

    fn visit_bin_op(&self, left: Node, op: Op, right: Node, span: Span) -> InterpreterResult {
        let left = self.visit(left)?;
        let right = self.visit(right)?;

        match op {
            // Op::Add => left.add(right),
            _ => Err(GlassError::UnknownError {
                message: "Unimplemented binary operation".into(),
            }),
        }
    }

    fn visit_unary_op(&self, op: Op, right: Node, span: Span) -> InterpreterResult {
        todo!()
    }

    fn visit_dict(&self, dict: Vec<(Box<Node>, Box<Node>)>, span: Span) -> InterpreterResult {
        let mut map = BTreeMap::new();

        for (key, value) in dict {
            let key = self.visit(*key)?;
            let value = self.visit(*value)?;

            map.insert(key, value);
        }

        self.make_val(ValueKind::Dict(BTreeMap::new()), span) // fixme: return real map
    }

    fn visit_list(&self, list: Vec<Box<Node>>, span: Span) -> InterpreterResult {
        let mut vec = vec![];

        for node in list {
            vec.push(self.visit(*node)?);
        }

        self.make_val(ValueKind::List(vec![]), span) // fixme: return real vec
    }
}
