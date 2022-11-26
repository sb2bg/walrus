use crate::ast::{Node, NodeKind, Op};
use crate::error::{interpreter_err_mapper, GlassError};
use crate::scope::Scope;
use crate::source_ref::SourceRef;
use crate::span::Span;
use crate::value::{Value, ValueKind};
use float_ord::FloatOrd;
use log::debug;
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

    pub fn source_ref(&self) -> &SourceRef<'a> {
        &self.source_ref
    }

    pub fn interpret(&self, ast: Node) -> InterpreterResult {
        self.visit(ast)
    }

    fn visit(&self, node: Node) -> InterpreterResult {
        // fixme: using copy to avoid borrow checker error, but this is probably not the best way to do this
        let span = *node.span();

        match node.into_kind() {
            NodeKind::Statement(nodes) => self.visit_statements(nodes),
            NodeKind::BinOp(left, op, right) => self.visit_bin_op(*left, op, *right, span),
            NodeKind::UnaryOp(op, right) => self.visit_unary_op(op, *right, span),
            NodeKind::Int(num) => self.make_val(ValueKind::Int(num), span),
            NodeKind::Float(num) => self.make_val(ValueKind::Float(FloatOrd(num)), span),
            NodeKind::Bool(boolean) => self.make_val(ValueKind::Bool(boolean), span),
            NodeKind::String(string) => self.make_val(ValueKind::String(string), span),
            NodeKind::Dict(dict) => self.visit_dict(dict, span),
            NodeKind::List(list) => self.visit_list(list, span),
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
            let node = self.visit(*node)?;
            debug!("{node:?}");
        }

        self.make_val(ValueKind::Void, Span(0, 0)) // todo: return real span (add span to statements)
    }

    fn visit_bin_op(&self, left: Node, op: Op, right: Node, span: Span) -> InterpreterResult {
        // fixme: using copy to avoid borrow checker error, but this is probably not the best way to do this
        let left_span = *left.span();
        let right_span = *right.span();

        let left_res = self.visit(left)?;
        let right_res = self.visit(right)?;

        match op {
            Op::Add => left_res.add(right_res),
            Op::Sub => left_res.sub(right_res),
            Op::Mul => left_res.mul(right_res),
            Op::Div => left_res.div(right_res),
            Op::Mod => left_res.rem(right_res),
            Op::Pow => left_res.pow(right_res),
            _ => Err(GlassError::UnknownError {
                message: "Unimplemented binary operation".into(),
            })?,
        }
        // fixme: extending the spans can cause the error span to sometimes be greater than the actual span of the
        // operation taking place because the spans get extended to the left and right of the operation
        .map_err(|err| interpreter_err_mapper(err, &self.source_ref, left_span.extend(right_span)))
    }

    fn visit_unary_op(&self, op: Op, right: Node, span: Span) -> InterpreterResult {
        todo!()
    }

    fn visit_dict(&self, dict: Vec<(Box<Node>, Box<Node>)>, span: Span) -> InterpreterResult {
        let mut map = BTreeMap::new();

        for (key, value) in dict {
            let key = self.visit(*key)?.into_value();
            let value = self.visit(*value)?.into_value();

            map.insert(key, value);
        }

        self.make_val(ValueKind::Dict(map), span)
    }

    fn visit_list(&self, list: Vec<Box<Node>>, span: Span) -> InterpreterResult {
        let mut vec = vec![];

        for node in list {
            vec.push(self.visit(*node)?.into_value());
        }

        self.make_val(ValueKind::List(vec), span)
    }
}
