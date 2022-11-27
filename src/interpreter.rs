use crate::ast::{Node, NodeKind, Op};
use crate::error::{interpreter_err_mapper, GlassError};
use crate::scope::Scope;
use crate::source_ref::SourceRef;
use crate::span::Span;
use crate::value::Value;
use log::debug;
use std::collections::BTreeMap;

pub struct Interpreter<'a> {
    scope: Scope<'a>,
    source_ref: SourceRef<'a>,
}

pub type InterpreterResult = Result<Value, GlassError>;

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
            NodeKind::Int(num) => Ok(Value::Int(num)),
            NodeKind::Float(num) => Ok(Value::Float(num)),
            NodeKind::Bool(boolean) => Ok(Value::Bool(boolean)),
            NodeKind::String(string) => Ok(Value::String(string)),
            NodeKind::Dict(dict) => self.visit_dict(dict),
            NodeKind::List(list) => self.visit_list(list),
            NodeKind::FunctionDefinition(name, args, body) => self.visit_fn_def(name, args, *body),
            node => Err(GlassError::UnknownError {
                message: format!("Unknown node: {:?}", node),
            }),
        }
    }

    fn visit_statements(&self, nodes: Vec<Box<Node>>) -> InterpreterResult {
        for node in nodes {
            let node = self.visit(*node)?;
            debug!("{node:?}");
        }

        Ok(Value::Void)
    }

    fn visit_bin_op(&self, left: Node, op: Op, right: Node, span: Span) -> InterpreterResult {
        let left_res = self.visit(left)?;
        let right_res = self.visit(right)?;

        match op {
            Op::Add => left_res.add(right_res),
            Op::Sub => left_res.sub(right_res),
            Op::Mul => left_res.mul(right_res),
            Op::Div => left_res.div(right_res),
            Op::Mod => left_res.rem(right_res),
            Op::Pow => left_res.pow(right_res),
            Op::Equal => left_res.eq(right_res),
            Op::NotEqual => left_res.ne(right_res),
            Op::Less => left_res.lt(right_res),
            Op::LessEqual => left_res.lt(right_res),
            Op::Greater => left_res.gt(right_res),
            Op::GreaterEqual => left_res.gt(right_res),
            Op::And => left_res.and(right_res),
            Op::Or => left_res.or(right_res),
            Op::Not => Err(GlassError::UnknownError {
                message: format!("Operator '{}' requires one operand", op),
            })?,
        }
        // fixme: the spans can cause the error span to sometimes be greater than the actual span of the
        // operation taking place because the spans get extended to the left and right of the operation
        // as we traverse the tree. its not a huge deal but it would be nice to fix
        .map_err(|err| interpreter_err_mapper(err, &self.source_ref, span))
    }

    fn visit_unary_op(&self, op: Op, right: Node, span: Span) -> InterpreterResult {
        let NodeKind::UnaryOp(op, right) = right.into_kind() else {
            Err(GlassError::UnknownError {
                message: "Unimplemented unary operation".into(),
            })?
        };

        let right_res = self.visit(*right)?;

        match op {
            Op::Sub => right_res.neg(),
            Op::Not => right_res.not(),
            _ => Err(GlassError::UnknownError {
                message: "Invalid unary operator".into(),
            })?,
        }
        .map_err(|err| interpreter_err_mapper(err, &self.source_ref, span))
    }

    fn visit_dict(&self, dict: Vec<(Box<Node>, Box<Node>)>) -> InterpreterResult {
        let mut map = BTreeMap::new();

        for (key, value) in dict {
            map.insert(self.visit(*key)?, self.visit(*value)?);
        }

        Ok(Value::Dict(map))
    }

    fn visit_list(&self, list: Vec<Box<Node>>) -> InterpreterResult {
        let mut vec = vec![];

        for node in list {
            vec.push(self.visit(*node)?);
        }

        Ok(Value::List(vec))
    }

    fn visit_fn_def(&self, name: String, args: Vec<String>, body: Node) -> InterpreterResult {
        Ok(Value::Function(name, args, body))
    }
}
