use crate::arenas::ValueHolder;
use crate::ast::{Node, NodeKind, Op};
use crate::error::WalrusError;
use crate::error::WalrusError::UndefinedVariable;
use crate::scope::Scope;
use crate::source_ref::SourceRef;
use crate::span::Span;
use crate::value::Value;
use float_ord::FloatOrd;
use std::collections::BTreeMap;
use uuid::Uuid;

pub struct Interpreter<'a> {
    scope: Scope<'a>,
    source_ref: SourceRef<'a>,
    returnable: bool,
    arena: ValueHolder,
}

pub type InterpreterResult<'a> = Result<Value, WalrusError>;

impl<'a> Interpreter<'a> {
    pub fn new(source_ref: SourceRef<'a>, returnable: bool) -> Self {
        Self {
            scope: Scope::new(),
            source_ref,
            returnable,
            arena: ValueHolder::new(),
        }
    }

    pub fn source_ref(&self) -> SourceRef<'a> {
        self.source_ref
    }

    pub fn interpret(&mut self, node: Node) -> InterpreterResult {
        // fixme: using copy to avoid borrow checker error, but this is probably not the best way to do this
        let span = *node.span();

        match node.into_kind() {
            NodeKind::Statement(nodes) => self.visit_statements(nodes),
            NodeKind::BinOp(left, op, right) => self.visit_bin_op(*left, op, *right, span),
            NodeKind::UnaryOp(op, value) => self.visit_unary_op(op, *value, span),
            NodeKind::Int(num) => Ok(Value::Int(num)),
            NodeKind::Float(num) => Ok(Value::Float(num)),
            NodeKind::Bool(boolean) => Ok(Value::Bool(boolean)),
            NodeKind::String(string) => Ok(self.arena.insert_string(string)),
            NodeKind::Dict(dict) => self.visit_dict(dict),
            NodeKind::List(list) => self.visit_list(list),
            NodeKind::FunctionDefinition(name, args, body) => self.visit_fn_def(name, args, *body),
            NodeKind::AnonFunctionDefinition(args, body) => self.visit_anon_fn_def(args, *body),
            NodeKind::Ident(ident) => self.visit_var(ident, span),
            NodeKind::Void => Ok(Value::Void),
            node => Err(WalrusError::UnknownError {
                message: format!("Unknown node: {:?}", node),
            }),
        }
    }

    fn visit_statements(&mut self, nodes: Vec<Box<Node>>) -> InterpreterResult {
        for node in nodes {
            match *node {
                Node {
                    kind: NodeKind::Return(ret),
                    ..
                } => {
                    // if !self.returnable {
                    //     return Err(WalrusError::ReturnOutsideFunction {
                    //         span: node.span(),
                    //         src: self.source_ref.source().into(),
                    //         filename: self.source_ref.filename().into(),
                    //     });
                    // }

                    return self.interpret(*ret);
                }
                _ => self.interpret(*node)?,
            };
        }

        Ok(Value::Void)
    }

    fn visit_bin_op(&mut self, left: Node, op: Op, right: Node, span: Span) -> InterpreterResult {
        let left_val = self.interpret(left)?;
        let right_val = self.interpret(right)?;

        match op {
            Op::Add => self.add(left_val, right_val, span),
            Op::Sub => self.sub(left_val, right_val, span),
            Op::Mul => self.mul(left_val, right_val, span),
            Op::Div => self.div(left_val, right_val, span),
            Op::Mod => self.rem(left_val, right_val, span),
            Op::Pow => self.pow(left_val, right_val, span),
            Op::Equal => self.equal(left_val, right_val),
            Op::NotEqual => self.not_equal(left_val, right_val),
            Op::Less => self.less(left_val, right_val, span),
            Op::LessEqual => self.less_equal(left_val, right_val, span),
            Op::Greater => self.greater(left_val, right_val, span),
            Op::GreaterEqual => self.greater_equal(left_val, right_val, span),
            Op::And => self.and(left_val, right_val, span),
            Op::Or => self.or(left_val, right_val, span),
            Op::Not => Err(WalrusError::UnknownError {
                message: format!("Operator '{}' requires one operand", op),
            })?,
        }
        // fixme: the spans can cause the error span to sometimes be greater than the actual span of the
        // operation taking place because the spans get extended to the left and right of the operation
        // as we traverse the tree. its not a huge deal but it would be nice to fix
    }

    fn visit_unary_op(&mut self, op: Op, value: Node, span: Span) -> InterpreterResult {
        let right_res = self.interpret(value)?;

        match op {
            Op::Sub => self.neg(right_res, span),
            Op::Not => self.not(right_res, span),
            _ => Err(WalrusError::UnknownError {
                message: "Invalid unary operator".into(),
            })?,
        }
    }

    fn visit_dict(&mut self, dict: Vec<(Box<Node>, Box<Node>)>) -> InterpreterResult {
        let mut map = BTreeMap::new();

        for (key, value) in dict {
            map.insert(self.interpret(*key)?, self.interpret(*value)?);
        }

        Ok(self.arena.insert_dict(map))
    }

    fn visit_list(&mut self, list: Vec<Box<Node>>) -> InterpreterResult {
        let mut vec = vec![];

        for node in list {
            vec.push(self.interpret(*node)?);
        }

        Ok(self.arena.insert_list(vec))
    }

    fn visit_anon_fn_def(&mut self, args: Vec<String>, body: Node) -> InterpreterResult {
        let fn_name = format!("anon_{}", Uuid::new_v4());
        Ok(self.arena.insert_function(fn_name, args, body))
    }

    fn visit_fn_def(&mut self, name: String, args: Vec<String>, body: Node) -> InterpreterResult {
        let value = self.arena.insert_function(name.clone(), args, body);
        self.scope.define(name, value);

        Ok(Value::Void)
    }

    fn visit_var(&self, name: String, span: Span) -> InterpreterResult {
        self.scope.get(&name).ok_or_else(|| UndefinedVariable {
            name,
            span,
            src: self.source_ref.source().into(),
            filename: self.source_ref.filename().into(),
        })
    }

    fn add(&mut self, left: Value, right: Value, span: Span) -> InterpreterResult<'a> {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                Ok(Value::Float(FloatOrd(a + b)))
            }
            (Value::String(a), Value::String(b)) => {
                let a_str = self.arena.get_string(a)?;
                let b_str = self.arena.get_string(b)?;
                Ok(self.arena.insert_string(a_str.clone() + b_str.as_str()))
            }
            (a, b) => Err(self.construct_err(Op::Add, a, b, span)),
        }
    }

    fn sub(&self, left: Value, right: Value, span: Span) -> InterpreterResult<'a> {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                Ok(Value::Float(FloatOrd(a - b)))
            }
            (a, b) => Err(self.construct_err(Op::Sub, a, b, span)),
        }
    }

    fn mul(&self, left: Value, right: Value, span: Span) -> InterpreterResult<'a> {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                Ok(Value::Float(FloatOrd(a * b)))
            }
            (a, b) => Err(self.construct_err(Op::Mul, a, b, span)),
        }
    }

    fn div(&self, left: Value, right: Value, span: Span) -> InterpreterResult<'a> {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a / b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                Ok(Value::Float(FloatOrd(a / b)))
            }
            (a, b) => Err(self.construct_err(Op::Div, a, b, span)),
        }
    }

    fn rem(&self, left: Value, right: Value, span: Span) -> InterpreterResult<'a> {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a % b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                Ok(Value::Float(FloatOrd(a % b)))
            }
            (a, b) => Err(self.construct_err(Op::Mod, a, b, span)),
        }
    }

    fn pow(&self, left: Value, right: Value, span: Span) -> InterpreterResult<'a> {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.pow(b as u32))),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                Ok(Value::Float(FloatOrd(a.powf(b))))
            }
            (a, b) => Err(self.construct_err(Op::Pow, a, b, span)),
        }
    }

    fn neg(&self, value: Value, span: Span) -> InterpreterResult<'a> {
        match value {
            Value::Int(a) => Ok(Value::Int(-a)),
            Value::Float(FloatOrd(a)) => Ok(Value::Float(FloatOrd(-a))),
            value => Err(WalrusError::InvalidUnaryOperation {
                op: Op::Sub,
                operand: value.get_type().to_string(),
                span,
                src: self.source_ref.filename().to_string(),
                filename: self.source_ref.filename().to_string(),
            }),
        }
    }

    fn equal(&self, left: Value, right: Value) -> InterpreterResult<'a> {
        Ok(Value::Bool(left == right))
    }

    fn not_equal(&self, left: Value, right: Value) -> InterpreterResult<'a> {
        Ok(Value::Bool(left != right))
    }

    fn less(&self, left: Value, right: Value, span: Span) -> InterpreterResult<'a> {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a < b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => Ok(Value::Bool(a < b)),
            (a, b) => Err(self.construct_err(Op::Less, a, b, span)),
        }
    }

    fn less_equal(&self, left: Value, right: Value, span: Span) -> InterpreterResult<'a> {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a <= b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => Ok(Value::Bool(a <= b)),
            (a, b) => Err(self.construct_err(Op::LessEqual, a, b, span)),
        }
    }

    fn greater(&self, left: Value, right: Value, span: Span) -> InterpreterResult<'a> {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a > b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => Ok(Value::Bool(a > b)),
            (a, b) => Err(self.construct_err(Op::Greater, a, b, span)),
        }
    }

    fn greater_equal(&self, left: Value, right: Value, span: Span) -> InterpreterResult<'a> {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a >= b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => Ok(Value::Bool(a >= b)),
            (a, b) => Err(self.construct_err(Op::GreaterEqual, a, b, span)),
        }
    }

    pub fn and(&self, left: Value, right: Value, span: Span) -> InterpreterResult<'a> {
        match (left, right) {
            (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a && b)),
            (a, b) => Err(self.construct_err(Op::And, a, b, span)),
        }
    }

    pub fn or(&self, left: Value, right: Value, span: Span) -> InterpreterResult<'a> {
        match (left, right) {
            (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a || b)),
            (a, b) => Err(self.construct_err(Op::Or, a, b, span)),
        }
    }

    pub fn not(&self, value: Value, span: Span) -> InterpreterResult<'a> {
        match value {
            Value::Bool(a) => Ok(Value::Bool(!a)),
            value => Err(WalrusError::InvalidUnaryOperation {
                op: Op::Not,
                operand: value.get_type().to_string(),
                span,
                src: self.source_ref.filename().to_string(),
                filename: self.source_ref.filename().to_string(),
            }),
        }
    }

    fn construct_err(&self, op: Op, left: Value, right: Value, span: Span) -> WalrusError {
        WalrusError::InvalidOperation {
            op,
            left: left.get_type().to_string(),
            right: right.get_type().to_string(),
            span,
            src: self.source_ref.source().to_string(),
            filename: self.source_ref.filename().to_string(),
        }
    }
}
