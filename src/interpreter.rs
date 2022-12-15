use crate::arenas::ValueHolder;
use crate::ast::{Node, NodeKind, Op};
use crate::error::WalrusError;
use crate::error::WalrusError::UndefinedVariable;
use crate::scope::Scope;
use crate::source_ref::SourceRef;
use crate::span::{Span, Spanned};
use crate::value::Value;
use float_ord::FloatOrd;
use log::debug;
use std::collections::HashMap;
use uuid::Uuid;

pub struct Interpreter<'a> {
    scope: Scope<'a>,
    source_ref: SourceRef<'a>,
    returnable: bool,
    arena: ValueHolder,
}

pub type InterpreterResult = Result<Value, WalrusError>;

// consider moving interpreter into scope instead of the other way around
impl<'a> Interpreter<'a> {
    pub fn new(source_ref: SourceRef<'a>) -> Self {
        let mut arena = ValueHolder::new();

        Self {
            scope: Scope::new(&mut arena),
            source_ref,
            returnable: false,
            arena,
        }
    }

    pub fn dump(&self) {
        debug!("Interpreter dump");
        debug!("Returnable: {}", self.returnable);
        self.scope.dump();
        self.arena.dump();
    }

    pub fn source_ref(&self) -> SourceRef<'a> {
        self.source_ref
    }

    pub fn set_source_ref(&mut self, source_ref: SourceRef<'a>) {
        self.source_ref = source_ref;
    }

    pub fn create_child(&'a self, name: String) -> Interpreter {
        Self {
            scope: self.scope.new_child(name),
            source_ref: self.source_ref,
            returnable: true,
            arena: self.arena.clone(), // todo: is this correct?
        }
    }

    pub fn interpret(&mut self, node: Node) -> InterpreterResult {
        // fixme: using copy to avoid borrow checker error, but this is probably not the best way to do this
        let span = *node.span();

        let res = match node.into_kind() {
            NodeKind::Statements(nodes) => self.visit_statements(nodes),
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
            NodeKind::If(condition, then, otherwise) => self.visit_if(*condition, *then, otherwise),
            NodeKind::While(condition, body) => self.visit_while(*condition, *body),
            NodeKind::Assign(name, value) => self.visit_assign(name, *value),
            NodeKind::Reassign(ident, value, op) => self.visit_reassign(ident, *value, op),
            NodeKind::Print(value) => self.visit_print(*value),
            NodeKind::Throw(value) => self.visit_throw(*value, span),
            node => Err(WalrusError::UnknownError {
                message: format!("Unknown node: {:?}", node),
            }),
        };

        debug!("{:?}", res);
        res
    }

    fn visit_statements(&mut self, nodes: Vec<Box<Node>>) -> InterpreterResult {
        for node in nodes {
            let span = *node.span();

            match *node {
                Node {
                    kind: NodeKind::Return(ret),
                    ..
                } => {
                    if !self.returnable {
                        return Err(WalrusError::ReturnOutsideFunction {
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        });
                    }

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
        let mut map = HashMap::new();

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

    fn visit_if(
        &mut self,
        condition: Node,
        body: Node,
        otherwise: Option<Box<Node>>,
    ) -> InterpreterResult {
        let cond_span = *condition.span();
        let condition = self.interpret(condition)?;

        if self.is_truthy(condition, cond_span)? {
            self.interpret(body)?;
        } else if let Some(otherwise) = otherwise {
            self.interpret(*otherwise)?;
        }

        Ok(Value::Void)
    }

    fn visit_while(&mut self, condition: Node, body: Node) -> InterpreterResult {
        // this is more complicated because we need to be able to repeatedly
        // evaluate the condition and body, which involves a lot of cloning
        // that we should avoid if possible. we could store our nodes in an
        // arena and copy the arena index instead of the node itself, but
        // that would require a lot of refactoring and would be a lot of work
        // for a small optimization, so for now, we'll just clone and revisit
        // this later
        let cond_span = *condition.span();

        loop {
            let condition = self.interpret(condition.clone())?;

            if self.is_truthy(condition, cond_span)? {
                self.interpret(body.clone())?;
            } else {
                break;
            }
        }

        Ok(Value::Void)
    }

    fn visit_assign(&mut self, name: String, value: Node) -> InterpreterResult {
        let value = self.interpret(value)?;
        self.scope.define(name, value);

        Ok(Value::Void)
    }

    fn visit_reassign(&mut self, ident: Spanned<String>, value: Node, op: Op) -> InterpreterResult {
        let value = self.interpret(value)?;

        // fixme: clone
        // fixme: operator such as +=, -=, etc should be handled here
        if !self.scope.reassign(ident.value().clone(), value) {
            return Err(UndefinedVariable {
                name: ident.value().clone(), // fixme: clone
                span: ident.span(),
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            });
        }

        Ok(Value::Void)
    }

    fn visit_throw(&mut self, value: Node, span: Span) -> InterpreterResult {
        let value = self.interpret(value)?;

        Err(WalrusError::Exception {
            message: value.to_string(),
            span,
            src: self.source_ref.source().into(),
            filename: self.source_ref.filename().into(),
        })
    }

    fn visit_print(&mut self, value: Node) -> InterpreterResult {
        let value = self.interpret(value)?;
        println!("{}", value);

        Ok(Value::Void)
    }

    fn add(&mut self, left: Value, right: Value, span: Span) -> InterpreterResult {
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

    fn sub(&self, left: Value, right: Value, span: Span) -> InterpreterResult {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                Ok(Value::Float(FloatOrd(a - b)))
            }
            (a, b) => Err(self.construct_err(Op::Sub, a, b, span)),
        }
    }

    fn mul(&self, left: Value, right: Value, span: Span) -> InterpreterResult {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                Ok(Value::Float(FloatOrd(a * b)))
            }
            (a, b) => Err(self.construct_err(Op::Mul, a, b, span)),
        }
    }

    fn div(&self, left: Value, right: Value, span: Span) -> InterpreterResult {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a / b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                Ok(Value::Float(FloatOrd(a / b)))
            }
            (a, b) => Err(self.construct_err(Op::Div, a, b, span)),
        }
    }

    fn rem(&self, left: Value, right: Value, span: Span) -> InterpreterResult {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a % b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                Ok(Value::Float(FloatOrd(a % b)))
            }
            (a, b) => Err(self.construct_err(Op::Mod, a, b, span)),
        }
    }

    fn pow(&self, left: Value, right: Value, span: Span) -> InterpreterResult {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.pow(b as u32))),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                Ok(Value::Float(FloatOrd(a.powf(b))))
            }
            (a, b) => Err(self.construct_err(Op::Pow, a, b, span)),
        }
    }

    fn neg(&self, value: Value, span: Span) -> InterpreterResult {
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

    fn equal(&self, left: Value, right: Value) -> InterpreterResult {
        Ok(Value::Bool(left == right))
    }

    fn not_equal(&self, left: Value, right: Value) -> InterpreterResult {
        Ok(Value::Bool(left != right))
    }

    fn less(&self, left: Value, right: Value, span: Span) -> InterpreterResult {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a < b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => Ok(Value::Bool(a < b)),
            (a, b) => Err(self.construct_err(Op::Less, a, b, span)),
        }
    }

    fn less_equal(&self, left: Value, right: Value, span: Span) -> InterpreterResult {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a <= b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => Ok(Value::Bool(a <= b)),
            (a, b) => Err(self.construct_err(Op::LessEqual, a, b, span)),
        }
    }

    fn greater(&self, left: Value, right: Value, span: Span) -> InterpreterResult {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a > b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => Ok(Value::Bool(a > b)),
            (a, b) => Err(self.construct_err(Op::Greater, a, b, span)),
        }
    }

    fn greater_equal(&self, left: Value, right: Value, span: Span) -> InterpreterResult {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a >= b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => Ok(Value::Bool(a >= b)),
            (a, b) => Err(self.construct_err(Op::GreaterEqual, a, b, span)),
        }
    }

    fn and(&self, left: Value, right: Value, span: Span) -> InterpreterResult {
        match (left, right) {
            (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a && b)),
            (a, b) => Err(self.construct_err(Op::And, a, b, span)),
        }
    }

    fn or(&self, left: Value, right: Value, span: Span) -> InterpreterResult {
        match (left, right) {
            (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a || b)),
            (a, b) => Err(self.construct_err(Op::Or, a, b, span)),
        }
    }

    fn not(&self, value: Value, span: Span) -> InterpreterResult {
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

    fn is_truthy(&self, value: Value, span: Span) -> Result<bool, WalrusError> {
        match value {
            Value::Bool(b) => Ok(b),
            value => Err(WalrusError::TypeMismatch {
                expected: "bool".to_string(),
                found: value.get_type().to_string(),
                span,
                src: self.source_ref.source().to_string(),
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
