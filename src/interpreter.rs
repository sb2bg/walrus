use crate::arenas::ValueHolder;
use crate::ast::{Node, NodeKind, Op};
use crate::error::WalrusError::UndefinedVariable;
use crate::error::{interpreter_err_mapper, WalrusError};
use crate::scope::Scope;
use crate::source_ref::SourceRef;
use crate::span::Span;
use crate::value::Value;
use crate::value::Value::Function;
use float_ord::FloatOrd;
use std::collections::BTreeMap;
use uuid::Uuid;

pub struct Interpreter<'a> {
    scope: Scope<'a>,
    source_ref: SourceRef<'a>,
    returnable: bool,
    arenas: ValueHolder,
}

pub type InterpreterResult<'a> = Result<Value, WalrusError>;

impl<'a> Interpreter<'a> {
    pub fn new(source_ref: SourceRef<'a>, returnable: bool) -> Self {
        Self {
            scope: Scope::new(),
            source_ref,
            returnable,
            arenas: ValueHolder::new(),
        }
    }

    pub fn source_ref(&self) -> &SourceRef<'a> {
        &self.source_ref
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
            NodeKind::String(string) => {
                let key = self.arenas.insert_string(string);
                Ok(Value::String(key))
            }
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
        let left_res = self.interpret(left)?;
        let right_res = self.interpret(right)?;

        match op {
            Op::Add => self.add(left_res, right_res, span),
            // Op::Sub => left_res.sub(right_res),
            // Op::Mul => left_res.mul(right_res),
            // Op::Div => left_res.div(right_res),
            // Op::Mod => left_res.rem(right_res),
            // Op::Pow => left_res.pow(right_res),
            // Op::Equal => left_res.eq(right_res),
            // Op::NotEqual => left_res.ne(right_res),
            // Op::Less => left_res.lt(right_res),
            // Op::LessEqual => left_res.lt(right_res),
            // Op::Greater => left_res.gt(right_res),
            // Op::GreaterEqual => left_res.gt(right_res),
            // Op::And => left_res.and(right_res),
            // Op::Or => left_res.or(right_res),
            Op::Not => Err(WalrusError::UnknownError {
                message: format!("Operator '{}' requires one operand", op),
            })?,
            _ => {
                return Err(WalrusError::UnknownError {
                    message: format!("Unknown operator: {}", op),
                })
            }
        }
        // fixme: the spans can cause the error span to sometimes be greater than the actual span of the
        // operation taking place because the spans get extended to the left and right of the operation
        // as we traverse the tree. its not a huge deal but it would be nice to fix
    }

    fn visit_unary_op(&mut self, op: Op, value: Node, span: Span) -> InterpreterResult {
        let right_res = self.interpret(value)?;

        match op {
            Op::Sub => right_res.neg(),
            Op::Not => right_res.not(),
            _ => Err(WalrusError::UnknownError {
                message: "Invalid unary operator".into(),
            })?,
        }
        .map_err(|err| interpreter_err_mapper(err, &self.source_ref, span))
    }

    fn visit_dict(&mut self, dict: Vec<(Box<Node>, Box<Node>)>) -> InterpreterResult {
        let mut map = BTreeMap::new();

        for (key, value) in dict {
            map.insert(self.interpret(*key)?, self.interpret(*value)?);
        }

        let key = self.arenas.insert_dict(map);
        Ok(Value::Dict(key))
    }

    fn visit_list(&mut self, list: Vec<Box<Node>>) -> InterpreterResult {
        let mut vec = vec![];

        for node in list {
            vec.push(self.interpret(*node)?);
        }

        let key = self.arenas.insert_list(vec);
        Ok(Value::List(key))
    }

    fn visit_anon_fn_def(&mut self, args: Vec<String>, body: Node) -> InterpreterResult {
        let fn_name = format!("anon_{}", Uuid::new_v4());
        let key = self.arenas.insert_function(fn_name, args, body);
        Ok(Value::Function(key))
    }

    fn visit_fn_def(&mut self, name: String, args: Vec<String>, body: Node) -> InterpreterResult {
        let key = self.arenas.insert_function(name.clone(), args, body);
        self.scope.define(name, Value::Function(key));

        Ok(Value::Void)
    }

    fn visit_var(&mut self, name: String, span: Span) -> InterpreterResult {
        self.scope.get(&name).ok_or_else(|| UndefinedVariable {
            name,
            span,
            src: self.source_ref.source().into(),
            filename: self.source_ref.filename().into(),
        })
    }

    fn add(&mut self, a: Value, b: Value, span: Span) -> InterpreterResult {
        match (a, b) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                Ok(Value::Float(FloatOrd(a + b)))
            }
            (Value::String(a), Value::String(b)) => {
                let a_str = self.arenas.get_string(a)?;
                let b_str = self.arenas.get_string(b)?;
                let res = a_str.clone() + b_str.as_str();
                let key = self.arenas.insert_string(res);

                Ok(Value::String(key))
            }
            (a, b) => Err(WalrusError::InvalidOperation {
                op: Op::Add,
                left: a.get_type().into(),
                right: b.get_type().into(),
                span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            })?,
        }
    }
}
