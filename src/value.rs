use crate::ast::Op;
use crate::error::InterpreterError;
use crate::source_ref::SourceRef;
use crate::span::Span;
use float_ord::FloatOrd;
use std::collections::BTreeMap;
use std::fmt::Debug;

// fixme: no longer need source_ref and span because we bubble it up to the interpreter to handle
pub struct Value<'a> {
    source_ref: &'a SourceRef<'a>,
    span: Span,
    value: ValueKind,
}

impl Debug for Value<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value({:?})", self.value)
    }
}

type OperationResult<'a> = Result<Value<'a>, InterpreterError>;

impl<'a> Value<'a> {
    pub fn new(source_ref: &'a SourceRef, span: Span, value: ValueKind) -> Self {
        Self {
            source_ref,
            span,
            value,
        }
    }

    pub fn into_value(self) -> ValueKind {
        self.value
    }

    // todo: consider implicit conversion from int to float
    pub fn add(self, other: Self) -> OperationResult<'a> {
        match (self.value, other.value) {
            (ValueKind::Int(a), ValueKind::Int(b)) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Int(a + b),
            )),
            (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Float(FloatOrd(a + b)),
            )),
            (ValueKind::String(a), ValueKind::String(b)) => {
                let mut a = a;
                a.push_str(&b);

                Ok(Value::new(
                    &self.source_ref,
                    self.span,
                    ValueKind::String(a),
                ))
            }
            (ValueKind::List(a), ValueKind::List(b)) => {
                let mut a = a;
                a.extend(b);

                Ok(Value::new(&self.source_ref, self.span, ValueKind::List(a)))
            }
            (ValueKind::Dict(a), ValueKind::Dict(b)) => {
                let mut a = a;
                a.extend(b);

                Ok(Value::new(&self.source_ref, self.span, ValueKind::Dict(a)))
            }
            (a, b) => Err(InterpreterError::InvalidOperation { op: Op::Add, a, b }),
        }
    }

    pub fn sub(self, other: Self) -> OperationResult<'a> {
        match (self.value, other.value) {
            (ValueKind::Int(a), ValueKind::Int(b)) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Int(a - b),
            )),
            (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Float(FloatOrd(a - b)),
            )),
            (a, b) => Err(InterpreterError::InvalidOperation { op: Op::Sub, a, b }),
        }
    }

    pub fn mul(self, other: Self) -> OperationResult<'a> {
        match (self.value, other.value) {
            (ValueKind::Int(a), ValueKind::Int(b)) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Int(a * b),
            )),
            (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Float(FloatOrd(a * b)),
            )),
            (a, b) => Err(InterpreterError::InvalidOperation { op: Op::Mul, a, b }),
        }
    }

    pub fn div(self, other: Self) -> OperationResult<'a> {
        match (self.value, other.value) {
            (ValueKind::Int(a), ValueKind::Int(b)) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Int(a / b),
            )),
            (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Float(FloatOrd(a / b)),
            )),
            (a, b) => Err(InterpreterError::InvalidOperation { op: Op::Div, a, b }),
        }
    }

    pub fn rem(self, other: Self) -> OperationResult<'a> {
        match (self.value, other.value) {
            (ValueKind::Int(a), ValueKind::Int(b)) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Int(a % b),
            )),
            (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Float(FloatOrd(a % b)),
            )),
            (a, b) => Err(InterpreterError::InvalidOperation { op: Op::Mod, a, b }),
        }
    }

    pub fn pow(self, other: Self) -> OperationResult<'a> {
        match (self.value, other.value) {
            (ValueKind::Int(a), ValueKind::Int(b)) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Int(a.pow(b as u32)),
            )),
            (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Float(FloatOrd(a.powf(b))),
            )),
            (a, b) => Err(InterpreterError::InvalidOperation { op: Op::Pow, a, b }),
        }
    }

    // fixme: clean up this mess
    pub fn neg(self) -> OperationResult<'a> {
        match self.value {
            ValueKind::Int(a) => Ok(Value::new(&self.source_ref, self.span, ValueKind::Int(-a))),
            ValueKind::Float(FloatOrd(a)) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Float(FloatOrd(-a)),
            )),
            value => Err(InterpreterError::InvalidUnaryOperation { op: Op::Sub, value }),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ValueKind {
    Int(i64),
    Float(FloatOrd<f64>),
    Bool(bool),
    String(String),
    List(Vec<ValueKind>),
    Dict(BTreeMap<ValueKind, ValueKind>),
    Void,
}

impl ValueKind {
    pub fn get_type(&self) -> &str {
        match self {
            ValueKind::Int(_) => "int",
            ValueKind::Float(_) => "float",
            ValueKind::Bool(_) => "bool",
            ValueKind::String(_) => "string",
            ValueKind::List(_) => "list",
            ValueKind::Dict(_) => "dict",
            ValueKind::Void => "void",
        }
    }
}
