use crate::ast::Op;
use crate::error::GlassError;
use crate::interpreter::InterpreterResult;
use crate::source_ref::SourceRef;
use crate::span::Span;
use float_ord::FloatOrd;
use std::collections::BTreeMap;
use std::fmt::Debug;

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

    fn create_error(
        op: Op,
        a: ValueKind,
        b: ValueKind,
        span: Span,
        source_ref: &SourceRef,
    ) -> InterpreterResult<'a> {
        Err(GlassError::InvalidOperation {
            op,
            left: a.get_type().into(),
            right: b.get_type().into(),
            span,
            src: source_ref.source.into(),
            filename: source_ref.filename.into(),
        })
    }

    // todo: consider implicit conversion from int to float
    pub fn add(self, other: Self) -> InterpreterResult<'a> {
        match (self.value, other.value) {
            (ValueKind::Int(a), ValueKind::Int(b)) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Int(a + b),
            )),
            (ValueKind::Float(a), ValueKind::Float(b)) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Float(FloatOrd(a.0 + b.0)),
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
            (a, b) => Self::create_error(Op::Add, a, b, self.span, self.source_ref),
        }
    }

    pub fn sub(self, other: Self) -> InterpreterResult<'a> {
        match (self.value, other.value) {
            (ValueKind::Int(a), ValueKind::Int(b)) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Int(a - b),
            )),
            (ValueKind::Float(a), ValueKind::Float(b)) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Float(FloatOrd(a.0 - b.0)),
            )),
            (a, b) => Self::create_error(Op::Sub, a, b, self.span, self.source_ref),
        }
    }

    pub fn mul(self, other: Self) -> InterpreterResult<'a> {
        match (self.value, other.value) {
            (ValueKind::Int(a), ValueKind::Int(b)) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Int(a * b),
            )),
            (ValueKind::Float(a), ValueKind::Float(b)) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Float(FloatOrd(a.0 * b.0)),
            )),
            (a, b) => Self::create_error(Op::Mul, a, b, self.span, self.source_ref),
        }
    }

    pub fn div(self, other: Self) -> InterpreterResult<'a> {
        match (self.value, other.value) {
            (ValueKind::Int(a), ValueKind::Int(b)) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Int(a / b),
            )),
            (ValueKind::Float(a), ValueKind::Float(b)) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Float(FloatOrd(a.0 / b.0)),
            )),
            (a, b) => Self::create_error(Op::Div, a, b, self.span, self.source_ref),
        }
    }

    pub fn rem(self, other: Self) -> InterpreterResult<'a> {
        match (self.value, other.value) {
            (ValueKind::Int(a), ValueKind::Int(b)) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Int(a % b),
            )),
            (ValueKind::Float(a), ValueKind::Float(b)) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Float(FloatOrd(a.0 % b.0)),
            )),
            (a, b) => Self::create_error(Op::Mod, a, b, self.span, self.source_ref),
        }
    }

    pub fn pow(self, other: Self) -> InterpreterResult<'a> {
        match (self.value, other.value) {
            (ValueKind::Int(a), ValueKind::Int(b)) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Int(a.pow(b as u32)),
            )),
            (ValueKind::Float(a), ValueKind::Float(b)) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Float(FloatOrd(a.0.powf(b.0))),
            )),
            (a, b) => Self::create_error(Op::Pow, a, b, self.span, self.source_ref),
        }
    }

    pub fn neg(self) -> InterpreterResult<'a> {
        match self.value {
            ValueKind::Int(a) => Ok(Value::new(&self.source_ref, self.span, ValueKind::Int(-a))),
            ValueKind::Float(a) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Float(FloatOrd(-a.0)),
            )),
            a => Err(GlassError::InvalidUnary {
                op: Op::Sub,
                operand: a.get_type().into(),
                span: self.span,
                src: self.source_ref.source.into(),
                filename: self.source_ref.filename.into(),
            }),
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
    fn get_type(&self) -> &str {
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
