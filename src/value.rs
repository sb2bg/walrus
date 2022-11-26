use crate::ast::Op;
use crate::error::GlassError;
use crate::interpreter::InterpreterResult;
use crate::source_ref::SourceRef;
use crate::span::Span;
use float_ord::FloatOrd;
use std::collections::BTreeMap;
use std::fmt::Debug;

#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord)] // todo: remove PartialEq, Eq, Hash, PartialOrd, Ord
pub struct Value<'a> {
    source_ref: &'a SourceRef<'a>,
    span: Span,
    value: ValueKind,
}

impl<'a> Value<'a> {
    pub fn new(source_ref: &'a SourceRef, span: Span, value: ValueKind) -> Self {
        Self {
            source_ref,
            span,
            value,
        }
    }

    pub fn value(&self) -> &ValueKind {
        &self.value
    }

    pub fn add(&self, other: Self) -> InterpreterResult {
        match (&self.value, &other.value) {
            (ValueKind::Int(left), ValueKind::Int(right)) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Int(left + right),
            )),
            (ValueKind::Float(left), ValueKind::Float(right)) => Ok(Value::new(
                &self.source_ref,
                self.span,
                ValueKind::Float(FloatOrd(left.0 + right.0)),
            )),
            (ValueKind::String(left), ValueKind::String(right)) => {
                let mut left = left.clone(); // fixme: clone
                left.push_str(right);

                Ok(Value::new(
                    &self.source_ref,
                    self.span,
                    ValueKind::String(left),
                ))
            }
            (ValueKind::List(left), ValueKind::List(right)) => {
                // left.extend(right);

                Ok(Value::new(
                    &self.source_ref,
                    self.span,
                    ValueKind::List(vec![]), // fixme: (return appended list)
                ))
            }
            (ValueKind::Dict(left), ValueKind::Dict(right)) => {
                // left.extend(right);

                Ok(Value::new(
                    &self.source_ref,
                    self.span,
                    ValueKind::Dict(BTreeMap::new()), // fixme (return appended dict)
                ))
            }
            (left, right) => Err(GlassError::InvalidOperation {
                operation: Op::Add,
                left: left.get_type().into(),
                right: right.get_type().into(),
                span: self.span.max(other.span),
                src: self.source_ref.source.into(), // fixme: avoid into
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
