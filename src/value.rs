use crate::arenas::{DictKey, FuncKey, ListKey, RustFuncKey, StringKey};
use crate::scope::Scope;
use float_ord::FloatOrd;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;

#[derive(Debug)]
pub struct Value<'a> {
    kind: ValueKind,
    scope: &'a Scope<'a>,
}

impl<'a> Value<'a> {
    pub fn new(kind: ValueKind, scope: &'a Scope<'a>) -> Self {
        Self { kind, scope }
    }

    pub fn kind(&self) -> &ValueKind {
        &self.kind
    }

    pub fn into_kind(self) -> ValueKind {
        self.kind
    }
}

const EXP_ERR: &str = "Attempted to access freed memory";

impl Display for Value<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.kind {
            ValueKind::Int(value) => write!(f, "{}", value),
            ValueKind::Float(FloatOrd(value)) => write!(f, "{}", value),
            ValueKind::Bool(value) => write!(f, "{}", value),
            ValueKind::String(key) => {
                write!(f, "{}", self.scope.arena().get_string(key).expect(EXP_ERR))
            }
            ValueKind::List(key) => {
                write!(f, "{:?}", self.scope.arena().get_list(key).expect(EXP_ERR))
            }
            ValueKind::Dict(key) => {
                write!(f, "{:?}", self.scope.arena().get_dict(key).expect(EXP_ERR))
            }
            ValueKind::Function(key) => {
                write!(
                    f,
                    "{:?}",
                    self.scope.arena().get_function(key).expect(EXP_ERR)
                )
            } // fixme: prints incorrectly
            ValueKind::RustFunction(_) => write!(f, "<rust function>"),
            ValueKind::Void => write!(f, "None"),
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Copy, Clone, Hash)]
pub enum ValueKind {
    Int(i64),
    Float(FloatOrd<f64>),
    Bool(bool),
    String(StringKey),
    List(ListKey),
    Dict(DictKey),
    Function(FuncKey),
    RustFunction(RustFuncKey),
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
            ValueKind::Function(..) => "function",
            ValueKind::RustFunction(..) => "builtin_function",
            ValueKind::Void => "void",
        }
    }
}

// fixme: maybe replace with a to_string method, not impl
impl Display for ValueKind {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            ValueKind::Int(i) => write!(f, "{}", i),
            ValueKind::Float(fo) => write!(f, "{}", fo.0),
            ValueKind::Bool(b) => write!(f, "{}", b),
            ValueKind::String(s) => write!(f, "{:?}", s),
            ValueKind::List(l) => write!(f, "{:?}", l),
            ValueKind::Dict(d) => write!(f, "{:?}", d),
            ValueKind::Function(fk) => write!(f, "{:?}", fk),
            ValueKind::RustFunction(fk) => write!(f, "{:?}", fk),
            ValueKind::Void => write!(f, "void"),
        }
    }
}
