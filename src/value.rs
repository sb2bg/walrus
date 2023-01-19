use crate::arenas::{DictKey, FuncKey, ListKey, RustFuncKey, RustFunction, StringKey};
use crate::ast::Node;
use crate::range::RangeValue;
use float_ord::FloatOrd;
use rustc_hash::FxHashMap;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;

pub enum HeapValue {
    List(Vec<ValueKind>),
    Dict(FxHashMap<ValueKind, ValueKind>),
    Function((String, Vec<String>, Node)),
    RustFunction(RustFunction),
    String(String),
}

#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash)]
pub enum ValueKind {
    // todo: consolidate ints and floats into single number type
    Int(i64),
    Float(FloatOrd<f64>),
    Bool(bool),
    Range(RangeValue),
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
            ValueKind::Range(_) => "range",
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
            ValueKind::Range(range) => write!(f, "{}", range),
            ValueKind::String(s) => write!(f, "{:?}", s),
            ValueKind::List(l) => write!(f, "{:?}", l),
            ValueKind::Dict(d) => write!(f, "{:?}", d),
            ValueKind::Function(fk) => write!(f, "{:?}", fk),
            ValueKind::RustFunction(fk) => write!(f, "{:?}", fk),
            ValueKind::Void => write!(f, "void"),
        }
    }
}
