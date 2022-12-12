use crate::arenas::{DictKey, FuncKey, ListKey, StringKey};
use float_ord::FloatOrd;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Copy, Clone)]
pub enum Value {
    Int(i64),
    Float(FloatOrd<f64>),
    Bool(bool),
    String(StringKey),
    List(ListKey),
    Dict(DictKey),
    Function(FuncKey),
    Void,
}

impl Value {
    pub fn get_type(&self) -> &str {
        match self {
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Bool(_) => "bool",
            Value::String(_) => "string",
            Value::List(_) => "list",
            Value::Dict(_) => "dict",
            Value::Function(..) => "function",
            Value::Void => "void",
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Value::Int(i) => write!(f, "{}", i),
            Value::Float(fo) => write!(f, "{}", fo.0),
            Value::Bool(b) => write!(f, "{}", b),
            Value::String(s) => write!(f, "{:?}", s),
            Value::List(l) => write!(f, "{:?}", l),
            Value::Dict(d) => write!(f, "{:?}", d),
            Value::Function(fk) => write!(f, "{:?}", fk),
            Value::Void => write!(f, "void"),
        }
    }
}

// fn vec_to_string<T: Display>(vec: &Vec<T>) -> String {
//     join(vec.iter(), ", ")
// }
//
// fn map_to_string(map: &BTreeMap<Value, Value>) -> String {
//     join(map.iter().map(|(k, v)| format!("{}: {}", k, v)), ", ")
// }
