use crate::arenas::{DictKey, FuncKey, ListKey, RustFuncKey};
use crate::ast::Node;
use crate::range::RangeValue;
use crate::rust_function::RustFunction;
use crate::scope::Scope;
use crate::WalrusResult;
use float_ord::FloatOrd;
use rustc_hash::FxHashMap;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use string_interner::DefaultSymbol;

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
    String(DefaultSymbol),
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

    pub fn stringify(self) -> WalrusResult<String> {
        Ok(match self {
            ValueKind::Int(i) => i.to_string(),
            ValueKind::Float(FloatOrd(f)) => f.to_string(),
            ValueKind::Bool(b) => b.to_string(),
            ValueKind::Range(r) => r.to_string(),
            ValueKind::String(s) => Scope::get_string(s)?.to_string(),
            ValueKind::List(l) => {
                let list = Scope::get_list(l)?;
                let mut s = String::new();

                s.push('[');

                for (i, item) in list.iter().enumerate() {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    s.push_str(&item.stringify()?);
                }

                s.push(']');
                s
            }
            ValueKind::Dict(d) => {
                let dict = Scope::get_dict(d)?;
                let mut s = String::new();

                s.push('{');

                for (i, (key, value)) in dict.iter().enumerate() {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    s.push_str(&key.stringify()?);
                    s.push_str(": ");
                    s.push_str(&value.stringify()?);
                }

                s.push('}');
                s
            }
            ValueKind::Function(f) => {
                let (name, args, _) = Scope::get_function(f)?;
                let mut s = String::new();

                s.push_str("function ");
                s.push_str(&name);
                s.push('(');

                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    s.push_str(arg);
                }

                s.push(')');
                s
            }
            ValueKind::RustFunction(r) => {
                let rust_func = Scope::get_rust_function(r)?;
                let mut s = String::new();

                s.push_str("rust_function ");
                s.push_str(rust_func.name());
                s
            }
            ValueKind::Void => "void".to_string(),
        })
    }
}

// fixme: maybe replace with a to_string method, not impl
impl Display for ValueKind {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            ValueKind::Int(int) => write!(f, "{}", int),
            ValueKind::Float(FloatOrd(float)) => write!(f, "{}", float),
            ValueKind::Bool(bool) => write!(f, "{}", bool),
            ValueKind::Range(range) => write!(f, "{}", range),
            ValueKind::String(str) => write!(f, "{:?}", str),
            ValueKind::List(list) => write!(f, "{:?}", list),
            ValueKind::Dict(dict) => write!(f, "{:?}", dict),
            ValueKind::Function(func) => write!(f, "{:?}", func),
            ValueKind::RustFunction(rust_func) => write!(f, "{:?}", rust_func),
            ValueKind::Void => write!(f, "void"),
        }
    }
}
