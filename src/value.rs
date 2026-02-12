use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;

use float_ord::FloatOrd;

use crate::WalrusResult;
use crate::arenas::{
    DictKey, FuncKey, IterKey, ListKey, Resolve, StringKey, StructDefKey, StructInstKey, TupleKey, ValueHolder,
};
use crate::iter::{CollectionIter, DictIter, RangeIter, StrIter, ValueIterator};
use crate::range::RangeValue;

#[derive(Debug, Clone)]
pub enum ValueIter {
    Range(RangeIter),
    Collection(CollectionIter),
    Dict(DictIter),
    Str(StrIter),
}

impl ValueIter {
    pub fn referenced_value(&self) -> Option<Value> {
        match self {
            ValueIter::Range(_) => None,
            ValueIter::Collection(iter) => Some(iter.source_value()),
            ValueIter::Dict(iter) => Some(iter.source_value()),
            ValueIter::Str(iter) => Some(iter.source_value()),
        }
    }
}

impl ValueIterator for ValueIter {
    fn next(&mut self, arena: &mut ValueHolder) -> Option<Value> {
        match self {
            ValueIter::Range(iter) => iter.next(arena),
            ValueIter::Collection(iter) => iter.next(arena),
            ValueIter::Dict(iter) => iter.next(arena),
            ValueIter::Str(iter) => iter.next(arena),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash)]
pub enum Value {
    // todo: consolidate ints and floats into single number type
    Int(i64),
    Float(FloatOrd<f64>),
    Bool(bool),
    Range(RangeValue),
    String(StringKey),
    List(ListKey),
    Tuple(TupleKey),
    Dict(DictKey),
    Function(FuncKey),
    Iter(IterKey),
    StructDef(StructDefKey),
    StructInst(StructInstKey),
    Void,
}

impl Value {
    pub fn get_type(&self) -> &str {
        match self {
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Bool(_) => "bool",
            Value::Range(_) => "range",
            Value::String(_) => "string",
            Value::List(_) => "list",
            Value::Tuple(_) => "tuple",
            Value::Dict(_) => "dict",
            Value::Function(_) => "function",
            Value::Iter(_) => "iter",
            Value::StructDef(_) => "struct",
            Value::StructInst(_) => "struct instance",
            Value::Void => "void",
        }
    }

    pub fn is_truthy(self) -> WalrusResult<bool> {
        Ok(match self {
            Value::Void => false,
            Value::Bool(b) => b,
            Value::Int(i) => i != 0,
            Value::Float(FloatOrd(f)) => f != 0.0,
            Value::String(s) => !s.resolve()?.is_empty(),
            Value::List(l) => !l.resolve()?.is_empty(),
            Value::Tuple(t) => !t.resolve()?.is_empty(),
            Value::Dict(d) => !d.resolve()?.is_empty(),
            Value::Range(r) => !r.is_empty(),
            Value::Function(_) => true,
            Value::Iter(_) => true,
            Value::StructDef(_) => true,
            Value::StructInst(_) => true,
        })
    }

    pub fn stringify(self) -> WalrusResult<String> {
        Ok(match self {
            Value::Int(i) => i.to_string(),
            Value::Float(FloatOrd(f)) => f.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Range(r) => r.to_string(),
            Value::String(s) => s.resolve()?.to_string(),
            Value::List(l) => {
                let list = l.resolve()?;
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
            Value::Tuple(t) => {
                let tuple = t.resolve()?;
                let mut s = String::new();

                s.push('(');

                for (i, item) in tuple.iter().enumerate() {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    s.push_str(&item.stringify()?);
                }

                s.push(')');
                s
            }
            Value::Dict(d) => {
                let dict = d.resolve()?;
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
            Value::Function(f) => {
                let func = f.resolve()?;
                func.to_string()
            }
            Value::Iter(_) => "iter".to_string(),
            Value::StructDef(s) => {
                use crate::arenas::with_arena;
                with_arena(|arena| {
                    arena.get_struct_def(s).map(|def| def.to_string())
                })?
            }
            Value::StructInst(s) => {
                use crate::arenas::with_arena;
                with_arena(|arena| {
                    arena.get_struct_inst(s).map(|inst| inst.to_string())
                })?
            }
            Value::Void => "void".to_string(),
        })
    }
}

// fixme: maybe replace with a to_string method, not impl
impl Display for Value {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Value::Int(int) => write!(f, "{}", int),
            Value::Float(FloatOrd(float)) => write!(f, "{}", float),
            Value::Bool(bool) => write!(f, "{}", bool),
            Value::Range(range) => write!(f, "{}", range),
            Value::String(str) => write!(f, "{:?}", str),
            Value::List(list) => write!(f, "{:?}", list),
            Value::Tuple(tuple) => write!(f, "{:?}", tuple),
            Value::Dict(dict) => write!(f, "{:?}", dict),
            Value::Function(func) => write!(f, "{:?}", func),
            Value::Iter(iter) => write!(f, "{:?}", iter),
            Value::StructDef(s) => write!(f, "{:?}", s),
            Value::StructInst(s) => write!(f, "{:?}", s),
            Value::Void => write!(f, "void"),
        }
    }
}
