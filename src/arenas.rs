use float_ord::FloatOrd;
use once_cell::sync::Lazy;
use rustc_hash::FxHashMap;
use slotmap::{new_key_type, DenseSlotMap};
use strena::{Interner, Symbol};

use crate::ast::Node;
use crate::error::WalrusError;
use crate::iter::{CollectionIter, DictIter, RangeIter, StrIter};
use crate::rust_function::RustFunction;
use crate::value::{Value, ValueIter};
use crate::WalrusResult;

pub static mut ARENA: Lazy<ValueHolder> = Lazy::new(ValueHolder::new);

// todo: maybe instead of this, we can use a single slotmap
// and use a different enum to differentiate between the
// types stored by value and the types stored by key
// fixme: eventually this will have to be garbage collected
// fixme: maybe we can just replace this with RC values in ValueKind
// and then just clone everything and the copy types would just
// be copied and the rc types would be cloned
// todo: maybe use a different arena library, DenseSlotMap is mid-performance
#[derive(Default)]
pub struct ValueHolder {
    dict_slotmap: DenseSlotMap<DictKey, FxHashMap<Value, Value>>,
    list_slotmap: DenseSlotMap<ListKey, Vec<Value>>,
    tuple_slotmap: DenseSlotMap<TupleKey, Vec<Value>>, // todo: use slice?
    string_interner: Interner,
    function_slotmap: DenseSlotMap<FuncKey, (String, Vec<String>, Node)>,
    rust_function_slotmap: DenseSlotMap<RustFuncKey, RustFunction>,
    iter_slotmap: DenseSlotMap<IterKey, ValueIter>,
}

impl ValueHolder {
    pub fn new() -> Self {
        Self {
            dict_slotmap: DenseSlotMap::with_key(),
            list_slotmap: DenseSlotMap::with_key(),
            tuple_slotmap: DenseSlotMap::with_key(),
            string_interner: Interner::default(), // todo: use FxHasher
            function_slotmap: DenseSlotMap::with_key(),
            rust_function_slotmap: DenseSlotMap::with_key(),
            iter_slotmap: DenseSlotMap::with_key(),
        }
    }

    pub fn free(&mut self, key: Value) -> bool {
        match key {
            Value::Dict(key) => self.dict_slotmap.remove(key).is_some(),
            Value::List(key) => self.list_slotmap.remove(key).is_some(),
            Value::String(key) => {
                // fixme: for now, no way to free strings
                // self.string_interner.remove(key).is_some()
                false
            }
            Value::Function(key) => self.function_slotmap.remove(key).is_some(),
            Value::RustFunction(key) => self.rust_function_slotmap.remove(key).is_some(),
            Value::Tuple(key) => self.tuple_slotmap.remove(key).is_some(),
            Value::Iter(key) => self.iter_slotmap.remove(key).is_some(),
            _ => false,
        }
    }

    // todo: split this into multiple functions for each type
    pub fn push(&mut self, value: HeapValue) -> Value {
        match value {
            HeapValue::List(list) => Value::List(self.list_slotmap.insert(list)),
            HeapValue::Tuple(tuple) => Value::Tuple(self.tuple_slotmap.insert(tuple.to_vec())),
            HeapValue::Dict(dict) => Value::Dict(self.dict_slotmap.insert(dict)),
            HeapValue::Function(func) => Value::Function(self.function_slotmap.insert(func)),
            HeapValue::RustFunction(rust_func) => {
                Value::RustFunction(self.rust_function_slotmap.insert(rust_func))
            }
            HeapValue::String(string) => Value::String(self.string_interner.get_or_insert(string)),
            HeapValue::Iter(iter) => Value::Iter(self.iter_slotmap.insert(iter)),
        }
    }

    pub fn is_truthy(&self, value: Value) -> WalrusResult<bool> {
        Ok(match value {
            Value::Bool(b) => b,
            Value::Int(i) => i != 0,
            Value::Float(f) => f != FloatOrd(0.0),
            Value::String(s) => !self.get_string(s)?.is_empty(),
            Value::List(l) => !self.get_list(l)?.is_empty(),
            Value::Dict(d) => !self.get_dict(d)?.is_empty(),
            Value::Tuple(t) => !self.get_tuple(t)?.is_empty(),
            Value::Range(range) => !range.is_empty(),
            Value::Function(_) => true,
            Value::Iter(_) => true,
            Value::RustFunction(_) => true,
            Value::Void => false,
        })
    }

    pub fn push_ident(&mut self, ident: &str) -> Symbol {
        self.string_interner.get_or_insert(ident)
    }

    pub fn get_rust_function(&self, key: RustFuncKey) -> WalrusResult<&RustFunction> {
        Self::check(self.rust_function_slotmap.get(key))
    }

    pub fn get_mut_dict(&mut self, key: DictKey) -> WalrusResult<&mut FxHashMap<Value, Value>> {
        Self::check(self.dict_slotmap.get_mut(key))
    }

    pub fn get_dict(&self, key: DictKey) -> WalrusResult<&FxHashMap<Value, Value>> {
        Self::check(self.dict_slotmap.get(key))
    }

    pub fn get_mut_list(&mut self, key: ListKey) -> WalrusResult<&mut Vec<Value>> {
        Self::check(self.list_slotmap.get_mut(key))
    }

    pub fn get_list(&self, key: ListKey) -> WalrusResult<&Vec<Value>> {
        Self::check(self.list_slotmap.get(key))
    }

    pub fn get_tuple(&self, key: TupleKey) -> WalrusResult<&Vec<Value>> {
        Self::check(self.tuple_slotmap.get(key))
    }

    pub fn get_string(&self, key: Symbol) -> WalrusResult<&str> {
        Self::check(self.string_interner.resolve(key))
    }

    pub fn get_function(&self, key: FuncKey) -> WalrusResult<&(String, Vec<String>, Node)> {
        Self::check(self.function_slotmap.get(key))
    }

    pub fn get_iter(&self, key: IterKey) -> WalrusResult<&ValueIter> {
        Self::check(self.iter_slotmap.get(key))
    }

    pub fn get_mut_iter(&mut self, key: IterKey) -> WalrusResult<&mut ValueIter> {
        Self::check(self.iter_slotmap.get_mut(key))
    }

    pub fn value_to_iter(&mut self, value: Value) -> WalrusResult<Value> {
        let key = match value {
            Value::List(list) => {
                let iter = CollectionIter::new(self.get_list(list)?.clone());
                self.push(HeapValue::Iter(Box::new(iter)))
            }
            Value::Tuple(tuple) => {
                let iter = CollectionIter::new(self.get_tuple(tuple)?.clone());
                self.push(HeapValue::Iter(Box::new(iter)))
            }
            Value::Dict(dict) => {
                let iter = DictIter::new(self.get_dict(dict)?.clone());
                self.push(HeapValue::Iter(Box::new(iter)))
            }
            Value::String(string) => {
                let iter = StrIter::new(self.get_string(string)?.to_string());
                self.push(HeapValue::Iter(Box::new(iter)))
            }
            Value::Range(range) => self.push(HeapValue::Iter(Box::new(RangeIter::new(range)))),
            Value::Iter(iter) => Value::Iter(iter),
            _ => {
                return Err(WalrusError::GenericError {
                    message: "TODO: implement error for not iterable".into(),
                })
            }
        };

        Ok(key)
    }

    fn check<T>(result: Option<T>) -> WalrusResult<T> {
        result.ok_or(WalrusError::UnknownError {
            message: "Attempt to access released memory".into(), // fixme: use correct AccessReleasedMemory error
        })
    }

    // todo: speed this up
    pub fn stringify(&self, value: Value) -> WalrusResult<String> {
        Ok(match value {
            Value::Int(i) => i.to_string(),
            Value::Float(FloatOrd(f)) => f.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Range(r) => r.to_string(),
            Value::String(s) => self.get_string(s)?.to_string(),
            Value::List(l) => {
                let list = self.get_list(l)?;
                let mut s = String::new();

                s.push('[');

                for (i, &item) in list.iter().enumerate() {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    s.push_str(&self.stringify(item)?);
                }

                s.push(']');
                s
            }
            Value::Tuple(t) => {
                let tuple = self.get_tuple(t)?;
                let mut s = String::new();

                s.push('(');

                for (i, &item) in tuple.iter().enumerate() {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    s.push_str(&self.stringify(item)?);
                }

                s.push(')');
                s
            }
            Value::Dict(d) => {
                let dict = self.get_dict(d)?;
                let mut s = String::new();

                s.push('{');

                for (i, (&key, &value)) in dict.iter().enumerate() {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    s.push_str(&self.stringify(key)?);
                    s.push_str(": ");
                    s.push_str(&self.stringify(value)?);
                }

                s.push('}');
                s
            }
            Value::Function(f) => {
                let (name, args, _) = self.get_function(f)?;
                let mut s = String::new();

                s.push_str("function ");
                s.push_str(name);
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
            Value::RustFunction(r) => {
                let rust_func = self.get_rust_function(r)?;
                let mut s = String::new();

                s.push_str("<rust function ");
                s.push_str(rust_func.name());
                s.push('>');

                s
            }
            Value::Iter(_) => "<iter object>".to_string(),
            Value::Void => "void".to_string(),
        })
    }
}

pub enum HeapValue<'a> {
    List(Vec<Value>),
    Tuple(&'a [Value]),
    Dict(FxHashMap<Value, Value>),
    Function((String, Vec<String>, Node)),
    RustFunction(RustFunction),
    String(&'a str),
    Iter(ValueIter),
}

impl HeapValue<'_> {
    pub fn alloc(self) -> Value {
        unsafe { ARENA.push(self) }
    }
}

new_key_type! {
    pub struct ListKey;
    pub struct DictKey;
    pub struct FuncKey;
    pub struct RustFuncKey;
    pub struct TupleKey;
    pub struct IterKey;
}

pub trait Free {
    fn free(&mut self) -> bool;
}

pub trait Resolve<'a> {
    type Output;
    fn resolve(self) -> WalrusResult<Self::Output>;
}

pub trait ResolveMut<'a> {
    type Output;
    fn resolve_mut(self) -> WalrusResult<Self::Output>;
}

impl Free for Value {
    fn free(&mut self) -> bool {
        unsafe { ARENA.free(*self) }
    }
}

impl<'a> Resolve<'a> for ListKey {
    type Output = &'a Vec<Value>;

    fn resolve(self) -> WalrusResult<Self::Output> {
        unsafe { ARENA.get_list(self) }
    }
}

impl<'a> Resolve<'a> for DictKey {
    type Output = &'a FxHashMap<Value, Value>;

    fn resolve(self) -> WalrusResult<Self::Output> {
        unsafe { ARENA.get_dict(self) }
    }
}

impl<'a> Resolve<'a> for FuncKey {
    type Output = &'a (String, Vec<String>, Node);

    fn resolve(self) -> WalrusResult<Self::Output> {
        unsafe { ARENA.get_function(self) }
    }
}

impl<'a> Resolve<'a> for RustFuncKey {
    type Output = &'a RustFunction;

    fn resolve(self) -> WalrusResult<Self::Output> {
        unsafe { ARENA.get_rust_function(self) }
    }
}

impl<'a> Resolve<'a> for Symbol {
    type Output = &'a str;

    fn resolve(self) -> WalrusResult<Self::Output> {
        unsafe { ARENA.get_string(self) }
    }
}

impl<'a> ResolveMut<'a> for ListKey {
    type Output = &'a mut Vec<Value>;

    fn resolve_mut(self) -> WalrusResult<Self::Output> {
        unsafe { ARENA.get_mut_list(self) }
    }
}

impl<'a> ResolveMut<'a> for DictKey {
    type Output = &'a mut FxHashMap<Value, Value>;

    fn resolve_mut(self) -> WalrusResult<Self::Output> {
        unsafe { ARENA.get_mut_dict(self) }
    }
}

impl<'a> Resolve<'a> for TupleKey {
    type Output = &'a Vec<Value>;

    fn resolve(self) -> WalrusResult<Self::Output> {
        unsafe { ARENA.get_tuple(self) }
    }
}
