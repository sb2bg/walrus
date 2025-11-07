use float_ord::FloatOrd;
use once_cell::sync::Lazy;
use rustc_hash::FxHashMap;
use slotmap::{DenseSlotMap, new_key_type};
use strena::{Interner, Symbol};

use crate::WalrusResult;
use crate::error::WalrusError;
use crate::function::WalrusFunction;
use crate::iter::{CollectionIter, DictIter, RangeIter, StrIter};
use crate::value::{Value, ValueIter};

pub static mut ARENA: Lazy<ValueHolder> = Lazy::new(ValueHolder::new);

// todo: maybe instead of this, we can use a single slotmap
// and use a different enum to differentiate between the
// types stored by value and the types stored by key
// fixme: eventually this will have to be garbage collected
// fixme: maybe we can just replace this with RC values in ValueKind
// and then just clone everything and the copy types would just
// be copied and the rc types would be cloned
// todo: maybe use a different arena library, DenseSlotMap is mid-performance
#[derive(Default, Debug, Clone)]
pub struct ValueHolder {
    dicts: DenseSlotMap<DictKey, FxHashMap<Value, Value>>,
    lists: DenseSlotMap<ListKey, Vec<Value>>,
    tuples: DenseSlotMap<TupleKey, Vec<Value>>, // todo: use slice?
    strings: Interner,
    functions: DenseSlotMap<FuncKey, WalrusFunction>,
    iterators: DenseSlotMap<IterKey, ValueIter>,
}

impl ValueHolder {
    pub fn new() -> Self {
        Self {
            dicts: DenseSlotMap::with_key(),
            lists: DenseSlotMap::with_key(),
            tuples: DenseSlotMap::with_key(),
            strings: Interner::default(), // todo: use FxHasher
            functions: DenseSlotMap::with_key(),
            iterators: DenseSlotMap::with_key(),
        }
    }

    pub fn free(&mut self, key: Value) -> bool {
        match key {
            Value::Dict(key) => self.dicts.remove(key).is_some(),
            Value::List(key) => self.lists.remove(key).is_some(),
            Value::String(_key) => {
                // fixme: for now, no way to free strings
                // self.string_interner.remove(key).is_some()
                false
            }
            Value::Function(key) => self.functions.remove(key).is_some(),
            Value::Tuple(key) => self.tuples.remove(key).is_some(),
            Value::Iter(key) => self.iterators.remove(key).is_some(),
            _ => false,
        }
    }

    // todo: split this into multiple functions for each type
    pub fn push(&mut self, value: HeapValue) -> Value {
        match value {
            HeapValue::List(list) => Value::List(self.lists.insert(list)),
            HeapValue::Tuple(tuple) => Value::Tuple(self.tuples.insert(tuple.to_vec())),
            HeapValue::Dict(dict) => Value::Dict(self.dicts.insert(dict)),
            HeapValue::Function(func) => Value::Function(self.functions.insert(func)),
            HeapValue::String(string) => Value::String(self.strings.get_or_insert(string)),
            HeapValue::Iter(iter) => Value::Iter(self.iterators.insert(iter)),
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
            Value::Void => false,
        })
    }

    pub fn push_ident(&mut self, ident: &str) -> Symbol {
        self.strings.get_or_insert(ident)
    }

    pub fn get_mut_dict(&mut self, key: DictKey) -> WalrusResult<&mut FxHashMap<Value, Value>> {
        Self::check(self.dicts.get_mut(key))
    }

    pub fn get_dict(&self, key: DictKey) -> WalrusResult<&FxHashMap<Value, Value>> {
        Self::check(self.dicts.get(key))
    }

    pub fn get_mut_list(&mut self, key: ListKey) -> WalrusResult<&mut Vec<Value>> {
        Self::check(self.lists.get_mut(key))
    }

    pub fn get_list(&self, key: ListKey) -> WalrusResult<&Vec<Value>> {
        Self::check(self.lists.get(key))
    }

    pub fn get_tuple(&self, key: TupleKey) -> WalrusResult<&Vec<Value>> {
        Self::check(self.tuples.get(key))
    }

    pub fn get_string(&self, key: Symbol) -> WalrusResult<&str> {
        Self::check(self.strings.resolve(key))
    }

    pub fn get_function(&self, key: FuncKey) -> WalrusResult<&WalrusFunction> {
        Self::check(self.functions.get(key))
    }

    pub fn get_iter(&self, key: IterKey) -> WalrusResult<&ValueIter> {
        Self::check(self.iterators.get(key))
    }

    pub fn get_mut_iter(&mut self, key: IterKey) -> WalrusResult<&mut ValueIter> {
        Self::check(self.iterators.get_mut(key))
    }

    pub fn value_to_iter(&mut self, value: Value) -> WalrusResult<Value> {
        let key = match value {
            Value::List(list) => {
                let iter = CollectionIter::new(self.get_list(list)?);
                self.push(HeapValue::Iter(ValueIter::Collection(iter)))
            }
            Value::Tuple(tuple) => {
                let iter = CollectionIter::new(self.get_tuple(tuple)?);
                self.push(HeapValue::Iter(ValueIter::Collection(iter)))
            }
            Value::Dict(dict) => {
                let iter = DictIter::new(self.get_dict(dict)?);
                self.push(HeapValue::Iter(ValueIter::Dict(iter)))
            }
            Value::String(string) => {
                let iter = StrIter::new(self.get_string(string)?);
                self.push(HeapValue::Iter(ValueIter::Str(iter)))
            }
            Value::Range(range) => {
                self.push(HeapValue::Iter(ValueIter::Range(RangeIter::new(range))))
            }
            Value::Iter(iter) => Value::Iter(iter),
            _ => {
                return Err(WalrusError::TodoError {
                    //todo
                    message: "TODO: implement error for not iterable".into(),
                });
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
                let func = self.get_function(f)?;
                func.to_string()
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
    Function(WalrusFunction),
    String(&'a str),
    Iter(ValueIter),
}

impl HeapValue<'_> {
    pub fn alloc(self) -> Value {
        unsafe { (*std::ptr::addr_of_mut!(ARENA)).push(self) }
    }
}

new_key_type! {
    pub struct ListKey;
    pub struct DictKey;
    pub struct FuncKey;
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
        unsafe { (*std::ptr::addr_of_mut!(ARENA)).free(*self) }
    }
}

impl<'a> Resolve<'a> for ListKey {
    type Output = &'a Vec<Value>;

    fn resolve(self) -> WalrusResult<Self::Output> {
        unsafe { (*std::ptr::addr_of_mut!(ARENA)).get_list(self) }
    }
}

impl<'a> Resolve<'a> for DictKey {
    type Output = &'a FxHashMap<Value, Value>;

    fn resolve(self) -> WalrusResult<Self::Output> {
        unsafe { (*std::ptr::addr_of_mut!(ARENA)).get_dict(self) }
    }
}

impl<'a> Resolve<'a> for FuncKey {
    type Output = &'a WalrusFunction;

    fn resolve(self) -> WalrusResult<Self::Output> {
        unsafe { (*std::ptr::addr_of_mut!(ARENA)).get_function(self) }
    }
}

impl<'a> Resolve<'a> for Symbol {
    type Output = &'a str;

    fn resolve(self) -> WalrusResult<Self::Output> {
        unsafe { (*std::ptr::addr_of_mut!(ARENA)).get_string(self) }
    }
}

impl<'a> ResolveMut<'a> for ListKey {
    type Output = &'a mut Vec<Value>;

    fn resolve_mut(self) -> WalrusResult<Self::Output> {
        unsafe { (*std::ptr::addr_of_mut!(ARENA)).get_mut_list(self) }
    }
}

impl<'a> ResolveMut<'a> for DictKey {
    type Output = &'a mut FxHashMap<Value, Value>;

    fn resolve_mut(self) -> WalrusResult<Self::Output> {
        unsafe { (*std::ptr::addr_of_mut!(ARENA)).get_mut_dict(self) }
    }
}

impl<'a> Resolve<'a> for TupleKey {
    type Output = &'a Vec<Value>;

    fn resolve(self) -> WalrusResult<Self::Output> {
        unsafe { (*std::ptr::addr_of_mut!(ARENA)).get_tuple(self) }
    }
}
