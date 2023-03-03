use float_ord::FloatOrd;
use once_cell::sync::Lazy;
use rustc_hash::FxHashMap;
use slotmap::{new_key_type, DenseSlotMap};
use strena::{Interner, Symbol};

use crate::ast::Node;
use crate::error::WalrusError;
use crate::rust_function::RustFunction;
use crate::value::ValueKind;
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
    dict_slotmap: DenseSlotMap<DictKey, FxHashMap<ValueKind, ValueKind>>,
    list_slotmap: DenseSlotMap<ListKey, Vec<ValueKind>>,
    string_interner: Interner,
    function_slotmap: DenseSlotMap<FuncKey, (String, Vec<String>, Node)>,
    rust_function_slotmap: DenseSlotMap<RustFuncKey, RustFunction>,
}

impl ValueHolder {
    pub fn new() -> Self {
        Self {
            dict_slotmap: DenseSlotMap::with_key(),
            list_slotmap: DenseSlotMap::with_key(),
            string_interner: Interner::default(), // todo: use FxHasher
            function_slotmap: DenseSlotMap::with_key(),
            rust_function_slotmap: DenseSlotMap::with_key(),
        }
    }

    pub fn free(&mut self, key: ValueKind) -> bool {
        match key {
            ValueKind::Dict(key) => self.dict_slotmap.remove(key).is_some(),
            ValueKind::List(key) => self.list_slotmap.remove(key).is_some(),
            ValueKind::String(key) => {
                // fixme: for now, no way to free strings
                // self.string_interner.remove(key).is_some()
                false
            }
            ValueKind::Function(key) => self.function_slotmap.remove(key).is_some(),
            ValueKind::RustFunction(key) => self.rust_function_slotmap.remove(key).is_some(),
            _ => false,
        }
    }

    // todo: split this into multiple functions for each type
    pub fn push(&mut self, value: HeapValue) -> ValueKind {
        match value {
            HeapValue::List(list) => ValueKind::List(self.list_slotmap.insert(list)),
            HeapValue::Dict(dict) => ValueKind::Dict(self.dict_slotmap.insert(dict)),
            HeapValue::Function(func) => ValueKind::Function(self.function_slotmap.insert(func)),
            HeapValue::RustFunction(rust_func) => {
                ValueKind::RustFunction(self.rust_function_slotmap.insert(rust_func))
            }
            HeapValue::String(string) => {
                ValueKind::String(self.string_interner.get_or_insert(&string))
            }
        }
    }

    pub fn push_ident(&mut self, ident: &str) -> Symbol {
        self.string_interner.get_or_insert(ident)
    }

    pub fn get_rust_function(&self, key: RustFuncKey) -> WalrusResult<&RustFunction> {
        Self::check(self.rust_function_slotmap.get(key))
    }

    pub fn get_mut_dict(
        &mut self,
        key: DictKey,
    ) -> WalrusResult<&mut FxHashMap<ValueKind, ValueKind>> {
        Self::check(self.dict_slotmap.get_mut(key))
    }

    pub fn get_dict(&self, key: DictKey) -> WalrusResult<&FxHashMap<ValueKind, ValueKind>> {
        Self::check(self.dict_slotmap.get(key))
    }

    pub fn get_mut_list(&mut self, key: ListKey) -> WalrusResult<&mut Vec<ValueKind>> {
        Self::check(self.list_slotmap.get_mut(key))
    }

    pub fn get_list(&self, key: ListKey) -> WalrusResult<&Vec<ValueKind>> {
        Self::check(self.list_slotmap.get(key))
    }

    pub fn get_string(&self, key: Symbol) -> WalrusResult<&str> {
        Self::check(self.string_interner.resolve(key))
    }

    pub fn get_function(&self, key: FuncKey) -> WalrusResult<&(String, Vec<String>, Node)> {
        Self::check(self.function_slotmap.get(key))
    }

    fn check<T>(result: Option<T>) -> WalrusResult<T> {
        result.ok_or_else(|| {
            Box::new(WalrusError::UnknownError {
                message: "Attempt to access released memory".into(), // fixme: use correct AccessReleasedMemory error
            })
        })
    }

    // todo: speed this up
    pub fn stringify(&self, value: ValueKind) -> WalrusResult<String> {
        Ok(match value {
            ValueKind::Int(i) => i.to_string(),
            ValueKind::Float(FloatOrd(f)) => f.to_string(),
            ValueKind::Bool(b) => b.to_string(),
            ValueKind::Range(r) => r.to_string(),
            ValueKind::String(s) => self.get_string(s)?.to_string(),
            ValueKind::List(l) => {
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
            ValueKind::Dict(d) => {
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
            ValueKind::Function(f) => {
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
            ValueKind::RustFunction(r) => {
                let rust_func = self.get_rust_function(r)?;
                let mut s = String::new();

                s.push_str("rust_function ");
                s.push_str(rust_func.name());
                s
            }
            ValueKind::Void => "void".to_string(),
        })
    }
}

pub enum HeapValue {
    List(Vec<ValueKind>),
    Dict(FxHashMap<ValueKind, ValueKind>),
    Function((String, Vec<String>, Node)),
    RustFunction(RustFunction),
    String(String),
}

impl HeapValue {
    pub fn alloc(self) -> ValueKind {
        unsafe { ARENA.push(self) }
    }
}

new_key_type! {
    pub struct ListKey;
    pub struct DictKey;
    pub struct FuncKey;
    pub struct RustFuncKey;
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

impl Free for ValueKind {
    fn free(&mut self) -> bool {
        unsafe { ARENA.free(*self) }
    }
}

impl<'a> Resolve<'a> for ListKey {
    type Output = &'a Vec<ValueKind>;

    fn resolve(self) -> WalrusResult<Self::Output> {
        unsafe { ARENA.get_list(self) }
    }
}

impl<'a> Resolve<'a> for DictKey {
    type Output = &'a FxHashMap<ValueKind, ValueKind>;

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
    type Output = &'a mut Vec<ValueKind>;

    fn resolve_mut(self) -> WalrusResult<Self::Output> {
        unsafe { ARENA.get_mut_list(self) }
    }
}

impl<'a> ResolveMut<'a> for DictKey {
    type Output = &'a mut FxHashMap<ValueKind, ValueKind>;

    fn resolve_mut(self) -> WalrusResult<Self::Output> {
        unsafe { ARENA.get_mut_dict(self) }
    }
}
