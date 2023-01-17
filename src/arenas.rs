use crate::ast::Node;
use crate::error::WalrusError;
use crate::interpreter::{Interpreter, InterpreterResult};
use crate::span::Span;
use crate::value::{HeapValue, ValueKind};
use rustc_hash::FxHashMap;
use slotmap::{new_key_type, DenseSlotMap};

new_key_type! {
    pub struct ListKey;
    pub struct DictKey;
    pub struct StringKey;
    pub struct FuncKey;
    pub struct RustFuncKey;
}

pub type ArenaResult<T> = Result<T, WalrusError>;
pub type RustFunction = (
    fn(Vec<ValueKind>, interpreter: &Interpreter, span: Span) -> InterpreterResult,
    Option<usize>,
);

// todo: maybe instead of this, we can use a single slotmap
// and use a different enum to differentiate between the
// types stored by value and the types stored by key
// fixme: eventually this will have to be garbage collected
// fixme: maybe we can just replace this with RC values in ValueKind
// and then just clone everything and the copy types would just
// be copied and the rc types would be cloned
pub struct ValueHolder {
    dict_slotmap: DenseSlotMap<DictKey, FxHashMap<ValueKind, ValueKind>>,
    list_slotmap: DenseSlotMap<ListKey, Vec<ValueKind>>,
    string_slotmap: DenseSlotMap<StringKey, String>,
    function_slotmap: DenseSlotMap<FuncKey, (String, Vec<String>, Node)>,
    rust_function_slotmap: DenseSlotMap<RustFuncKey, RustFunction>,
}

impl ValueHolder {
    pub fn new() -> Self {
        Self {
            dict_slotmap: DenseSlotMap::with_key(),
            list_slotmap: DenseSlotMap::with_key(),
            string_slotmap: DenseSlotMap::with_key(),
            function_slotmap: DenseSlotMap::with_key(),
            rust_function_slotmap: DenseSlotMap::with_key(),
        }
    }

    pub fn free(&mut self, key: ValueKind) -> bool {
        match key {
            ValueKind::Dict(key) => self.dict_slotmap.remove(key).is_some(),
            ValueKind::List(key) => self.list_slotmap.remove(key).is_some(),
            ValueKind::String(key) => self.string_slotmap.remove(key).is_some(),
            ValueKind::Function(key) => self.function_slotmap.remove(key).is_some(),
            ValueKind::RustFunction(key) => self.rust_function_slotmap.remove(key).is_some(),
            _ => false,
        }
    }

    pub fn alloc(&mut self, value: HeapValue) -> ValueKind {
        match value {
            HeapValue::List(list) => ValueKind::List(self.list_slotmap.insert(list)),
            HeapValue::Dict(dict) => ValueKind::Dict(self.dict_slotmap.insert(dict)),
            HeapValue::Function(func) => ValueKind::Function(self.function_slotmap.insert(func)),
            HeapValue::RustFunction(rust_func) => {
                ValueKind::RustFunction(self.rust_function_slotmap.insert(rust_func))
            }
            HeapValue::String(string) => ValueKind::String(self.string_slotmap.insert(string)),
        }
    }

    pub fn get_rust_function(&self, key: RustFuncKey) -> ArenaResult<&RustFunction> {
        Self::check(self.rust_function_slotmap.get(key))
    }

    pub fn get_mut_dict(
        &mut self,
        key: DictKey,
    ) -> ArenaResult<&mut FxHashMap<ValueKind, ValueKind>> {
        Self::check(self.dict_slotmap.get_mut(key))
    }

    pub fn get_dict(&self, key: DictKey) -> ArenaResult<&FxHashMap<ValueKind, ValueKind>> {
        Self::check(self.dict_slotmap.get(key))
    }

    pub fn get_mut_list(&mut self, key: ListKey) -> ArenaResult<&mut Vec<ValueKind>> {
        Self::check(self.list_slotmap.get_mut(key))
    }

    pub fn get_list(&self, key: ListKey) -> ArenaResult<&Vec<ValueKind>> {
        Self::check(self.list_slotmap.get(key))
    }

    pub fn get_string(&self, key: StringKey) -> ArenaResult<&String> {
        Self::check(self.string_slotmap.get(key))
    }

    pub fn get_function(&self, key: FuncKey) -> ArenaResult<&(String, Vec<String>, Node)> {
        Self::check(self.function_slotmap.get(key))
    }

    fn check<T>(result: Option<T>) -> Result<T, WalrusError> {
        result.ok_or(WalrusError::UnknownError {
            message: "Attempt to access released memory".into(), // fixme: use correct AccessReleasedMemory error
        })
    }
}
