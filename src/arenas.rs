use crate::ast::Node;
use crate::error::WalrusError;
use crate::interpreter::InterpreterResult;
use crate::value::ValueKind;
use log::debug;
use slotmap::{new_key_type, DenseSlotMap};
use std::collections::HashMap;

new_key_type! {
    pub struct ListKey;
    pub struct DictKey;
    pub struct StringKey;
    pub struct FuncKey;
    pub struct RustFuncKey;
}

type ArenaResult<T> = Result<T, WalrusError>;
type RustFunction = fn(Vec<ValueKind>) -> InterpreterResult;

// todo: maybe instead of this, we can use a single slotmap
// and use a different enum to differentiate between the
// types stored by value and the types stored by key
// fixme: eventually this will have to be garbage collected
#[derive(Debug, Clone)]
pub struct ValueHolder {
    dict_slotmap: DenseSlotMap<DictKey, HashMap<ValueKind, ValueKind>>,
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

    pub fn dump(&self) {
        debug!("Arena dump");

        debug!("Dictionaries: {:?}", self.dict_slotmap);
        debug!("Lists: {:?}", self.list_slotmap);
        debug!("Strings: {:?}", self.string_slotmap);
        debug!("Functions: {:?}", self.function_slotmap);
        debug!("Rust Functions: {:?}", self.rust_function_slotmap);
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

    pub fn insert_dict(&mut self, dict: HashMap<ValueKind, ValueKind>) -> ValueKind {
        ValueKind::Dict(self.dict_slotmap.insert(dict))
    }

    pub fn insert_list(&mut self, list: Vec<ValueKind>) -> ValueKind {
        ValueKind::List(self.list_slotmap.insert(list))
    }

    pub fn insert_string(&mut self, string: String) -> ValueKind {
        ValueKind::String(self.string_slotmap.insert(string))
    }

    pub fn insert_function(&mut self, name: String, args: Vec<String>, body: Node) -> ValueKind {
        ValueKind::Function(self.function_slotmap.insert((name, args, body)))
    }

    pub fn insert_rust_function(&mut self, func: RustFunction) -> ValueKind {
        ValueKind::RustFunction(self.rust_function_slotmap.insert(func))
    }

    pub fn get_rust_function(&self, key: RustFuncKey) -> ArenaResult<&RustFunction> {
        Self::check(self.rust_function_slotmap.get(key))
    }

    pub fn get_dict(&self, key: DictKey) -> ArenaResult<&HashMap<ValueKind, ValueKind>> {
        Self::check(self.dict_slotmap.get(key))
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
            message: "Attempt to access released memory".into(),
        })
    }
}
