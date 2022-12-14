use crate::ast::Node;
use crate::error::WalrusError;
use crate::value::Value;
use slotmap::{new_key_type, DenseSlotMap};
use std::collections::HashMap;

new_key_type! {
    pub struct ListKey;
    pub struct DictKey;
    pub struct StringKey;
    pub struct FuncKey;
}

type ArenaResult<T> = Result<T, WalrusError>;

// todo: maybe instead of this, we can use a single slotmap
// and use a different enum to differentiate between the
// types stored by value and the types stored by key
#[derive(Debug)]
pub struct ValueHolder {
    dict_slotmap: DenseSlotMap<DictKey, HashMap<Value, Value>>,
    list_slotmap: DenseSlotMap<ListKey, Vec<Value>>,
    string_slotmap: DenseSlotMap<StringKey, String>,
    function_slotmap: DenseSlotMap<FuncKey, (String, Vec<String>, Node)>,
}

impl ValueHolder {
    pub fn new() -> Self {
        Self {
            dict_slotmap: DenseSlotMap::with_key(),
            list_slotmap: DenseSlotMap::with_key(),
            string_slotmap: DenseSlotMap::with_key(),
            function_slotmap: DenseSlotMap::with_key(),
        }
    }

    pub fn insert_dict(&mut self, dict: HashMap<Value, Value>) -> Value {
        Value::Dict(self.dict_slotmap.insert(dict))
    }

    pub fn insert_list(&mut self, list: Vec<Value>) -> Value {
        Value::List(self.list_slotmap.insert(list))
    }

    pub fn insert_string(&mut self, string: String) -> Value {
        Value::String(self.string_slotmap.insert(string))
    }

    pub fn insert_function(&mut self, name: String, args: Vec<String>, body: Node) -> Value {
        Value::Function(self.function_slotmap.insert((name, args, body)))
    }

    pub fn get_dict(&self, key: DictKey) -> ArenaResult<&HashMap<Value, Value>> {
        Self::check(self.dict_slotmap.get(key))
    }

    pub fn get_list(&self, key: ListKey) -> ArenaResult<&Vec<Value>> {
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
            message: "Attempt to access released reference".into(),
        })
    }
}
