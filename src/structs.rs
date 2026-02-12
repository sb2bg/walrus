use rustc_hash::FxHashMap;
use std::fmt::{Display, Formatter};

use crate::arenas::StructDefKey;
use crate::function::WalrusFunction;
use crate::value::Value;

/// Represents a struct definition (the "type" or "class")
#[derive(Debug, Clone)]
pub struct StructDefinition {
    name: String,
    methods: FxHashMap<String, WalrusFunction>,
}

impl StructDefinition {
    pub fn new(name: String) -> Self {
        Self {
            name,
            methods: FxHashMap::default(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn add_method(&mut self, name: String, function: WalrusFunction) {
        self.methods.insert(name, function);
    }

    pub fn get_method(&self, name: &str) -> Option<&WalrusFunction> {
        self.methods.get(name)
    }

    pub fn methods(&self) -> &FxHashMap<String, WalrusFunction> {
        &self.methods
    }
}

impl Display for StructDefinition {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "<struct {}>", self.name)
    }
}

/// Represents an instance of a struct
#[derive(Debug, Clone)]
pub struct StructInstance {
    struct_name: String,
    struct_def: StructDefKey,
    fields: FxHashMap<String, Value>,
}

impl StructInstance {
    pub fn new(struct_name: String, struct_def: StructDefKey) -> Self {
        Self {
            struct_name,
            struct_def,
            fields: FxHashMap::default(),
        }
    }

    pub fn struct_name(&self) -> &str {
        &self.struct_name
    }

    pub fn struct_def(&self) -> StructDefKey {
        self.struct_def
    }

    pub fn set_field(&mut self, name: String, value: Value) {
        self.fields.insert(name, value);
    }

    pub fn get_field(&self, name: &str) -> Option<&Value> {
        self.fields.get(name)
    }

    pub fn fields(&self) -> &FxHashMap<String, Value> {
        &self.fields
    }

    pub fn fields_mut(&mut self) -> &mut FxHashMap<String, Value> {
        &mut self.fields
    }
}

impl Display for StructInstance {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "<instance of {}>", self.struct_name)
    }
}
