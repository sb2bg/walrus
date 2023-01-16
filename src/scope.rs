use crate::arenas::{
    ArenaResult, DictKey, FuncKey, ListKey, RustFuncKey, RustFunction, StringKey, ValueHolder,
};
use crate::ast::Node;
use crate::value::{HeapValue, ValueKind};
use log::debug;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::ptr::NonNull;

static mut ARENA: Lazy<ValueHolder> = Lazy::new(ValueHolder::new);

#[derive(Debug)]
pub struct Scope {
    name: String,
    // todo: add line and file name for stack trace
    vars: HashMap<String, ValueKind>,
    parent: Option<NonNull<Scope>>,
}

impl Scope {
    pub fn new() -> Self {
        Self {
            name: "global".to_string(),
            vars: HashMap::new(),
            parent: None,
        }
    }

    pub fn dump(&self) {
        unsafe {
            ARENA.dump();
        }

        debug!("Scope dump: {}", self.name);

        for (k, v) in self.vars.iter() {
            debug!("{}: {}", k, v);
        }

        if let Some(parent) = self.parent {
            unsafe { parent.as_ref().dump() };
        }
    }

    pub fn new_child(&self, name: String) -> Self {
        Self {
            name,
            vars: HashMap::new(),
            parent: Some(NonNull::from(self)),
        }
    }

    pub fn get(&self, name: &str) -> Option<ValueKind> {
        if let Some(value) = self.vars.get(name) {
            Some(*value)
        } else {
            match self.parent {
                Some(parent) => unsafe { parent.as_ref().get(name) },
                _ => None,
            }
        }
    }

    pub fn heap_alloc(&mut self, value: HeapValue) -> ValueKind {
        unsafe { ARENA.alloc(value) }
    }

    pub fn define(&mut self, name: String, value: ValueKind) {
        if name == "_" {
            return;
        }

        self.vars.insert(name, value);
    }

    pub fn reassign(&mut self, name: String, value: ValueKind) -> Result<(), String> {
        if self.vars.contains_key(&name) {
            self.define(name, value);
            Ok(())
        } else {
            match self.parent {
                Some(mut parent) => unsafe { parent.as_mut().reassign(name, value) },
                _ => Err(name),
            }
        }
    }

    pub fn get_rust_function(&self, key: RustFuncKey) -> ArenaResult<&RustFunction> {
        unsafe { ARENA.get_rust_function(key) }
    }

    pub fn get_dict(&self, key: DictKey) -> ArenaResult<&HashMap<ValueKind, ValueKind>> {
        unsafe { ARENA.get_dict(key) }
    }

    pub fn get_list(&self, key: ListKey) -> ArenaResult<&Vec<ValueKind>> {
        unsafe { ARENA.get_list(key) }
    }

    pub fn get_string(&self, key: StringKey) -> ArenaResult<&String> {
        unsafe { ARENA.get_string(key) }
    }

    pub fn get_function(&self, key: FuncKey) -> ArenaResult<&(String, Vec<String>, Node)> {
        unsafe { ARENA.get_function(key) }
    }

    pub fn free(&mut self, value: ValueKind) -> bool {
        unsafe { ARENA.free(value) }
    }

    // todo: make this print the right way around and do better than just a string
    // something like StackTraceEntry
    pub fn stack_trace(&self) -> String {
        if let Some(parent) = self.parent {
            format!("{} -> {}", self.name, unsafe {
                parent.as_ref().stack_trace()
            })
        } else {
            self.name.clone()
        }
    }
}
