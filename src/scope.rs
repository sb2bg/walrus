use crate::value::Value;
use std::collections::{HashMap, VecDeque};

#[derive(Debug)]
pub struct Scope<'a> {
    name: String,
    // todo: add line and file name
    vars: HashMap<String, Value>,
    parent: Option<&'a Scope<'a>>,
}

impl<'a> Scope<'a> {
    pub fn new() -> Self {
        Self {
            name: "global".into(),
            vars: HashMap::new(),
            parent: None,
        }
    }

    pub fn new_child(&'a self, name: String) -> Self {
        Self {
            name,
            vars: HashMap::new(),
            parent: Some(self),
        }
    }

    pub fn get(&self, name: &str) -> Option<Value> {
        if let Some(value) = self.vars.get(name) {
            // instead of cloning, we could return a reference but then
            // we have to handle references in the interpreter and not
            // owned values, and also this would make every value (in the lang)
            // mutable, which i dont think we want so we should clone
            Some(value.clone())
        } else {
            match self.parent {
                Some(parent) => parent.get(name),
                _ => None,
            }
        }
    }

    pub fn define(&mut self, name: String, value: Value) {
        self.vars.insert(name, value);
    }

    pub fn stack_trace(&self) -> String {
        let mut stack = VecDeque::new();
        let mut current = self;

        while let Some(parent) = current.parent {
            stack.push_front(current.name.clone());
            current = parent;
        }

        stack.push_front(current.name.clone());
        stack.make_contiguous().join(" -> ")
    }
}
