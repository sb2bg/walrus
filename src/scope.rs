use crate::value::Value;
use std::collections::{HashMap, VecDeque};

pub struct Scope<'a> {
    name: &'a str,
    // todo: add line and file name
    vars: HashMap<&'a str, Value<'a>>,
    parent: Option<&'a Scope<'a>>,
}

impl<'a> Scope<'a> {
    pub fn new() -> Self {
        Self {
            name: "global",
            vars: HashMap::new(),
            parent: None,
        }
    }

    pub fn new_child(&'a self, name: &'a str) -> Self {
        Self {
            name,
            vars: HashMap::new(),
            parent: Some(self),
        }
    }

    pub fn get(&self, name: &str) -> bool {
        if self.vars.contains_key(name) {
            true
        } else if let Some(parent) = self.parent {
            parent.get(name)
        } else {
            false
        }
    }

    pub fn set(&mut self, name: &'a str, value: Value<'a>) {
        self.vars.insert(name, value);
    }

    pub fn stack_trace(&self) -> String {
        let mut stack = VecDeque::new();
        let mut current = self;

        while let Some(parent) = &current.parent {
            stack.push_front(current.name.clone());
            current = parent;
        }

        stack.push_front(current.name.clone());
        stack.make_contiguous().join(" -> ")
    }
}
