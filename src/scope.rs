use crate::arenas::ValueHolder;
use crate::value::Value;
use log::debug;
use std::collections::{HashMap, VecDeque};

#[derive(Debug)]
pub struct Scope<'a> {
    name: String,
    // todo: add line and file name
    vars: HashMap<String, Value>,
    parent: Option<&'a Scope<'a>>,
}

impl<'a> Scope<'a> {
    pub fn new(arena: &mut ValueHolder) -> Self {
        Self {
            name: "global".to_string(),
            vars: Self::global_vars(arena),
            parent: None,
        }
    }

    pub fn dump(&self) {
        debug!("Scope dump: {}", self.name);

        for (k, v) in self.vars.iter() {
            debug!("{}: {}", k, v);
        }

        if let Some(parent) = self.parent {
            parent.dump();
        }
    }

    // todo: I want this to be behind an import but I'm not sure how to do that currently
    fn global_vars(arena: &mut ValueHolder) -> HashMap<String, Value> {
        let mut vars = HashMap::new();

        vars.insert(
            "print".to_string(),
            arena.insert_rust_function(|args| {
                println!("{:?}", args); // todo: make this print what it should
                Value::Void
            }),
        );

        vars
    }

    pub fn new_child(&'a self, name: String) -> Self {
        Self {
            name,
            vars: HashMap::new(),
            parent: Some(self),
        }
    }

    pub fn get(&self, name: &str) -> Option<Value> {
        // todo: should this be the behavior? maybe be more explicit
        if name == "_" {
            return Some(Value::Void);
        }

        if let Some(value) = self.vars.get(name) {
            Some(*value)
        } else {
            match self.parent {
                Some(parent) => parent.get(name),
                _ => None,
            }
        }
    }

    pub fn define(&mut self, name: String, value: Value) {
        if name == "_" {
            return;
        }

        self.vars.insert(name, value);
    }

    pub fn reassign(&mut self, name: String, value: Value) -> bool {
        if self.vars.contains_key(&name) {
            self.define(name, value);
            true
        } else {
            match self.parent {
                // fixme: mut borrow of self.parent is not allowed
                // Some(parent) => parent.reassign(name, value),
                _ => false,
            }
        }
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
