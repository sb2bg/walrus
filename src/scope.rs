use crate::arenas::ValueHolder;
use crate::value::ValueKind;
use log::debug;
use std::collections::{HashMap, VecDeque};

#[derive(Debug)]
pub struct Scope<'a> {
    name: String,
    // todo: add line and file name
    vars: HashMap<String, ValueKind>,
    arena: ValueHolder,
    parent: Option<&'a Scope<'a>>,
}

impl<'a> Scope<'a> {
    pub fn new() -> Self {
        Self {
            name: "global".to_string(),
            vars: HashMap::new(),
            parent: None,
            arena: ValueHolder::new(),
        }
    }

    pub fn dump(&self) {
        self.arena.dump();

        debug!("Scope dump: {}", self.name);

        for (k, v) in self.vars.iter() {
            debug!("{}: {}", k, v);
        }

        if let Some(parent) = self.parent {
            parent.dump();
        }
    }

    pub fn mut_arena(&mut self) -> &mut ValueHolder {
        &mut self.arena
    }

    pub fn arena(&self) -> &ValueHolder {
        &self.arena
    }

    pub fn new_child(&'a self, name: String) -> Self {
        Self {
            name,
            vars: HashMap::new(),
            parent: Some(self),
            arena: self.arena.clone(), // fixme: eventually we just want to borrow the arena
        }
    }

    pub fn get(&self, name: &str) -> Option<ValueKind> {
        // todo: should this be the behavior? maybe be more explicit
        if name == "_" {
            return Some(ValueKind::Void);
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

    pub fn define(&mut self, name: String, value: ValueKind) {
        if name == "_" {
            return;
        }

        self.vars.insert(name, value);
    }

    pub fn reassign(&mut self, name: String, value: ValueKind) -> bool {
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
