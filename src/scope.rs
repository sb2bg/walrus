use std::collections::{HashMap, VecDeque};

#[derive(Clone)]
pub struct Scope {
    name: String,
    // todo: add line and file name
    vars: HashMap<String, ()>,
    parent: Option<Box<Self>>,
}

impl Scope {
    pub fn new() -> Self {
        Self {
            name: "global".into(),
            vars: HashMap::new(),
            parent: None,
        }
    }

    pub fn new_child(&self, name: String) -> Self {
        Self {
            name,
            vars: HashMap::new(),
            parent: Some(Box::new(self.clone())),
        }
    }

    pub fn get(&self, name: &str) -> bool {
        if self.vars.contains_key(name) {
            true
        } else if let Some(parent) = &self.parent {
            parent.get(name)
        } else {
            false
        }
    }

    pub fn set(&mut self, name: &str) {
        self.vars.insert(name.into(), ());
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
