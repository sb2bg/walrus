use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct SymbolTable {
    locals: HashMap<String, Vec<Local>>,
    depth: usize,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self {
            locals: HashMap::new(),
            depth: 0,
        }
    }

    pub fn push(&mut self, name: String) -> usize {
        let local = Local::new(self.depth);
        self.locals.entry(name).or_default().push(local);
        self.locals.len() - 1
    }

    pub fn resolve_index(&self, name: &str) -> Option<usize> {
        self.locals.get(name).and_then(|locals| {
            locals
                .iter()
                .rev()
                .find(|local| local.depth <= self.depth)
                .map(|local| self.locals.len() - local.index - 1)
        })
    }

    pub fn resolve_depth(&self, name: &str) -> Option<usize> {
        self.locals.get(name).and_then(|locals| {
            locals
                .iter()
                .rev()
                .find(|local| local.depth <= self.depth)
                .map(|local| local.depth)
        })
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn inc_depth(&mut self) {
        self.depth += 1;
    }

    pub fn dec_depth(&mut self) {
        self.depth -= 1;
        self.locals.retain(|_, locals| {
            locals.retain(|local| local.depth <= self.depth);
            !locals.is_empty()
        });
    }

    pub fn clear(&mut self) {
        self.locals.clear();
    }

    pub fn len(&self) -> usize {
        self.locals.values().map(Vec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.locals.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Local> {
        self.locals.values().flatten()
    }
}

#[derive(Debug)]
pub struct Local {
    index: usize,
    depth: usize,
}

impl Local {
    pub fn new(depth: usize) -> Self {
        Self {
            index: depth,
            depth,
        }
    }
}
