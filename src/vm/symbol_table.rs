#[derive(Debug, Default)]
pub struct SymbolTable {
    locals: Vec<Local>, // fixme: use a hashmap
    depth: usize,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self {
            locals: Vec::new(),
            depth: 0,
        }
    }

    pub fn push(&mut self, name: String) -> usize {
        self.locals.push(Local::new(name, self.depth));
        self.locals.len() - 1
    }

    pub fn resolve_depth(&self, name: &str) -> Option<usize> {
        self.locals
            .iter()
            .rev()
            .find(|local| local.name() == name)
            .map(|local| local.depth())
    }

    pub fn resolve_index(&self, name: &str) -> Option<usize> {
        self.locals
            .iter()
            .rev()
            .position(|local| local.name() == name)
            .map(|index| self.locals.len() - index - 1)
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn inc_depth(&mut self) {
        self.depth += 1;
    }

    pub fn dec_depth(&mut self) -> usize {
        self.depth -= 1;
        let len = self.locals.len();
        self.locals.retain(|local| local.depth() <= self.depth);
        len - self.locals.len()
    }

    pub fn len(&self) -> usize {
        self.locals.len()
    }
}

#[derive(Debug)]
pub struct Local {
    name: String,
    depth: usize,
}

impl Local {
    pub fn new(name: String, depth: usize) -> Self {
        Self { name, depth }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn depth(&self) -> usize {
        self.depth
    }
}
