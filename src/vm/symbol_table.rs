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
        self.locals.push(Local::new(name, self.depth, discriminant));
        self.locals.len() - 1
    }

    pub fn resolve(&self, name: &str) -> Option<usize> {
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

    pub fn dec_depth(&mut self) {
        self.depth -= 1;
        self.locals.retain(|local| local.depth() <= self.depth);
    }

    pub fn clear(&mut self) {
        self.locals.clear();
    }

    pub fn len(&self) -> usize {
        self.locals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.locals.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<Local> {
        self.locals.iter()
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
