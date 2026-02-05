use rustc_hash::FxHashMap;

#[derive(Debug, Default, Clone)]
pub struct SymbolTable {
    /// Ordered list of locals (index = local slot)
    locals: Vec<Local>,
    /// O(1) lookup: name -> (index, depth) for the most recent binding
    lookup: FxHashMap<String, (usize, usize)>,
    depth: usize,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self {
            locals: Vec::new(),
            lookup: FxHashMap::default(),
            depth: 0,
        }
    }

    pub fn push(&mut self, name: String) -> usize {
        let index = self.locals.len();
        let depth = self.depth;
        self.locals.push(Local::new(name.clone(), depth));
        self.lookup.insert(name, (index, depth));
        index
    }

    pub fn resolve_depth(&self, name: &str) -> Option<usize> {
        self.lookup.get(name).map(|&(_, depth)| depth)
    }

    pub fn resolve_index(&self, name: &str) -> Option<usize> {
        self.lookup.get(name).map(|&(index, _)| index)
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

        // Collect names to remove from lookup before modifying locals
        let names_to_remove: Vec<String> = self
            .locals
            .iter()
            .filter(|local| local.depth() > self.depth)
            .map(|local| local.name().to_string())
            .collect();

        // Remove from lookup
        for name in &names_to_remove {
            self.lookup.remove(name);
        }

        // Retain only locals at current depth or shallower
        self.locals.retain(|local| local.depth() <= self.depth);

        // Rebuild lookup for any shadowed variables that are now visible again
        // This handles the case where an inner scope shadowed an outer variable
        for (index, local) in self.locals.iter().enumerate() {
            self.lookup
                .insert(local.name().to_string(), (index, local.depth()));
        }

        len - self.locals.len()
    }

    pub fn len(&self) -> usize {
        self.locals.len()
    }

    /// Iterate over all locals
    pub fn iter_locals(&self) -> impl Iterator<Item = &Local> {
        self.locals.iter()
    }

    /// Get the name of a local at the given index
    pub fn get_name(&self, index: usize) -> Option<&str> {
        self.locals.get(index).map(|l| l.name())
    }

    /// Get all local names as a Vec (for debug info)
    pub fn get_all_names(&self) -> Vec<String> {
        self.locals.iter().map(|l| l.name().to_string()).collect()
    }

    /// Remove the last n locals from the symbol table
    pub fn pop_n(&mut self, n: usize) {
        let new_len = self.locals.len().saturating_sub(n);

        // Remove from lookup
        for local in self.locals.iter().skip(new_len) {
            self.lookup.remove(local.name());
        }

        self.locals.truncate(new_len);

        // Rebuild lookup for shadowed variables
        for (index, local) in self.locals.iter().enumerate() {
            self.lookup
                .insert(local.name().to_string(), (index, local.depth()));
        }
    }
}

#[derive(Debug, Clone)]
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
