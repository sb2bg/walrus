use std::fmt::Display;

use log::debug;

use crate::WalrusResult;
use crate::arenas::ValueHolder;
use crate::value::Value;
use crate::vm::opcode::Instruction;
use crate::vm::symbol_table::SymbolTable;

/// Metadata about a loop for JIT compilation
#[derive(Debug, Clone)]
pub struct LoopMetadata {
    /// IP of the loop header (where iteration check happens)
    pub header_ip: usize,
    /// IP of the backward jump (end of loop body)
    pub back_edge_ip: usize,
    /// IP to jump to when loop exits
    pub exit_ip: usize,
    /// True if this is an optimized range-based loop
    pub is_range_loop: bool,
}

/// Metadata about a function for JIT compilation
#[derive(Debug, Clone)]
pub struct FunctionMetadata {
    /// Function name
    pub name: String,
    /// IP where function body starts
    pub start_ip: usize,
    /// Number of parameters
    pub arity: usize,
}

#[derive(Default, Debug, Clone)]
pub struct InstructionSet {
    // TODO: name these so we can have better disassembly output
    pub instructions: Vec<Instruction>,
    pub constants: Vec<Value>,
    pub locals: SymbolTable,
    pub globals: SymbolTable,
    // Note: heap is stored globally in ARENA, not per InstructionSet

    // JIT metadata (Phase 1)
    /// Detected loops for hot-spot tracking
    pub loops: Vec<LoopMetadata>,
    /// Registered functions for call tracking
    pub functions: Vec<FunctionMetadata>,
}

impl InstructionSet {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            constants: Vec::new(),
            locals: SymbolTable::new(),
            globals: SymbolTable::new(),
            loops: Vec::new(),
            functions: Vec::new(),
        }
    }

    pub fn new_with(locals: SymbolTable) -> Self {
        Self {
            instructions: Vec::new(),
            constants: Vec::new(),
            locals,
            globals: SymbolTable::new(),
            loops: Vec::new(),
            functions: Vec::new(),
        }
    }

    pub fn new_child_with_globals(globals: SymbolTable) -> Self {
        Self {
            instructions: Vec::new(),
            constants: Vec::new(),
            locals: SymbolTable::new(),
            globals,
            loops: Vec::new(),
            functions: Vec::new(),
        }
    }

    /// Register a loop for JIT hot-spot detection
    pub fn register_loop(&mut self, header_ip: usize, back_edge_ip: usize, exit_ip: usize, is_range_loop: bool) {
        self.loops.push(LoopMetadata {
            header_ip,
            back_edge_ip,
            exit_ip,
            is_range_loop,
        });
    }

    /// Register a function for JIT hot-spot detection
    pub fn register_function(&mut self, name: String, start_ip: usize, arity: usize) {
        self.functions.push(FunctionMetadata {
            name,
            start_ip,
            arity,
        });
    }

    pub fn push(&mut self, instruction: Instruction) {
        self.instructions.push(instruction);
    }

    pub fn get(&self, index: usize) -> Instruction {
        self.instructions[index]
    }

    pub fn push_constant(&mut self, value: Value) -> u32 {
        self.constants.push(value);
        (self.constants.len() - 1) as u32
    }

    pub fn get_constant(&self, index: u32) -> Value {
        self.constants[index as usize]
    }

    pub fn get_heap_mut(&mut self) -> &mut ValueHolder {
        unsafe { &mut *crate::arenas::get_arena_ptr() }
    }

    pub fn get_heap(&self) -> &ValueHolder {
        unsafe { &*crate::arenas::get_arena_ptr() }
    }

    pub fn push_local(&mut self, name: String) -> u32 {
        self.locals.push(name) as u32
    }

    pub fn push_global(&mut self, name: String) -> u32 {
        self.globals.push(name) as u32
    }

    pub fn local_len(&self) -> usize {
        self.locals.len()
    }

    pub fn resolve_index(&self, name: &str) -> Option<usize> {
        self.locals.resolve_index(name)
    }

    pub fn resolve_local_index(&self, name: &str) -> Option<usize> {
        self.locals.resolve_index(name)
    }

    pub fn resolve_global_index(&self, name: &str) -> Option<usize> {
        self.globals.resolve_index(name)
    }

    pub fn resolve_depth(&self, name: &str) -> Option<usize> {
        self.locals.resolve_depth(name)
    }

    pub fn local_depth(&self) -> usize {
        self.locals.depth()
    }

    pub fn inc_depth(&mut self) {
        self.locals.inc_depth();
    }

    pub fn dec_depth(&mut self) -> usize {
        self.locals.dec_depth()
    }

    /// Remove the last n locals from the symbol table (for optimized for-range loops)
    pub fn pop_locals(&mut self, n: usize) {
        self.locals.pop_n(n);
    }

    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }

    pub fn set(&mut self, index: usize, instruction: Instruction) {
        self.instructions[index] = instruction;
    }

    pub fn disassemble(&self) {
        debug!("{}", self);
    }

    pub fn stringify(&self, value: Value) -> WalrusResult<String> {
        crate::arenas::with_arena(|arena| arena.stringify(value))
    }

    pub fn disassemble_single(&self, index: usize, title: &str) {
        debug!(
            "| {} | {index:03} {:?}",
            title,
            self.instructions[index].opcode(),
        );
    }
}

impl Display for InstructionSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "| ===  disassembly  ===")?;
        writeln!(f, "| === constant pool ===")?;

        for (i, _) in self.constants.iter().enumerate() {
            writeln!(f, "| C{i:02} {:?}", self.constants[i],)?;
        }

        writeln!(f)?;
        writeln!(f, "| ===  instructions  ===")?;

        for (i, _) in self.instructions.iter().enumerate() {
            writeln!(f, "| {i:03} {:?}", self.instructions[i].opcode(),)?;
        }

        Ok(())
    }
}
