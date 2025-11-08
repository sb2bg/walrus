use std::fmt::Display;

use log::debug;

use crate::WalrusResult;
use crate::arenas::ValueHolder;
use crate::value::Value;
use crate::vm::opcode::Instruction;
use crate::vm::symbol_table::SymbolTable;

#[derive(Default, Debug, Clone)]
pub struct InstructionSet {
    pub instructions: Vec<Instruction>,
    pub constants: Vec<Value>,
    pub locals: SymbolTable,
    pub globals: SymbolTable,
    // Note: heap is stored globally in ARENA, not per InstructionSet
}

impl InstructionSet {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            constants: Vec::new(),
            locals: SymbolTable::new(),
            globals: SymbolTable::new(),
        }
    }

    pub fn new_with(locals: SymbolTable) -> Self {
        Self {
            instructions: Vec::new(),
            constants: Vec::new(),
            locals,
            globals: SymbolTable::new(),
        }
    }

    pub fn new_child_with_globals(globals: SymbolTable) -> Self {
        Self {
            instructions: Vec::new(),
            constants: Vec::new(),
            locals: SymbolTable::new(),
            globals,
        }
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
        unsafe { &mut *std::ptr::addr_of_mut!(crate::arenas::ARENA) }
    }

    pub fn get_heap(&self) -> &ValueHolder {
        unsafe { &*std::ptr::addr_of!(crate::arenas::ARENA) }
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
        unsafe { (&*std::ptr::addr_of!(crate::arenas::ARENA)).stringify(value) }
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
