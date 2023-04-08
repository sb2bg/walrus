use log::debug;

use crate::arenas::ValueHolder;
use crate::value::ValueKind;
use crate::vm::opcode::Instruction;
use crate::vm::symbol_table::SymbolTable;
use crate::WalrusResult;

#[derive(Default)]
pub struct InstructionSet {
    instructions: Vec<Instruction>,
    constants: Vec<ValueKind>,
    locals: SymbolTable,
    heap: ValueHolder,
}

impl InstructionSet {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            constants: Vec::new(),
            locals: SymbolTable::new(),
            heap: ValueHolder::new(),
        }
    }

    pub fn push(&mut self, instruction: Instruction) {
        self.instructions.push(instruction);
    }

    pub fn get(&self, index: usize) -> Instruction {
        self.instructions[index]
    }

    pub fn push_constant(&mut self, value: ValueKind) -> usize {
        self.constants.push(value);
        self.constants.len() - 1
    }

    pub fn get_constant(&self, index: usize) -> ValueKind {
        self.constants[index]
    }

    pub fn get_heap_mut(&mut self) -> &mut ValueHolder {
        &mut self.heap
    }

    pub fn get_heap(&self) -> &ValueHolder {
        &self.heap
    }

    pub fn push_local(&mut self, name: String) {
        self.locals.push(name);
    }

    pub fn resolve_index(&self, name: &str) -> Option<usize> {
        self.locals.resolve_index(name)
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

    pub fn disassemble(&self) {
        if !log::log_enabled!(log::Level::Debug) {
            return;
        }

        debug!("| == disassemble ==");
        debug!("| sizeof(instructions) = {}", self.instructions.len());

        for (i, _) in self.instructions.iter().enumerate() {
            self.disassemble_single(i);
        }
    }

    pub fn stringify(&self, value: ValueKind) -> WalrusResult<String> {
        self.heap.stringify(value)
    }

    pub fn disassemble_single(&self, index: usize) {
        debug!("| {index:03} {:?}", self.instructions[index].opcode(),);
    }
}
