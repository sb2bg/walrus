use crate::value::ValueKind;
use crate::vm::opcode::Instruction;
use log::debug;

#[derive(Default)]
pub struct InstructionSet {
    instructions: Vec<Instruction>,
    constants: Vec<ValueKind>,
}

impl InstructionSet {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            constants: Vec::new(),
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

    pub fn disassemble_single(&self, index: usize) {
        debug!("| {index:03} {:?}", self.instructions[index].opcode(),);
    }
}
