use crate::span::Span;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    Constant(usize),
    Return,
    Pop,
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Power,
    Negate,
    Not,
    Print,
    Println,
}

#[derive(Debug, Clone, Copy)]
pub struct Instruction {
    opcode: Opcode,
    span: Span,
}

impl Instruction {
    pub fn new(opcode: Opcode, span: Span) -> Self {
        Self { opcode, span }
    }

    pub fn opcode(&self) -> Opcode {
        self.opcode
    }

    pub fn span(&self) -> Span {
        self.span
    }
}
