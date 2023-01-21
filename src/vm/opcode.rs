use crate::span::Span;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    Constant,
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
    Nop,
}

#[derive(Debug, Clone, Copy)]
pub enum OpcodeOrByte {
    Opcode(Opcode),
    Byte(u8),
}

#[derive(Debug, Clone, Copy)]
pub struct Instruction {
    data: OpcodeOrByte,
    span: Span,
}

impl Instruction {
    pub fn new_opcode(opcode: Opcode, span: Span) -> Self {
        Self {
            data: OpcodeOrByte::Opcode(opcode),
            span,
        }
    }

    pub fn new_byte(byte: u8, span: Span) -> Self {
        Self {
            data: OpcodeOrByte::Byte(byte),
            span,
        }
    }

    pub fn opcode(&self) -> Opcode {
        let OpcodeOrByte::Opcode(opcode) = self.data else {
            todo!("Expected opcode, got byte");
        };

        opcode
    }

    pub fn byte(&self) -> u8 {
        let OpcodeOrByte::Byte(byte) = self.data else {
            todo!("Expected byte, got opcode");
        };

        byte
    }

    pub fn span(&self) -> Span {
        self.span
    }
}
