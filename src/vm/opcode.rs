use std::fmt::{Display, Formatter};

use crate::span::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    LoadConst(usize), // usize pads the enum by 8 bytes
    Load(usize),
    List(usize),
    Dict(usize),
    Reassign(usize),
    JumpIfFalse(usize),
    Jump(usize),
    PopLocal(usize),
    IterNext(usize),
    StoreAt(usize),
    Call(usize),
    GetIter,
    Store,
    Range,
    True,
    False,
    Void,
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
    Equal,
    NotEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    And,
    Or,
    Print,
    Println,
    Index,
    Nop,
}

impl Display for Opcode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Opcode::Multiply => write!(f, "*"),
            Opcode::Divide => write!(f, "/"),
            Opcode::Add => write!(f, "+"),
            Opcode::Subtract => write!(f, "-"),
            Opcode::Modulo => write!(f, "%"),
            Opcode::Power => write!(f, "**"),
            Opcode::Not => write!(f, "not"),
            Opcode::Equal => write!(f, "=="),
            Opcode::NotEqual => write!(f, "!="),
            Opcode::Greater => write!(f, ">"),
            Opcode::GreaterEqual => write!(f, ">="),
            Opcode::Less => write!(f, "<"),
            Opcode::LessEqual => write!(f, "<="),
            Opcode::And => write!(f, "and"),
            Opcode::Or => write!(f, "or"),
            _ => write!(f, "{:?}", self),
        }
    }
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
