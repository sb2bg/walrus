use std::fmt::{Display, Formatter};

use crate::span::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    // Memory-optimized: u32 instead of usize reduces enum size from 16 to 12 bytes
    LoadConst(u32),
    Load(u32),
    LoadGlobal(u32),
    List(u32),
    Dict(u32),
    Reassign(u32),
    ReassignGlobal(u32),
    JumpIfFalse(u32),
    Jump(u32),
    PopLocal(u32),
    IterNext(u32),
    StoreAt(u32),
    StoreGlobal(u32),
    Call(u32),
    TailCall(u32), // Tail call optimization: reuses current frame instead of pushing new one

    // Specialized constant loading (zero-operand for efficiency)
    LoadConst0,  // Load constant at index 0
    LoadConst1,  // Load constant at index 1
    LoadLocal0,  // Load local at index 0
    LoadLocal1,  // Load local at index 1
    LoadLocal2,  // Load local at index 2
    LoadLocal3,  // Load local at index 3
    LoadGlobal0, // Load global at index 0
    LoadGlobal1, // Load global at index 1
    LoadGlobal2, // Load global at index 2
    LoadGlobal3, // Load global at index 3

    // Stack manipulation
    Dup,  // Duplicate top of stack
    Swap, // Swap top two stack values
    Pop2, // Pop two values
    Pop3, // Pop three values

    // Zero-operand opcodes
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
    StoreIndex, // Store value at index (for list[i] = x and dict[k] = v)
    Nop,
    MakeStruct, // Create a struct instance from a struct definition
    GetMethod,  // Get a method from a struct definition
    CallMethod, // Call a method on a struct instance

    // Builtins
    Len,  // Get length of a string, list, or dict
    Str,  // Convert value to string
    Type, // Get type name of a value
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
