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
    AddAssignLocal(u32),
    AddAssignGlobal(u32),
    AddAssignLocalInt(u32),
    AddAssignGlobalInt(u32),
    JumpIfFalse(u32),
    Jump(u32),
    PopLocal(u32),
    IterNext(u32),
    StoreAt(u32),
    StoreGlobal(u32),
    IndexConst(u32),
    IndexLocal(u32),
    IndexLocalConst(u32, u32),
    StoreIndexLocal(u32),
    Call(u32),
    CallGlobal1(u32),          // Statically known global call with one argument
    CallGlobal(u32, u32),      // (global index, arg count) for statically known global callees
    TailCall(u32), // Tail call optimization: reuses current frame instead of pushing new one
    PushExceptionHandler(u32), // Catch target IP
    PopExceptionHandler,

    // Specialized constant loading (zero-operand for efficiency)
    LoadConst0,      // Load constant at index 0
    LoadConst1,      // Load constant at index 1
    LoadLocal0,      // Load local at index 0
    LoadLocal1,      // Load local at index 1
    LoadLocal2,      // Load local at index 2
    LoadLocal3,      // Load local at index 3
    LoadLocal4,      // Load local at index 4
    LoadLocal5,      // Load local at index 5
    LoadLocal6,      // Load local at index 6
    LoadLocal7,      // Load local at index 7
    LoadLocal8,      // Load local at index 8
    LoadLocal9,      // Load local at index 9
    LoadLocal10,     // Load local at index 10
    LoadLocal11,     // Load local at index 11
    StoreLocal0,     // Store/pop into local 0, appending if needed
    StoreLocal1,     // Store/pop into local 1, appending if needed
    StoreLocal2,     // Store/pop into local 2, appending if needed
    StoreLocal3,     // Store/pop into local 3, appending if needed
    StoreLocal4,     // Store/pop into local 4, appending if needed
    StoreLocal5,     // Store/pop into local 5, appending if needed
    StoreLocal6,     // Store/pop into local 6, appending if needed
    StoreLocal7,     // Store/pop into local 7, appending if needed
    StoreLocal8,     // Store/pop into local 8, appending if needed
    StoreLocal9,     // Store/pop into local 9, appending if needed
    StoreLocal10,    // Store/pop into local 10, appending if needed
    StoreLocal11,    // Store/pop into local 11, appending if needed
    ReassignLocal0,  // Reassign existing local 0
    ReassignLocal1,  // Reassign existing local 1
    ReassignLocal2,  // Reassign existing local 2
    ReassignLocal3,  // Reassign existing local 3
    ReassignLocal4,  // Reassign existing local 4
    ReassignLocal5,  // Reassign existing local 5
    ReassignLocal6,  // Reassign existing local 6
    ReassignLocal7,  // Reassign existing local 7
    ReassignLocal8,  // Reassign existing local 8
    ReassignLocal9,  // Reassign existing local 9
    ReassignLocal10, // Reassign existing local 10
    ReassignLocal11, // Reassign existing local 11
    LoadGlobal0,     // Load global at index 0
    LoadGlobal1,     // Load global at index 1
    LoadGlobal2,     // Load global at index 2
    LoadGlobal3,     // Load global at index 3

    // Specialized arithmetic (hot-path optimizations)
    IncrementLocal(u32), // local[idx] += 1 (extremely common in loops)
    DecrementLocal(u32), // local[idx] -= 1
    AddInt,              // Integer addition (no type checking)
    AddInt1,             // Integer addition by 1
    SubtractInt,         // Integer subtraction (no type checking)
    SubtractInt1,        // Integer subtraction by 1
    SubtractInt2,        // Integer subtraction by 2
    MultiplyInt,         // Integer multiplication (no type checking)
    DivideInt,           // Integer division (no type checking)
    ModuloInt,           // Integer modulo (no type checking)
    LessInt,             // Integer less-than comparison
    LessEqualInt,        // Integer less-equal comparison
    LessEqualInt1,       // Integer less-equal comparison against 1

    // Optimized range loops (avoids heap allocation for iterators)
    // ForRangeInit: pops end, start from stack, stores in locals[idx] and locals[idx+1]
    // ForRangeNext(jump_target, idx): if locals[idx] < locals[idx+1], push locals[idx]++, else jump
    ForRangeInit(u32), // Initialize range loop: (local_idx for current, end stored at idx+1)
    ForRangeNext(u32, u32), // (jump_if_done, local_idx) - increment and check
    ForRangeNextDiscard(u32, u32), // Like ForRangeNext but discards the loop variable value

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
    Await,
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
    Throw,
    Index,
    StoreIndex, // Store value at index (for list[i] = x and dict[k] = v)
    Nop,
    ListPush,        // Push value into a list and return void
    MakeStruct,      // Create a struct instance from a struct definition
    GetMethod,       // Get a method from a struct definition
    CallMethod(u32), // Call a method on any value (arg count as operand, method name + object + args on stack)

    // Runtime helpers
    Str,    // Convert top-of-stack value to string (used by f-strings)
    Import, // (module_name) -> module value
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
