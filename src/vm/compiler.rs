use crate::ast::{Node, NodeKind};
use crate::scope::Scope;
use crate::value::{HeapValue, ValueKind};
use crate::vm::instruction_set::InstructionSet;
use crate::vm::opcode::{Instruction, Opcode};

pub struct BytecodeEmitter {
    instructions: InstructionSet,
}

impl BytecodeEmitter {
    pub fn new() -> Self {
        Self {
            instructions: InstructionSet::new(),
        }
    }

    pub fn emit(&mut self, node: Node) {
        let span = *node.span();

        match node.into_kind() {
            NodeKind::Int(value) => {
                let index = self.instructions.push_constant(ValueKind::Int(value));
                self.instructions
                    .push(Instruction::new(Opcode::LoadConst(index), span));
            }
            NodeKind::Float(value) => {
                let index = self.instructions.push_constant(ValueKind::Float(value));
                self.instructions
                    .push(Instruction::new(Opcode::LoadConst(index), span));
            }
            NodeKind::String(value) => {
                let index = self
                    .instructions
                    .push_constant(Scope::heap_alloc(HeapValue::String(value)));
                self.instructions
                    .push(Instruction::new(Opcode::LoadConst(index), span));
            }
            NodeKind::List(nodes) => {
                for node in nodes {
                    self.emit(*node);
                }

                let index = self
                    .instructions
                    .push_constant(Scope::heap_alloc(HeapValue::List(Vec::new())));
                self.instructions
                    .push(Instruction::new(Opcode::LoadConst(index), span));
            }
            NodeKind::Void => {
                self.instructions
                    .push(Instruction::new(Opcode::LoadConst(0), span));
            }
            NodeKind::BinOp(left, op, right) => {
                self.emit(*left);
                self.emit(*right);

                match op {
                    crate::ast::Op::Add => {
                        self.instructions.push(Instruction::new(Opcode::Add, span));
                    }
                    crate::ast::Op::Sub => {
                        self.instructions
                            .push(Instruction::new(Opcode::Subtract, span));
                    }
                    crate::ast::Op::Mul => {
                        self.instructions
                            .push(Instruction::new(Opcode::Multiply, span));
                    }
                    crate::ast::Op::Div => {
                        self.instructions
                            .push(Instruction::new(Opcode::Divide, span));
                    }
                    crate::ast::Op::Mod => {
                        self.instructions
                            .push(Instruction::new(Opcode::Modulo, span));
                    }
                    crate::ast::Op::Pow => {
                        self.instructions
                            .push(Instruction::new(Opcode::Power, span));
                    }
                    _ => unimplemented!(),
                }
            }
            NodeKind::Println(node) => {
                self.emit(*node);

                self.instructions
                    .push(Instruction::new(Opcode::Println, span));
            }
            NodeKind::Print(node) => {
                self.emit(*node);

                self.instructions
                    .push(Instruction::new(Opcode::Print, span));
            }
            NodeKind::Statements(nodes) => {
                for node in nodes {
                    self.emit(*node);
                }
            }
            NodeKind::Return(node) => {
                self.emit(*node);

                self.instructions
                    .push(Instruction::new(Opcode::Return, span));
            }
            _ => unimplemented!(),
        }
    }

    pub fn instruction_set(self) -> InstructionSet {
        self.instructions
    }
}
