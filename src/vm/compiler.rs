use crate::arenas::HeapValue;
use crate::ast::{Node, NodeKind};
use crate::value::ValueKind;
use crate::vm::instruction_set::InstructionSet;
use crate::vm::opcode::{Instruction, Opcode};
use crate::WalrusResult;
use rustc_hash::FxHashMap;

pub struct BytecodeEmitter {
    instructions: InstructionSet,
}

impl BytecodeEmitter {
    pub fn new() -> Self {
        Self {
            instructions: InstructionSet::new(),
        }
    }

    pub fn emit(&mut self, node: Node) -> WalrusResult<()> {
        let kind = node.kind().to_string();
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
            NodeKind::Bool(value) => {
                self.instructions.push(Instruction::new(
                    if value { Opcode::True } else { Opcode::False },
                    span,
                ));
            }
            NodeKind::String(value) => {
                // fixme: I need a new ValueHolder that stores strings instead of interning them
                // we should use a variant of the below code: but for now we do something else
                // let index = self.instructions.push_constant(ValueKind::String(value));
                // self.instructions
                //     .push(Instruction::new(Opcode::LoadConst(index), span));

                let index = self
                    .instructions
                    .push_constant(HeapValue::String(value).alloc());
                self.instructions
                    .push(Instruction::new(Opcode::LoadConst(index), span));
            }
            NodeKind::List(nodes) => {
                let cap = nodes.len();

                for node in nodes {
                    self.emit(*node)?;
                }

                let index = self
                    .instructions
                    .push_constant(HeapValue::List(Vec::with_capacity(cap)).alloc());
                self.instructions
                    .push(Instruction::new(Opcode::LoadConst(index), span));
            }
            NodeKind::Dict(nodes) => {
                let cap = nodes.len();

                for (key, value) in nodes {
                    self.emit(*key)?;
                    self.emit(*value)?;
                }

                let index = self.instructions.push_constant(
                    HeapValue::Dict(FxHashMap::with_capacity_and_hasher(cap, Default::default()))
                        .alloc(),
                );
                self.instructions
                    .push(Instruction::new(Opcode::LoadConst(index), span));
            }
            NodeKind::Void => {
                self.instructions.push(Instruction::new(Opcode::Void, span));
            }
            NodeKind::BinOp(left, op, right) => {
                self.emit(*left)?;
                self.emit(*right)?;

                self.instructions.push(Instruction::new(op, span));
            }
            NodeKind::UnaryOp(op, node) => {
                self.emit(*node)?;
                self.instructions.push(Instruction::new(op, span));
            }
            NodeKind::Println(node) => {
                self.emit(*node)?;

                self.instructions
                    .push(Instruction::new(Opcode::Println, span));
            }
            NodeKind::Print(node) => {
                self.emit(*node)?;

                self.instructions
                    .push(Instruction::new(Opcode::Print, span));
            }
            NodeKind::Statements(nodes) => {
                for node in nodes {
                    self.emit(*node)?;
                }
            }
            NodeKind::Return(node) => {
                self.emit(*node)?;

                self.instructions
                    .push(Instruction::new(Opcode::Return, span));
            }
            _ => unimplemented!("{}", kind),
        }

        Ok(())
    }

    pub fn instruction_set(self) -> InstructionSet {
        self.instructions
    }
}
