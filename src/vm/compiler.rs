use crate::arenas::HeapValue;
use crate::ast::{Node, NodeKind};
use crate::error::WalrusError;
use crate::source_ref::SourceRef;
use crate::span::Span;
use crate::value::Value;
use crate::vm::instruction_set::InstructionSet;
use crate::vm::opcode::{Instruction, Opcode};
use crate::WalrusResult;

pub struct BytecodeEmitter<'a> {
    instructions: InstructionSet,
    source_ref: SourceRef<'a>,
}

impl<'a> BytecodeEmitter<'a> {
    pub fn new(source_ref: SourceRef<'a>) -> Self {
        Self {
            instructions: InstructionSet::new(),
            source_ref,
        }
    }

    pub fn emit(&mut self, node: Node) -> WalrusResult<()> {
        let kind = node.kind().to_string();
        let span = *node.span();

        match node.into_kind() {
            NodeKind::Program(nodes) => {
                for node in nodes {
                    self.emit(node)?;
                }
            }
            NodeKind::Int(value) => {
                let index = self.instructions.push_constant(Value::Int(value));
                self.instructions
                    .push(Instruction::new(Opcode::LoadConst(index), span));
            }
            NodeKind::Float(value) => {
                let index = self.instructions.push_constant(Value::Float(value));
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
                let value = self
                    .instructions
                    .get_heap_mut()
                    .push(HeapValue::String(&value));
                let index = self.instructions.push_constant(value);
                self.instructions
                    .push(Instruction::new(Opcode::LoadConst(index), span));
            }
            NodeKind::List(nodes) => {
                let cap = nodes.len();

                for node in nodes {
                    self.emit(node)?;
                }

                self.instructions
                    .push(Instruction::new(Opcode::List(cap), span));
            }
            NodeKind::Dict(nodes) => {
                let cap = nodes.len();

                for (key, value) in nodes {
                    self.emit(key)?;
                    self.emit(value)?;
                }

                self.instructions
                    .push(Instruction::new(Opcode::Dict(cap), span));
            }
            NodeKind::Range(left, right) => {
                if let Some(right) = right {
                    self.emit(*right)?;
                } else {
                    self.instructions.push(Instruction::new(Opcode::Void, span));
                }

                if let Some(left) = left {
                    self.emit(*left)?;
                } else {
                    self.instructions.push(Instruction::new(Opcode::Void, span));
                }

                self.instructions
                    .push(Instruction::new(Opcode::Range, span));
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
            NodeKind::If(cond, then, otherwise) => {
                self.emit(*cond)?;

                let jump = self.instructions.len();

                self.instructions
                    .push(Instruction::new(Opcode::JumpIfFalse(0), span));

                self.emit(*then)?;

                let jump_else = self.instructions.len();

                self.instructions
                    .push(Instruction::new(Opcode::Jump(0), span));

                self.instructions.set(
                    jump,
                    Instruction::new(Opcode::JumpIfFalse(self.instructions.len()), span),
                );

                if let Some(otherwise) = otherwise {
                    self.emit(*otherwise)?;
                }

                self.instructions.set(
                    jump_else,
                    // todo: check if span is right
                    Instruction::new(Opcode::Jump(self.instructions.len()), span),
                );
            }
            NodeKind::While(cond, body) => {
                let start = self.instructions.len();

                self.emit(*cond)?;

                let jump = self.instructions.len();

                self.instructions
                    .push(Instruction::new(Opcode::JumpIfFalse(0), span));

                self.emit(*body)?;

                self.instructions
                    .push(Instruction::new(Opcode::Jump(start), span));

                self.instructions.set(
                    jump,
                    Instruction::new(Opcode::JumpIfFalse(self.instructions.len()), span),
                );
            }
            NodeKind::For(name, iter, body) => {
                self.emit(*iter)?;

                self.instructions
                    .push(Instruction::new(Opcode::GetIter, span));

                let jump = self.instructions.len();

                self.instructions
                    .push(Instruction::new(Opcode::IterNext(0), span));

                let index = self.instructions.push_local(name);

                self.instructions
                    .push(Instruction::new(Opcode::StoreAt(index), span));

                self.emit(*body)?;

                self.instructions
                    .push(Instruction::new(Opcode::Jump(jump), span));

                self.instructions.set(
                    jump,
                    Instruction::new(Opcode::IterNext(self.instructions.len()), span),
                );
            }
            NodeKind::Index(node, index) => {
                self.emit(*node)?;
                self.emit(*index)?;

                self.instructions
                    .push(Instruction::new(Opcode::Index, span));
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
            NodeKind::Ident(name) => {
                match self.instructions.resolve_index(&name) {
                    Some(index) => {
                        self.instructions
                            .push(Instruction::new(Opcode::Load(index), span));
                    }
                    None => {
                        return Err(WalrusError::UndefinedVariable {
                            name,
                            span,
                            src: self.source_ref.source().to_string(),
                            filename: self.source_ref.filename().to_string(),
                        })
                    }
                };
            }
            NodeKind::Assign(name, node) => {
                if let Some(depth) = self.instructions.resolve_depth(&name) {
                    if depth >= self.instructions.local_depth() {
                        return Err(WalrusError::RedefinedLocal {
                            name,
                            span,
                            src: self.source_ref.source().to_string(),
                            filename: self.source_ref.filename().to_string(),
                        });
                    }
                }

                self.emit(*node)?;
                self.instructions.push_local(name);

                self.instructions
                    .push(Instruction::new(Opcode::Store, span));
            }
            NodeKind::Reassign(name, node, op) => {
                let index = match self.instructions.resolve_index(name.value()) {
                    Some(index) => index,
                    None => {
                        return Err(WalrusError::UndefinedVariable {
                            name: name.value().to_string(),
                            span,
                            src: self.source_ref.source().to_string(),
                            filename: self.source_ref.filename().to_string(),
                        })
                    }
                };

                match op {
                    Opcode::Add => {
                        self.instructions
                            .push(Instruction::new(Opcode::Load(index), span));

                        self.emit(*node)?;

                        self.instructions.push(Instruction::new(Opcode::Add, span));
                    }
                    Opcode::Subtract => {
                        self.instructions
                            .push(Instruction::new(Opcode::Load(index), span));

                        self.emit(*node)?;

                        self.instructions
                            .push(Instruction::new(Opcode::Subtract, span));
                    }
                    Opcode::Multiply => {
                        self.instructions
                            .push(Instruction::new(Opcode::Load(index), span));

                        self.emit(*node)?;

                        self.instructions
                            .push(Instruction::new(Opcode::Multiply, span));
                    }
                    Opcode::Divide => {
                        self.instructions
                            .push(Instruction::new(Opcode::Load(index), span));

                        self.emit(*node)?;

                        self.instructions
                            .push(Instruction::new(Opcode::Divide, span));
                    }
                    Opcode::Modulo => {
                        self.instructions
                            .push(Instruction::new(Opcode::Load(index), span));

                        self.emit(*node)?;

                        self.instructions
                            .push(Instruction::new(Opcode::Modulo, span));
                    }
                    _ => {
                        self.emit(*node)?;
                    }
                }

                self.instructions
                    .push(Instruction::new(Opcode::Reassign(index), span));
            }
            NodeKind::Statements(nodes) => {
                self.inc_depth();

                for node in nodes {
                    self.emit(node)?;
                }

                self.dec_depth(span);
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

    fn inc_depth(&mut self) {
        self.instructions.inc_depth();
    }

    fn dec_depth(&mut self, span: Span) {
        let popped = self.instructions.dec_depth();

        self.instructions
            .push(Instruction::new(Opcode::PopLocal(popped), span));
    }

    pub fn instruction_set(self) -> InstructionSet {
        // self.instructions.disassemble();
        self.instructions
    }
}
