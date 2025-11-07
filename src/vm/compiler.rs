use crate::WalrusResult;
use crate::arenas::HeapValue;
use crate::ast::{Node, NodeKind};
use crate::error::WalrusError;
use crate::function::{VmFunction, WalrusFunction};
use crate::source_ref::SourceRef;
use crate::span::Span;
use crate::value::Value;
use crate::vm::instruction_set::InstructionSet;
use crate::vm::opcode::{Instruction, Opcode};

pub struct BytecodeEmitter<'a> {
    instructions: InstructionSet,
    source_ref: SourceRef<'a>,
    depth: usize, // 0 = global scope, >0 = local scope
}

impl<'a> BytecodeEmitter<'a> {
    pub fn new(source_ref: SourceRef<'a>) -> Self {
        Self {
            instructions: InstructionSet::new(),
            source_ref,
            depth: 0, // Start at global scope
        }
    }

    fn new_child(&self) -> Self {
        Self {
            instructions: InstructionSet::new_child_with_globals(self.instructions.globals.clone()),
            source_ref: self.source_ref,
            depth: 1, // Functions start at local scope depth 1
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

                self.inc_depth();

                let index = self.instructions.push_local(name);

                self.instructions
                    .push(Instruction::new(Opcode::StoreAt(index), span));

                self.emit(*body)?;

                self.dec_depth_no_pop();

                self.instructions
                    .push(Instruction::new(Opcode::Jump(jump), span));

                self.instructions.set(
                    jump,
                    Instruction::new(Opcode::IterNext(self.instructions.len()), span),
                );
            }
            NodeKind::FunctionDefinition(name, args, body) => {
                let is_global = self.depth == 0;

                // Pre-register the function name
                let func_index = if is_global {
                    self.instructions.push_global(name.clone())
                } else {
                    self.instructions.push_local(name.clone())
                };

                // Create a child emitter - functions are always local scope
                let mut emitter = self.new_child();
                let arg_len = args.len();

                // Define function parameters as locals in the function scope
                // Don't emit Store opcodes - values will already be in locals when called
                for arg in args {
                    emitter.define_parameter(arg);
                }

                emitter.emit(*body)?;

                // Create the function heap value
                let func =
                    self.instructions
                        .get_heap_mut()
                        .push(HeapValue::Function(WalrusFunction::Vm(VmFunction::new(
                            name.clone(),
                            arg_len,
                            emitter.instruction_set(),
                        ))));

                // Load the function constant and store it
                let index = self.instructions.push_constant(func);
                self.instructions
                    .push(Instruction::new(Opcode::LoadConst(index), span));

                if is_global {
                    // Top-level function definitions go into globals
                    self.instructions
                        .push(Instruction::new(Opcode::StoreGlobal(func_index), span));
                } else {
                    // Nested function definitions go into locals
                    self.instructions
                        .push(Instruction::new(Opcode::StoreAt(func_index), span));
                }
            }
            NodeKind::AnonFunctionDefinition(args, body) => {
                // TODO: model this after FunctionDefinition
            }
            NodeKind::FunctionCall(func, args) => {
                // TODO: try to check arity at compile time, if possible
                // we may need to evaluate func first, which is a Box<Node>
                // and resolve it to a function, then check the arity, then
                // evaluate the arguments, but we can't do that because we
                // don't have the function table here
                let arg_len = args.len();

                for arg in args {
                    self.emit(arg)?;
                }

                self.emit(*func)?;

                self.instructions
                    .push(Instruction::new(Opcode::Call(arg_len), span));
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
                // Check locals first, then globals
                if let Some(index) = self.instructions.resolve_local_index(&name) {
                    self.instructions
                        .push(Instruction::new(Opcode::Load(index), span));
                } else if let Some(index) = self.instructions.resolve_global_index(&name) {
                    self.instructions
                        .push(Instruction::new(Opcode::LoadGlobal(index), span));
                } else {
                    return Err(WalrusError::UndefinedVariable {
                        name,
                        span,
                        src: self.source_ref.source().to_string(),
                        filename: self.source_ref.filename().to_string(),
                    });
                }
            }
            NodeKind::Assign(name, node) => {
                let is_global = self.depth == 0;
                
                if !is_global {
                    // Check for redefinition only in local scopes
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
                }

                self.emit(*node)?;
                
                if is_global {
                    self.define_global_variable(name, span);
                } else {
                    self.define_variable(name, span);
                }
            }
            NodeKind::Reassign(name, node, op) => {
                // Check locals first, then globals
                let (index, is_global) = if let Some(index) = self.instructions.resolve_local_index(name.value()) {
                    (index, false)
                } else if let Some(index) = self.instructions.resolve_global_index(name.value()) {
                    (index, true)
                } else {
                    return Err(WalrusError::UndefinedVariable {
                        name: name.value().to_string(),
                        span,
                        src: self.source_ref.source().to_string(),
                        filename: self.source_ref.filename().to_string(),
                    });
                };

                // For compound assignments (+=, -=, etc.), we need to load the current value first
                match op {
                    Opcode::Add => {
                        if is_global {
                            self.instructions
                                .push(Instruction::new(Opcode::LoadGlobal(index), span));
                        } else {
                            self.instructions
                                .push(Instruction::new(Opcode::Load(index), span));
                        }
                        self.emit(*node)?;
                        self.instructions.push(Instruction::new(Opcode::Add, span));
                    }
                    Opcode::Subtract => {
                        if is_global {
                            self.instructions
                                .push(Instruction::new(Opcode::LoadGlobal(index), span));
                        } else {
                            self.instructions
                                .push(Instruction::new(Opcode::Load(index), span));
                        }
                        self.emit(*node)?;
                        self.instructions
                            .push(Instruction::new(Opcode::Subtract, span));
                    }
                    Opcode::Multiply => {
                        if is_global {
                            self.instructions
                                .push(Instruction::new(Opcode::LoadGlobal(index), span));
                        } else {
                            self.instructions
                                .push(Instruction::new(Opcode::Load(index), span));
                        }
                        self.emit(*node)?;
                        self.instructions
                            .push(Instruction::new(Opcode::Multiply, span));
                    }
                    Opcode::Divide => {
                        if is_global {
                            self.instructions
                                .push(Instruction::new(Opcode::LoadGlobal(index), span));
                        } else {
                            self.instructions
                                .push(Instruction::new(Opcode::Load(index), span));
                        }
                        self.emit(*node)?;
                        self.instructions
                            .push(Instruction::new(Opcode::Divide, span));
                    }
                    Opcode::Modulo => {
                        if is_global {
                            self.instructions
                                .push(Instruction::new(Opcode::LoadGlobal(index), span));
                        } else {
                            self.instructions
                                .push(Instruction::new(Opcode::Load(index), span));
                        }
                        self.emit(*node)?;
                        self.instructions
                            .push(Instruction::new(Opcode::Modulo, span));
                    }
                    _ => {
                        // Simple assignment (=), just emit the new value
                        self.emit(*node)?;
                    }
                }

                // Store the result back
                if is_global {
                    self.instructions
                        .push(Instruction::new(Opcode::ReassignGlobal(index), span));
                } else {
                    self.instructions
                        .push(Instruction::new(Opcode::Reassign(index), span));
                }
            }
            NodeKind::Statements(nodes) => {
                self.inc_depth();

                for node in nodes {
                    self.emit(node)?;
                }

                self.dec_depth(span);
            }
            NodeKind::UnscopedStatements(nodes) => {
                for node in nodes {
                    self.emit(node)?;
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

    fn inc_depth(&mut self) {
        self.instructions.inc_depth();
    }

    fn dec_depth(&mut self, span: Span) {
        let popped = self.instructions.dec_depth();

        if popped > 0 {
            self.instructions
                .push(Instruction::new(Opcode::PopLocal(popped), span));
        }
    }

    fn dec_depth_no_pop(&mut self) {
        self.instructions.dec_depth();
    }

    fn define_variable(&mut self, name: String, span: Span) {
        self.instructions.push_local(name);

        self.instructions
            .push(Instruction::new(Opcode::Store, span));
    }

    fn define_global_variable(&mut self, name: String, span: Span) {
        let index = self.instructions.push_global(name);

        self.instructions
            .push(Instruction::new(Opcode::StoreGlobal(index), span));
    }

    fn define_parameter(&mut self, name: String) {
        // Define parameter in locals without emitting Store opcode
        // The value will already be in locals when the function is called
        self.instructions.push_local(name);
    }

    pub fn instruction_set(self) -> InstructionSet {
        self.instructions.disassemble();
        self.instructions
    }

    pub fn emit_void(&mut self, span: Span) {
        self.instructions.push(Instruction::new(Opcode::Void, span));
    }

    pub fn emit_return(&mut self, span: Span) {
        self.instructions
            .push(Instruction::new(Opcode::Return, span));
    }
}
