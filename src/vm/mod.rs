use crate::ast::Op;
use crate::error::WalrusError;
use crate::scope::Scope;
use crate::source_ref::SourceRef;
use crate::span::Span;
use crate::value::{HeapValue, ValueKind};
use crate::WalrusResult;
use float_ord::FloatOrd;
use instruction_set::InstructionSet;
use log::{debug, log_enabled};
use opcode::Opcode;

pub mod instruction_set;
pub mod opcode;

pub struct VM<'a> {
    stack: Vec<ValueKind>,
    ip: usize,
    is: InstructionSet,
    source_ref: SourceRef<'a>,
}

impl<'a> VM<'a> {
    pub fn new(source_ref: SourceRef<'a>, is: InstructionSet) -> Self {
        Self {
            stack: Vec::new(),
            ip: 0,
            is,
            source_ref,
        }
    }

    pub fn run(&mut self) -> WalrusResult<ValueKind> {
        loop {
            self.is.disassemble_single(self.ip);

            let instruction = self.is.get(self.ip);
            let opcode = instruction.opcode();
            let span = instruction.span();

            self.ip += 1;

            match opcode {
                Opcode::Constant(index) => {
                    self.stack.push(self.is.get_constant(index));
                }
                Opcode::Pop => {
                    self.pop(opcode, span)?;
                }
                Opcode::Add => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (ValueKind::Int(a), ValueKind::Int(b)) => {
                            self.stack.push(ValueKind::Int(a + b));
                        }
                        (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                            self.stack.push(ValueKind::Float(FloatOrd(a + b)));
                        }
                        (ValueKind::String(a), ValueKind::String(b)) => {
                            let mut a = Scope::get_string(a)?.to_string();
                            let b = Scope::get_string(b)?;
                            a.push_str(b);

                            self.stack.push(Scope::heap_alloc(HeapValue::String(a)));
                        }
                        (ValueKind::List(a), ValueKind::List(b)) => {
                            let mut a = Scope::get_list(a)?.to_vec();
                            let b = Scope::get_list(b)?;
                            a.extend(b);

                            self.stack.push(Scope::heap_alloc(HeapValue::List(a)));
                        }
                        (ValueKind::Dict(a), ValueKind::Dict(b)) => {
                            let mut a = Scope::get_dict(a)?.to_owned();
                            let b = Scope::get_dict(b)?;
                            a.extend(b);

                            self.stack.push(Scope::heap_alloc(HeapValue::Dict(a)));
                        }
                        _ => return Err(self.construct_err(Op::Add, a, b, span)),
                    }
                }
                Opcode::Subtract => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (ValueKind::Int(a), ValueKind::Int(b)) => {
                            self.stack.push(ValueKind::Int(a - b));
                        }
                        (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                            self.stack.push(ValueKind::Float(FloatOrd(a - b)));
                        }
                        _ => return Err(self.construct_err(Op::Sub, a, b, span)),
                    }
                }
                Opcode::Multiply => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (ValueKind::Int(a), ValueKind::Int(b)) => {
                            self.stack.push(ValueKind::Int(a * b));
                        }
                        (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                            self.stack.push(ValueKind::Float(FloatOrd(a * b)));
                        }
                        _ => return Err(self.construct_err(Op::Mul, a, b, span)),
                    }
                }
                Opcode::Divide => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (ValueKind::Int(a), ValueKind::Int(b)) => {
                            self.stack.push(ValueKind::Int(a / b));
                        }
                        (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                            self.stack.push(ValueKind::Float(FloatOrd(a / b)));
                        }
                        _ => return Err(self.construct_err(Op::Div, a, b, span)),
                    }
                }
                Opcode::Power => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (ValueKind::Int(a), ValueKind::Int(b)) => {
                            self.stack.push(ValueKind::Int(a.pow(b as u32)));
                        }
                        (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                            self.stack.push(ValueKind::Float(FloatOrd(a.powf(b))));
                        }
                        _ => return Err(self.construct_err(Op::Pow, a, b, span)),
                    }
                }
                Opcode::Modulo => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (ValueKind::Int(a), ValueKind::Int(b)) => {
                            self.stack.push(ValueKind::Int(a % b));
                        }
                        (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                            self.stack.push(ValueKind::Float(FloatOrd(a % b)));
                        }
                        _ => return Err(self.construct_err(Op::Mod, a, b, span)),
                    }
                }
                Opcode::Negate => {
                    let a = self.pop(opcode, span)?;

                    match a {
                        ValueKind::Int(a) => {
                            self.stack.push(ValueKind::Int(-a));
                        }
                        ValueKind::Float(FloatOrd(a)) => {
                            self.stack.push(ValueKind::Float(FloatOrd(-a)));
                        }
                        _ => {
                            return Err(WalrusError::InvalidUnaryOperation {
                                op: Op::Sub,
                                operand: a.get_type().to_string(),
                                span,
                                src: self.source_ref.source().to_string(),
                                filename: self.source_ref.filename().to_string(),
                            })
                        }
                    }
                }
                Opcode::Not => {
                    let a = self.pop(opcode, span)?;

                    match a {
                        ValueKind::Bool(a) => {
                            self.stack.push(ValueKind::Bool(!a));
                        }
                        _ => {
                            return Err(WalrusError::InvalidUnaryOperation {
                                op: Op::Not,
                                operand: a.get_type().to_string(),
                                span,
                                src: self.source_ref.source().to_string(),
                                filename: self.source_ref.filename().to_string(),
                            })
                        }
                    }
                }
                Opcode::Print => {
                    let a = self.pop(opcode, span)?;
                    print!("{}", a.stringify()?);
                }
                Opcode::Println => {
                    let a = self.pop(opcode, span)?;
                    println!("{}", a.stringify()?);
                }
                Opcode::Return => {
                    let a = self.pop(opcode, span)?;
                    return Ok(a);
                }
            }

            self.stack_trace();
        }
    }

    fn construct_err(&self, op: Op, left: ValueKind, right: ValueKind, span: Span) -> WalrusError {
        WalrusError::InvalidOperation {
            op,
            left: left.get_type().to_string(),
            right: right.get_type().to_string(),
            span,
            src: self.source_ref.source().to_string(),
            filename: self.source_ref.filename().to_string(),
        }
    }

    fn push(&mut self, value: ValueKind) {
        self.stack.push(value);
    }

    fn pop(&mut self, op: Opcode, span: Span) -> WalrusResult<ValueKind> {
        self.stack.pop().ok_or(WalrusError::StackUnderflow {
            op,
            span,
            src: self.source_ref.source().to_string(),
            filename: self.source_ref.filename().to_string(),
        })
    }

    fn stack_trace(&self) {
        if !log_enabled!(log::Level::Debug) {
            return;
        }

        debug!("| == Stack trace ==");

        for (i, frame) in self.stack.iter().enumerate() {
            debug!("| {}: {}", i, frame);
        }
    }
}
