use crate::error::WalrusError;
use crate::scope::Scope;
use crate::source_ref::SourceRef;
use crate::span::Span;
use crate::value::{HeapValue, ValueKind};
use crate::vm::opcode::Opcode;
use crate::WalrusResult;
use float_ord::FloatOrd;
use instruction_set::InstructionSet;
use log::{debug, log_enabled};

pub mod compiler;
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
                Opcode::LoadConst(index) => {
                    let value = match self.is.get_constant(index) {
                        // todo: make own Opcode for list and dict
                        ValueKind::List(key) => {
                            let list = Scope::get_mut_list(key)?;

                            for _ in 0..list.capacity() {
                                list.push(self.pop(opcode, span)?);
                            }

                            list.reverse(); // todo: can we avoid this?

                            ValueKind::List(key)
                        }
                        ValueKind::Dict(key) => {
                            let dict = Scope::get_mut_dict(key)?;

                            // fixme: this is wrong
                            for _ in 0..dict.len() {
                                let value = self.pop(opcode, span)?;
                                let key = self.pop(opcode, span)?;

                                dict.insert(key, value);
                            }

                            ValueKind::Dict(key)
                        }
                        value => value,
                    };

                    self.push(value);
                }
                Opcode::Pop => {
                    self.pop(opcode, span)?;
                }
                Opcode::Add => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (ValueKind::Int(a), ValueKind::Int(b)) => {
                            self.push(ValueKind::Int(a + b));
                        }
                        (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                            self.push(ValueKind::Float(FloatOrd(a + b)));
                        }
                        (ValueKind::String(a), ValueKind::String(b)) => {
                            let mut a = Scope::get_string(a)?.to_string();
                            let b = Scope::get_string(b)?;
                            a.push_str(b);

                            self.push(Scope::heap_alloc(HeapValue::String(a)));
                        }
                        (ValueKind::List(a), ValueKind::List(b)) => {
                            let mut a = Scope::get_list(a)?.to_vec();
                            let b = Scope::get_list(b)?;
                            a.extend(b);

                            self.push(Scope::heap_alloc(HeapValue::List(a)));
                        }
                        (ValueKind::Dict(a), ValueKind::Dict(b)) => {
                            let mut a = Scope::get_dict(a)?.to_owned();
                            let b = Scope::get_dict(b)?;
                            a.extend(b);

                            self.push(Scope::heap_alloc(HeapValue::Dict(a)));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Subtract => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (ValueKind::Int(a), ValueKind::Int(b)) => {
                            self.push(ValueKind::Int(a - b));
                        }
                        (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                            self.push(ValueKind::Float(FloatOrd(a - b)));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Multiply => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (ValueKind::Int(a), ValueKind::Int(b)) => {
                            self.push(ValueKind::Int(a * b));
                        }
                        (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                            self.push(ValueKind::Float(FloatOrd(a * b)));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Divide => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (ValueKind::Int(a), ValueKind::Int(b)) => {
                            self.push(ValueKind::Int(a / b));
                        }
                        (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                            self.push(ValueKind::Float(FloatOrd(a / b)));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Power => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (ValueKind::Int(a), ValueKind::Int(b)) => {
                            self.push(ValueKind::Int(a.pow(b as u32)));
                        }
                        (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                            self.push(ValueKind::Float(FloatOrd(a.powf(b))));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Modulo => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (ValueKind::Int(a), ValueKind::Int(b)) => {
                            self.push(ValueKind::Int(a % b));
                        }
                        (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                            self.push(ValueKind::Float(FloatOrd(a % b)));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Negate => {
                    let a = self.pop(opcode, span)?;

                    match a {
                        ValueKind::Int(a) => {
                            self.push(ValueKind::Int(-a));
                        }
                        ValueKind::Float(FloatOrd(a)) => {
                            self.push(ValueKind::Float(FloatOrd(-a)));
                        }
                        _ => return Err(self.construct_err(opcode, a, None, span)),
                    }
                }
                Opcode::Not => {
                    let a = self.pop(opcode, span)?;

                    match a {
                        ValueKind::Bool(a) => {
                            self.push(ValueKind::Bool(!a));
                        }
                        _ => return Err(self.construct_err(opcode, a, None, span)),
                    }
                }
                Opcode::And => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (ValueKind::Bool(a), ValueKind::Bool(b)) => {
                            self.push(ValueKind::Bool(a && b));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Or => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (ValueKind::Bool(a), ValueKind::Bool(b)) => {
                            self.push(ValueKind::Bool(a || b));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Equal => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (ValueKind::List(a), ValueKind::List(b)) => {
                            let a = Scope::get_list(a)?;
                            let b = Scope::get_list(b)?;

                            self.push(ValueKind::Bool(a == b));
                        }
                        (ValueKind::Dict(a), ValueKind::Dict(b)) => {
                            let a = Scope::get_dict(a)?;
                            let b = Scope::get_dict(b)?;

                            self.push(ValueKind::Bool(a == b));
                        }
                        (ValueKind::Function(a), ValueKind::Function(b)) => {
                            let a_func = Scope::get_function(a)?;
                            let b_func = Scope::get_function(b)?;

                            self.push(ValueKind::Bool(a_func == b_func));
                        }
                        (ValueKind::Int(a), ValueKind::Float(FloatOrd(b))) => {
                            self.push(ValueKind::Bool(a as f64 == b));
                        }
                        (ValueKind::Float(FloatOrd(a)), ValueKind::Int(b)) => {
                            self.push(ValueKind::Bool(a == b as f64));
                        }
                        _ => self.push(ValueKind::Bool(a == b)),
                    }
                }
                Opcode::NotEqual => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (ValueKind::List(a), ValueKind::List(b)) => {
                            let a = Scope::get_list(a)?;
                            let b = Scope::get_list(b)?;

                            self.push(ValueKind::Bool(a != b));
                        }
                        (ValueKind::Dict(a), ValueKind::Dict(b)) => {
                            let a = Scope::get_dict(a)?;
                            let b = Scope::get_dict(b)?;

                            self.push(ValueKind::Bool(a != b));
                        }
                        (ValueKind::Function(a), ValueKind::Function(b)) => {
                            let a_func = Scope::get_function(a)?;
                            let b_func = Scope::get_function(b)?;

                            self.push(ValueKind::Bool(a_func != b_func));
                        }
                        (ValueKind::Int(a), ValueKind::Float(FloatOrd(b))) => {
                            self.push(ValueKind::Bool(a as f64 != b));
                        }
                        (ValueKind::Float(FloatOrd(a)), ValueKind::Int(b)) => {
                            self.push(ValueKind::Bool(a != b as f64));
                        }
                        _ => self.push(ValueKind::Bool(a != b)),
                    }
                }
                Opcode::Greater => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (ValueKind::Int(a), ValueKind::Int(b)) => {
                            self.push(ValueKind::Bool(a > b));
                        }
                        (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                            self.push(ValueKind::Bool(a > b));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::GreaterEqual => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (ValueKind::Int(a), ValueKind::Int(b)) => {
                            self.push(ValueKind::Bool(a >= b));
                        }
                        (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                            self.push(ValueKind::Bool(a >= b));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Less => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (ValueKind::Int(a), ValueKind::Int(b)) => {
                            self.push(ValueKind::Bool(a < b));
                        }
                        (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                            self.push(ValueKind::Bool(a < b));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::LessEqual => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (ValueKind::Int(a), ValueKind::Int(b)) => {
                            self.push(ValueKind::Bool(a <= b));
                        }
                        (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                            self.push(ValueKind::Bool(a <= b));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
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
                Opcode::Nop => {}
            }

            self.stack_trace();
        }
    }

    fn construct_err(
        &self,
        op: Opcode,
        a: ValueKind,
        b: Option<ValueKind>,
        span: Span,
    ) -> WalrusError {
        if let Some(b) = b {
            WalrusError::InvalidOperation {
                op,
                left: a.get_type().to_string(),
                right: b.get_type().to_string(),
                span,
                src: self.source_ref.source().to_string(),
                filename: self.source_ref.filename().to_string(),
            }
        } else {
            WalrusError::InvalidUnaryOperation {
                op,
                operand: a.get_type().to_string(),
                span,
                src: self.source_ref.source().to_string(),
                filename: self.source_ref.filename().to_string(),
            }
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

        for (i, frame) in self.stack.iter().enumerate() {
            debug!("| {}: {}", i, frame);
        }
    }
}
