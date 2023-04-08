use float_ord::FloatOrd;
use log::{debug, log_enabled};
use rustc_hash::FxHashMap;

use instruction_set::InstructionSet;

use crate::arenas::HeapValue;
use crate::error::WalrusError;
use crate::range::RangeValue;
use crate::source_ref::SourceRef;
use crate::span::Span;
use crate::value::ValueKind;
use crate::vm::opcode::Opcode;
use crate::WalrusResult;

pub mod compiler;
pub mod instruction_set;
pub mod opcode;
mod symbol_table;

pub struct VM<'a> {
    stack: Vec<ValueKind>,
    ip: usize,
    is: InstructionSet,
    source_ref: SourceRef<'a>,
    locals: Vec<ValueKind>,
}

impl<'a> VM<'a> {
    pub fn new(source_ref: SourceRef<'a>, is: InstructionSet) -> Self {
        Self {
            stack: Vec::new(),
            locals: Vec::new(),
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
                    self.push(self.is.get_constant(index));
                }
                Opcode::Load(index) => {
                    let index = self.locals.len() - index - 1;
                    self.push(self.locals[index]);
                }
                Opcode::Store => {
                    let value = self.pop(opcode, span)?;
                    self.locals.push(value);
                }
                Opcode::Reassign(index) => {
                    let value = self.pop(opcode, span)?;
                    let index = self.locals.len() - index - 1;
                    self.locals[index] = value;
                }
                Opcode::List(cap) => {
                    let mut list = Vec::with_capacity(cap);

                    for _ in 0..cap {
                        list.push(self.pop(opcode, span)?);
                    }

                    // todo: can we avoid the reverse here?
                    list.reverse();

                    let value = self.is.get_heap_mut().push(HeapValue::List(list));
                    self.push(value);
                }
                Opcode::Dict(cap) => {
                    let mut dict = FxHashMap::default();

                    for _ in 0..cap {
                        let value = self.pop(opcode, span)?;
                        let key = self.pop(opcode, span)?;

                        dict.insert(key, value);
                    }

                    let value = self.is.get_heap_mut().push(HeapValue::Dict(dict));
                    self.push(value);
                }
                Opcode::Range => {
                    let left = self.pop(opcode, span)?;
                    let right = self.pop(opcode, span)?;

                    // fixme: the spans are wrong here
                    match (left, right) {
                        (ValueKind::Void, ValueKind::Void) => {
                            self.push(ValueKind::Range(RangeValue::new(0, span, -1, span)));
                        }
                        (ValueKind::Void, ValueKind::Int(right)) => {
                            self.push(ValueKind::Range(RangeValue::new(0, span, right, span)));
                        }
                        (ValueKind::Int(left), ValueKind::Void) => {
                            self.push(ValueKind::Range(RangeValue::new(left, span, -1, span)));
                        }
                        (ValueKind::Int(left), ValueKind::Int(right)) => {
                            self.push(ValueKind::Range(RangeValue::new(left, span, right, span)));
                        }
                        // fixme: this is a catch all for now, break it into
                        // errors for left and right and then both
                        (left, right) => {
                            return Err(WalrusError::TypeMismatch {
                                expected: "int".to_string(),
                                found: format!("{} and {}", left.get_type(), right.get_type()),
                                span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            });
                        }
                    }
                }
                Opcode::True => self.push(ValueKind::Bool(true)),
                Opcode::False => self.push(ValueKind::Bool(false)),
                Opcode::Void => self.push(ValueKind::Void),
                Opcode::Pop => {
                    self.pop(opcode, span)?;
                }
                Opcode::PopLocal => {
                    self.locals.pop();
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
                            let a = self.is.get_heap().get_string(a)?;
                            let b = self.is.get_heap().get_string(b)?;

                            let mut s = String::with_capacity(a.len() + b.len());
                            s.push_str(a);
                            s.push_str(b);

                            let value = self.is.get_heap_mut().push(HeapValue::String(s));
                            self.push(value);
                        }
                        (ValueKind::List(a), ValueKind::List(b)) => {
                            let mut a = self.is.get_heap().get_list(a)?.to_vec();
                            let b = self.is.get_heap().get_list(b)?;
                            a.extend(b);

                            let value = self.is.get_heap_mut().push(HeapValue::List(a));
                            self.push(value);
                        }
                        (ValueKind::Dict(a), ValueKind::Dict(b)) => {
                            let mut a = self.is.get_heap().get_dict(a)?.clone();
                            let b = self.is.get_heap().get_dict(b)?;
                            a.extend(b);

                            let value = self.is.get_heap_mut().push(HeapValue::Dict(a));
                            self.push(value);
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
                            let a = self.is.get_heap().get_list(a)?;
                            let b = self.is.get_heap().get_list(b)?;

                            self.push(ValueKind::Bool(a == b));
                        }
                        (ValueKind::Dict(a), ValueKind::Dict(b)) => {
                            let a = self.is.get_heap().get_dict(a)?;
                            let b = self.is.get_heap().get_dict(b)?;

                            self.push(ValueKind::Bool(a == b));
                        }
                        (ValueKind::Function(a), ValueKind::Function(b)) => {
                            let a_func = self.is.get_heap().get_function(a)?;
                            let b_func = self.is.get_heap().get_function(b)?;

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
                            let a = self.is.get_heap().get_list(a)?;
                            let b = self.is.get_heap().get_list(b)?;

                            self.push(ValueKind::Bool(a != b));
                        }
                        (ValueKind::Dict(a), ValueKind::Dict(b)) => {
                            let a = self.is.get_heap().get_dict(a)?;
                            let b = self.is.get_heap().get_dict(b)?;

                            self.push(ValueKind::Bool(a != b));
                        }
                        (ValueKind::Function(a), ValueKind::Function(b)) => {
                            let a_func = self.is.get_heap().get_function(a)?;
                            let b_func = self.is.get_heap().get_function(b)?;

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
                Opcode::Index => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (ValueKind::List(a), ValueKind::Int(b)) => {
                            let a = self.is.get_heap().get_list(a)?;
                            let mut b = b;
                            let original = b;

                            // todo: merge code with other index ops
                            if b < 0 {
                                b += a.len() as i64;
                            }

                            if b < 0 || b >= a.len() as i64 {
                                return Err(WalrusError::IndexOutOfBounds {
                                    index: original,
                                    len: a.len(),
                                    span,
                                    src: self.source_ref.source().to_string(),
                                    filename: self.source_ref.filename().to_string(),
                                });
                            }

                            self.push(a[b as usize]);
                        }
                        (ValueKind::String(a), ValueKind::Int(b)) => {
                            let a = self.is.get_heap().get_string(a)?;
                            let mut b = b;
                            let original = b;

                            if b < 0 {
                                b += a.len() as i64;
                            }

                            if b < 0 || b >= a.len() as i64 {
                                return Err(WalrusError::IndexOutOfBounds {
                                    index: original,
                                    len: a.len(),
                                    span,
                                    src: self.source_ref.source().to_string(),
                                    filename: self.source_ref.filename().to_string(),
                                });
                            }

                            let b = b as usize;
                            let res = a[b..b + 1].to_string();
                            let value = self.is.get_heap_mut().push(HeapValue::String(res));

                            self.push(value);
                        }
                        (ValueKind::Dict(a), b) => {
                            let a = self.is.get_heap().get_dict(a)?;

                            if let Some(value) = a.get(&b) {
                                self.push(*value);
                            } else {
                                // fixme: sometimes this throws even when the objects are equal because
                                // we are comparing the arena keys and not the values
                                todo!("Key not found");
                            }
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Print => {
                    let a = self.pop(opcode, span)?;
                    print!("{}", self.is.stringify(a)?);
                }
                Opcode::Println => {
                    let a = self.pop(opcode, span)?;
                    println!("{}", self.is.stringify(a)?);
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

    fn get(&self, index: usize) -> WalrusResult<&ValueKind> {
        self.stack.get(index).ok_or(WalrusError::UnknownError {
            message: "Failed to resolve local variable stack index".to_string(),
        })
    }

    fn set(&mut self, index: usize, value: ValueKind) -> WalrusResult<()> {
        if let Some(slot) = self.stack.get_mut(index) {
            *slot = value;
            Ok(())
        } else {
            Err(WalrusError::UnknownError {
                message: "Failed to resolve local variable stack index".to_string(),
            })
        }
    }

    fn pop(&mut self, op: Opcode, span: Span) -> WalrusResult<ValueKind> {
        self.stack.pop().ok_or_else(|| WalrusError::StackUnderflow {
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
