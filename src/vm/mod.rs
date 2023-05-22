use std::ptr::NonNull;

use float_ord::FloatOrd;
use log::{debug, log_enabled};
use rustc_hash::FxHashMap;

use instruction_set::InstructionSet;

use crate::arenas::HeapValue;
use crate::error::WalrusError;
use crate::range::RangeValue;
use crate::source_ref::SourceRef;
use crate::span::Span;
use crate::value::Value;
use crate::vm::opcode::Opcode;
use crate::WalrusResult;

pub mod compiler;
pub mod instruction_set;
pub mod opcode;
mod symbol_table;

pub struct VM<'a> {
    stack: Vec<Value>,
    ip: usize,
    is: InstructionSet,
    source_ref: SourceRef<'a>,
    locals: Vec<Value>,
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

    pub fn run(&mut self) -> WalrusResult<Value> {
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
                    self.push(self.locals[index]);
                }
                Opcode::Store => {
                    let value = self.pop(opcode, span)?;
                    self.locals.push(value);
                }
                Opcode::StoreAt(index) => {
                    let value = self.pop(opcode, span)?;

                    if index == self.locals.len() {
                        self.locals.push(value);
                    }

                    self.locals[index] = value;
                }
                Opcode::Reassign(index) => {
                    let value = self.pop(opcode, span)?;
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
                        (Value::Void, Value::Void) => {
                            self.push(Value::Range(RangeValue::new(0, span, -1, span)));
                        }
                        (Value::Void, Value::Int(right)) => {
                            self.push(Value::Range(RangeValue::new(0, span, right, span)));
                        }
                        (Value::Int(left), Value::Void) => {
                            self.push(Value::Range(RangeValue::new(left, span, -1, span)));
                        }
                        (Value::Int(left), Value::Int(right)) => {
                            self.push(Value::Range(RangeValue::new(left, span, right, span)));
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
                Opcode::True => self.push(Value::Bool(true)),
                Opcode::False => self.push(Value::Bool(false)),
                Opcode::Void => self.push(Value::Void),
                Opcode::Pop => {
                    self.pop(opcode, span)?;
                }
                Opcode::PopLocal(num) => {
                    for _ in 0..num {
                        self.locals.pop();
                    }
                }
                Opcode::JumpIfFalse(offset) => {
                    let value = self.pop(opcode, span)?;

                    if let Value::Bool(false) = value {
                        self.ip = offset;
                    }
                }
                Opcode::Jump(offset) => {
                    self.ip = offset;
                }
                Opcode::GetIter => {
                    let value = self.pop(opcode, span)?;
                    let iter = self.is.get_heap_mut().value_to_iter(value)?;
                    self.push(iter);
                }
                Opcode::IterNext(offset) => {
                    let iter = self.pop(opcode, span)?;

                    match iter {
                        Value::Iter(key) => unsafe {
                            let mut ptr = NonNull::from(self.is.get_heap_mut());
                            let iter = ptr.as_mut().get_mut_iter(key)?;

                            if let Some(value) = iter.next(self.is.get_heap_mut()) {
                                // fixme: if another value gets pushed on the stack, this will be wrong
                                self.push(Value::Iter(key));
                                self.push(value);
                            } else {
                                self.ip = offset;
                            }
                        },
                        value => {
                            return Err(WalrusError::NotIterable {
                                type_name: value.get_type().to_string(),
                                span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            });
                        }
                    }
                }
                Opcode::Call(_args) => {}
                Opcode::Add => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::Int(a), Value::Int(b)) => {
                            self.push(Value::Int(a + b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Float(FloatOrd(a + b)));
                        }
                        (Value::String(a), Value::String(b)) => {
                            let a = self.is.get_heap().get_string(a)?;
                            let b = self.is.get_heap().get_string(b)?;

                            let mut s = String::with_capacity(a.len() + b.len());
                            s.push_str(a);
                            s.push_str(b);

                            let value = self.is.get_heap_mut().push(HeapValue::String(&s));
                            self.push(value);
                        }
                        (Value::List(a), Value::List(b)) => {
                            let mut a = self.is.get_heap().get_list(a)?.to_vec();
                            let b = self.is.get_heap().get_list(b)?;
                            a.extend(b);

                            let value = self.is.get_heap_mut().push(HeapValue::List(a));
                            self.push(value);
                        }
                        (Value::Dict(a), Value::Dict(b)) => {
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
                        (Value::Int(a), Value::Int(b)) => {
                            self.push(Value::Int(a - b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Float(FloatOrd(a - b)));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Multiply => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::Int(a), Value::Int(b)) => {
                            self.push(Value::Int(a * b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Float(FloatOrd(a * b)));
                        }
                        (Value::List(a), Value::Int(b)) | (Value::Int(b), Value::List(a)) => {
                            let a = self.is.get_heap().get_list(a)?;
                            let mut list = Vec::with_capacity(a.len() * b as usize);

                            for _ in 0..b {
                                list.extend(a);
                            }

                            let value = self.is.get_heap_mut().push(HeapValue::List(list));
                            self.push(value);
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Divide => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::Int(a), Value::Int(b)) => {
                            self.push(Value::Int(a / b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Float(FloatOrd(a / b)));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Power => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::Int(a), Value::Int(b)) => {
                            self.push(Value::Int(a.pow(b as u32)));
                        }
                        (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Float(FloatOrd(a.powf(b))));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Modulo => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::Int(a), Value::Int(b)) => {
                            self.push(Value::Int(a % b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Float(FloatOrd(a % b)));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Negate => {
                    let a = self.pop(opcode, span)?;

                    match a {
                        Value::Int(a) => {
                            self.push(Value::Int(-a));
                        }
                        Value::Float(FloatOrd(a)) => {
                            self.push(Value::Float(FloatOrd(-a)));
                        }
                        _ => return Err(self.construct_err(opcode, a, None, span)),
                    }
                }
                Opcode::Not => {
                    let a = self.pop(opcode, span)?;

                    match a {
                        Value::Bool(a) => {
                            self.push(Value::Bool(!a));
                        }
                        _ => return Err(self.construct_err(opcode, a, None, span)),
                    }
                }
                Opcode::And => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::Bool(a), Value::Bool(b)) => {
                            self.push(Value::Bool(a && b));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Or => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::Bool(a), Value::Bool(b)) => {
                            self.push(Value::Bool(a || b));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Equal => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::List(a), Value::List(b)) => {
                            let a = self.is.get_heap().get_list(a)?;
                            let b = self.is.get_heap().get_list(b)?;

                            self.push(Value::Bool(a == b));
                        }
                        (Value::Dict(a), Value::Dict(b)) => {
                            let a = self.is.get_heap().get_dict(a)?;
                            let b = self.is.get_heap().get_dict(b)?;

                            self.push(Value::Bool(a == b));
                        }
                        (Value::Function(a), Value::Function(b)) => {
                            let a_func = self.is.get_heap().get_function(a)?;
                            let b_func = self.is.get_heap().get_function(b)?;

                            self.push(Value::Bool(a_func == b_func));
                        }
                        (Value::Int(a), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Bool(a as f64 == b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                            self.push(Value::Bool(a == b as f64));
                        }
                        _ => self.push(Value::Bool(a == b)),
                    }
                }
                Opcode::NotEqual => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::List(a), Value::List(b)) => {
                            let a = self.is.get_heap().get_list(a)?;
                            let b = self.is.get_heap().get_list(b)?;

                            self.push(Value::Bool(a != b));
                        }
                        (Value::Dict(a), Value::Dict(b)) => {
                            let a = self.is.get_heap().get_dict(a)?;
                            let b = self.is.get_heap().get_dict(b)?;

                            self.push(Value::Bool(a != b));
                        }
                        (Value::Function(a), Value::Function(b)) => {
                            let a_func = self.is.get_heap().get_function(a)?;
                            let b_func = self.is.get_heap().get_function(b)?;

                            self.push(Value::Bool(a_func != b_func));
                        }
                        (Value::Int(a), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Bool(a as f64 != b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                            self.push(Value::Bool(a != b as f64));
                        }
                        _ => self.push(Value::Bool(a != b)),
                    }
                }
                Opcode::Greater => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::Int(a), Value::Int(b)) => {
                            self.push(Value::Bool(a > b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Bool(a > b));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::GreaterEqual => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::Int(a), Value::Int(b)) => {
                            self.push(Value::Bool(a >= b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Bool(a >= b));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Less => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::Int(a), Value::Int(b)) => {
                            self.push(Value::Bool(a < b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Bool(a < b));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::LessEqual => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::Int(a), Value::Int(b)) => {
                            self.push(Value::Bool(a <= b));
                        }
                        (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Bool(a <= b));
                        }
                        _ => return Err(self.construct_err(opcode, a, Some(b), span)),
                    }
                }
                Opcode::Index => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;

                    match (a, b) {
                        (Value::List(a), Value::Int(b)) => {
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
                        (Value::String(a), Value::Int(b)) => {
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
                            let value = self.is.get_heap_mut().push(HeapValue::String(&res));

                            self.push(value);
                        }
                        (Value::Dict(a), b) => {
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

    fn construct_err(&self, op: Opcode, a: Value, b: Option<Value>, span: Span) -> WalrusError {
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

    fn push(&mut self, value: Value) {
        self.stack.push(value);
    }

    fn get(&self, index: usize) -> WalrusResult<&Value> {
        self.stack.get(index).ok_or(WalrusError::UnknownError {
            message: "Failed to resolve local variable stack index".to_string(),
        })
    }

    fn set(&mut self, index: usize, value: Value) -> WalrusResult<()> {
        if let Some(slot) = self.stack.get_mut(index) {
            *slot = value;
            Ok(())
        } else {
            Err(WalrusError::UnknownError {
                message: "Failed to resolve local variable stack index".to_string(),
            })
        }
    }

    fn pop(&mut self, op: Opcode, span: Span) -> WalrusResult<Value> {
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
