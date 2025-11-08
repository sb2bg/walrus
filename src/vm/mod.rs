use std::cell::RefCell;
use std::io;
use std::io::Write;
use std::ptr::NonNull;
use std::rc::Rc;

use float_ord::FloatOrd;
use log::{debug, log_enabled};
use rustc_hash::FxHashMap;

use instruction_set::InstructionSet;

use crate::WalrusResult;
use crate::arenas::HeapValue;
use crate::error::WalrusError;
use crate::function::WalrusFunction;
use crate::iter::ValueIterator;
use crate::range::RangeValue;
use crate::source_ref::SourceRef;
use crate::span::Span;
use crate::value::Value;
use crate::vm::opcode::{Instruction, Opcode};

pub mod compiler;
pub mod instruction_set;
pub mod opcode;
mod symbol_table;

#[derive(Debug)]
pub struct VM<'a> {
    title: String,
    stack: Vec<Value>,
    ip: usize,
    is: InstructionSet,
    source_ref: SourceRef<'a>,
    locals: Vec<Value>,
    globals: Rc<RefCell<Vec<Value>>>,
    paused: bool,
    breakpoints: Vec<usize>,
}

impl<'a> VM<'a> {
    pub fn new(source_ref: SourceRef<'a>, is: InstructionSet) -> Self {
        Self {
            title: "<main>".to_string(),
            stack: Vec::new(),
            locals: Vec::new(),
            globals: Rc::new(RefCell::new(Vec::new())),
            ip: 0,
            is,
            source_ref,
            paused: false,
            breakpoints: Vec::new(),
        }
    }

    fn create_child(&self, new_is: InstructionSet, title: String) -> Self {
        Self {
            title,
            stack: Vec::new(),
            locals: Vec::new(),                // Functions start with empty locals
            globals: Rc::clone(&self.globals), // Share globals with parent via Rc
            ip: 0,
            is: new_is,
            source_ref: self.source_ref,
            paused: self.paused,
            breakpoints: self.breakpoints.clone(),
        }
    }

    pub fn run(&mut self) -> WalrusResult<Value> {
        while self.ip < self.is.instructions.len() {
            self.is.disassemble_single(self.ip, &self.title);

            if self.paused || self.breakpoints.contains(&self.ip) {
                self.debug_prompt()?;
            }

            let instruction = self.is.get(self.ip);
            let opcode = instruction.opcode();
            let span = instruction.span();

            self.ip += 1;

            match opcode {
                Opcode::LoadConst(index) => {
                    self.push(self.is.get_constant(index));
                }
                Opcode::LoadConst0 => {
                    self.push(self.is.get_constant(0));
                }
                Opcode::LoadConst1 => {
                    self.push(self.is.get_constant(1));
                }
                Opcode::Load(index) => {
                    self.push(self.locals[index as usize]);
                }
                Opcode::LoadLocal0 => {
                    self.push(self.locals[0]);
                }
                Opcode::LoadLocal1 => {
                    self.push(self.locals[1]);
                }
                Opcode::LoadLocal2 => {
                    self.push(self.locals[2]);
                }
                Opcode::LoadLocal3 => {
                    self.push(self.locals[3]);
                }
                Opcode::LoadGlobal(index) => {
                    let value = {
                        let globals = self.globals.borrow();
                        globals[index as usize]
                    };
                    self.push(value);
                }
                Opcode::LoadGlobal0 => {
                    let value = {
                        let globals = self.globals.borrow();
                        globals[0]
                    };
                    self.push(value);
                }
                Opcode::LoadGlobal1 => {
                    let value = {
                        let globals = self.globals.borrow();
                        globals[1]
                    };
                    self.push(value);
                }
                Opcode::LoadGlobal2 => {
                    let value = {
                        let globals = self.globals.borrow();
                        globals[2]
                    };
                    self.push(value);
                }
                Opcode::LoadGlobal3 => {
                    let value = {
                        let globals = self.globals.borrow();
                        globals[3]
                    };
                    self.push(value);
                }
                Opcode::Store => {
                    let value = self.pop(opcode, span)?;
                    self.locals.push(value);
                }
                Opcode::StoreAt(index) => {
                    let value = self.pop(opcode, span)?;
                    let index = index as usize;

                    if index == self.locals.len() {
                        self.locals.push(value);
                    } else {
                        self.locals[index] = value;
                    }
                }
                Opcode::StoreGlobal(index) => {
                    let value = self.pop(opcode, span)?;
                    let index = index as usize;
                    let mut globals = self.globals.borrow_mut();

                    if index == globals.len() {
                        globals.push(value);
                    } else {
                        globals[index] = value;
                    }
                }
                Opcode::Reassign(index) => {
                    let value = self.pop(opcode, span)?;
                    self.locals[index as usize] = value;
                }
                Opcode::ReassignGlobal(index) => {
                    let value = self.pop(opcode, span)?;
                    let mut globals = self.globals.borrow_mut();
                    globals[index as usize] = value;
                }
                Opcode::List(cap) => {
                    let cap = cap as usize;
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
                    let cap = cap as usize;
                    let mut dict = FxHashMap::with_capacity_and_hasher(cap, Default::default());

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
                                expected: "type: todo".to_string(),
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
                        self.ip = offset as usize;
                    }
                }
                Opcode::Jump(offset) => {
                    self.ip = offset as usize;
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
                                // fixme: if another value gets pushed on the stack, this cause a non iterable error
                                self.push(Value::Iter(key));
                                self.push(value);
                            } else {
                                self.ip = offset as usize;
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
                Opcode::Call(args) => {
                    let args = args as usize;
                    let func = self.pop(opcode, span)?;
                    let args = self.pop_n(args, opcode, span)?;

                    match func {
                        Value::Function(key) => {
                            let func = self.is.get_heap().get_function(key)?;

                            match func {
                                WalrusFunction::Rust(func) => {
                                    let result = func.call(args, self.source_ref, span)?;
                                    self.push(result);
                                }
                                WalrusFunction::Vm(func) => {
                                    // All VMs share the global ARENA heap now
                                    let mut child = self.create_child(
                                        InstructionSet {
                                            instructions: func.code.instructions.clone(),
                                            constants: func.code.constants.clone(),
                                            locals: func.code.locals.clone(),
                                            globals: func.code.globals.clone(),
                                        },
                                        format!("fn<{}>", func.name),
                                    );

                                    // Store the arguments as local variables
                                    // The function parameters are already defined as locals during compilation
                                    for arg in args {
                                        child.locals.push(arg);
                                    }

                                    let result = child.run()?;
                                    self.push(result);
                                }
                                _ => {
                                    // In theory, this should never happen because the compiler
                                    // should not compile a call to a node function (but just in case)
                                    return Err(WalrusError::Exception {
                                        message: "Cannot call a node function from the VM"
                                            .to_string(),
                                        span,
                                        src: self.source_ref.source().into(),
                                        filename: self.source_ref.filename().into(),
                                    });
                                }
                            }
                        }
                        _ => {
                            println!("func: {:?}", func);
                            return Err(WalrusError::NotCallable {
                                value: func.get_type().to_string(),
                                span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            });
                        }
                    }
                }
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
                        (Value::String(a), Value::Int(b)) | (Value::Int(b), Value::String(a)) => {
                            let a = self.is.get_heap().get_string(a)?;
                            let mut s = String::with_capacity(a.len() * b as usize);

                            for _ in 0..b {
                                s.push_str(a);
                            }

                            let value = self.is.get_heap_mut().push(HeapValue::String(&s));
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
                            if b < 0 {
                                // Convert to float for negative exponents
                                let result = (a as f64).powf(b as f64);
                                self.push(Value::Float(FloatOrd(result)));
                            } else {
                                self.push(Value::Int(a.pow(b as u32)));
                            }
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
                        (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                            self.push(Value::Bool(a > b as f64));
                        }
                        (Value::Int(a), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Bool(a as f64 > b));
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
                        (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                            self.push(Value::Bool(a >= b as f64));
                        }
                        (Value::Int(a), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Bool(a as f64 >= b));
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
                        (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                            self.push(Value::Bool(a < b as f64));
                        }
                        (Value::Int(a), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Bool(a as f64 < b));
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
                        (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                            self.push(Value::Bool(a <= b as f64));
                        }
                        (Value::Int(a), Value::Float(FloatOrd(b))) => {
                            self.push(Value::Bool(a as f64 <= b));
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
                                // fixme: sometimes this is thrown even when the key exists
                                // this is because it compares the key, not the value, which
                                // means while the key may be different, the contents of the
                                // key may be the same
                                let b_str = b.stringify()?;

                                return Err(WalrusError::KeyNotFound {
                                    key: b_str,
                                    span: span,
                                    src: self.source_ref.source().to_string(),
                                    filename: self.source_ref.filename().to_string(),
                                });
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
                // Stack manipulation opcodes
                Opcode::Dup => {
                    let a = self.pop(opcode, span)?;
                    self.push(a);
                    self.push(a);
                }
                Opcode::Swap => {
                    let b = self.pop(opcode, span)?;
                    let a = self.pop(opcode, span)?;
                    self.push(b);
                    self.push(a);
                }
                Opcode::Pop2 => {
                    self.pop(opcode, span)?;
                    self.pop(opcode, span)?;
                }
                Opcode::Pop3 => {
                    self.pop(opcode, span)?;
                    self.pop(opcode, span)?;
                    self.pop(opcode, span)?;
                }
                Opcode::Nop => {}
            }

            self.stack_trace();
        }

        Err(WalrusError::UnknownError {
            message: "Instruction pointer out of bounds".to_string(),
        })
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

    fn pop_n(&mut self, n: usize, op: Opcode, span: Span) -> WalrusResult<Vec<Value>> {
        let mut values = Vec::with_capacity(n);

        for _ in 0..n {
            values.push(self.pop(op, span)?);
        }

        values.reverse();
        Ok(values)
    }

    fn stack_trace(&self) {
        if !log_enabled!(log::Level::Debug) {
            return;
        }

        for (i, frame) in self.stack.iter().enumerate() {
            debug!("| {} | {}: {}", self.title, i, frame);
        }
    }

    fn debug_prompt(&mut self) -> WalrusResult<()> {
        loop {
            print!("(debug) ");
            io::stdout().flush().expect("Failed to flush stdout");

            let mut input = String::new();
            io::stdin()
                .read_line(&mut input)
                .expect("Failed to read line");

            match input.trim() {
                "s" | "step" => {
                    self.paused = true;
                    break;
                }
                "c" | "continue" => {
                    self.paused = false;
                    break;
                }
                "p" | "print" => self.print_debug_info(),
                "b" | "breakpoint" => {
                    print!("Enter breakpoint line number: ");
                    io::stdout().flush().expect("Failed to flush stdout");
                    let mut line = String::new();
                    io::stdin()
                        .read_line(&mut line)
                        .expect("Failed to read line");
                    if let Ok(line_num) = line.trim().parse::<usize>() {
                        self.breakpoints.push(line_num);
                        debug!("Breakpoint set at line {}", line_num);
                    } else {
                        debug!("Invalid line number");
                    }
                }
                "q" | "quit" => {
                    return Err(WalrusError::UnknownError {
                        message: "Debugger quit".to_string(),
                    });
                }
                _ => debug!(
                    "Unknown command. Available commands: step (s), continue (c), print (p), breakpoint (b), quit (q)"
                ),
            }
        }

        Ok(())
    }

    fn print_current_instruction(&self) {
        let instruction = self.is.get(self.ip);
        debug!(
            "Executing {} -> {}",
            instruction.opcode(),
            &self.source_ref.source()[instruction.span().0..instruction.span().1],
        );
    }

    fn print_debug_info(&self) {
        self.print_current_instruction();
        debug!("Current instruction pointer -> {}", self.ip);
        debug!("Stack ->");
        for (i, value) in self.stack.iter().enumerate() {
            debug!("  {}: {:?}", i, value);
        }
        debug!("Locals ->");
        for (i, value) in self.locals.iter().enumerate() {
            debug!("  {}: {:?}", i, value);
        }
    }
}
