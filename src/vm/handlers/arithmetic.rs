//! Arithmetic operation handlers: Add, Subtract, Multiply, Divide, Power, Modulo, Negate

use float_ord::FloatOrd;

use crate::WalrusResult;
use crate::arenas::HeapValue;
use crate::error::WalrusError;
use crate::jit::WalrusType;
use crate::span::Span;
use crate::value::Value;
use crate::vm::opcode::Opcode;
use crate::vm::VM;

impl<'a> VM<'a> {
    #[inline(always)]
    pub(crate) fn handle_add(&mut self, span: Span) -> WalrusResult<()> {
        let b = self.pop_unchecked();
        let a = self.pop_unchecked();

        // JIT TYPE PROFILING: Track operand types for arithmetic
        if self.profiling_enabled {
            let type_a = WalrusType::from_value(&a);
            let type_b = WalrusType::from_value(&b);
            self.type_profile.observe(self.ip - 1, type_a);
            self.type_profile.observe(self.ip - 1, type_b);
        }

        match (a, b) {
            (Value::Int(a), Value::Int(b)) => {
                self.push(Value::Int(a + b));
            }
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                self.push(Value::Float(FloatOrd(a + b)));
            }
            (Value::Int(a), Value::Float(FloatOrd(b))) => {
                self.push(Value::Float(FloatOrd(a as f64 + b)));
            }
            (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                self.push(Value::Float(FloatOrd(a + b as f64)));
            }
            (Value::String(a), Value::String(b)) => {
                let a = self.get_heap().get_string(a)?;
                let b = self.get_heap().get_string(b)?;

                let mut s = String::with_capacity(a.len() + b.len());
                s.push_str(a);
                s.push_str(b);

                let value = self.get_heap_mut().push(HeapValue::String(&s));
                self.push(value);
            }
            (Value::List(a), Value::List(b)) => {
                let mut a = self.get_heap().get_list(a)?.to_vec();
                let b = self.get_heap().get_list(b)?;
                a.extend(b);

                let value = self.get_heap_mut().push(HeapValue::List(a));
                self.push(value);
            }
            (Value::Dict(a), Value::Dict(b)) => {
                let mut a = self.get_heap().get_dict(a)?.clone();
                let b = self.get_heap().get_dict(b)?;
                a.extend(b);

                let value = self.get_heap_mut().push(HeapValue::Dict(a));
                self.push(value);
            }
            _ => return Err(self.construct_err(Opcode::Add, a, Some(b), span)),
        }
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_add_int(&mut self, span: Span) -> WalrusResult<()> {
        let b = self.pop_unchecked();
        let a = self.pop_unchecked();
        if let (Value::Int(a), Value::Int(b)) = (a, b) {
            self.push(Value::Int(a + b));
            Ok(())
        } else {
            Err(WalrusError::Exception {
                message: "AddInt requires integers".to_string(),
                span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            })
        }
    }

    #[inline(always)]
    pub(crate) fn handle_subtract(&mut self, span: Span) -> WalrusResult<()> {
        let b = self.pop_unchecked();
        let a = self.pop_unchecked();

        match (a, b) {
            (Value::Int(a), Value::Int(b)) => {
                self.push(Value::Int(a - b));
            }
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                self.push(Value::Float(FloatOrd(a - b)));
            }
            (Value::Int(a), Value::Float(FloatOrd(b))) => {
                self.push(Value::Float(FloatOrd(a as f64 - b)));
            }
            (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                self.push(Value::Float(FloatOrd(a - b as f64)));
            }
            _ => return Err(self.construct_err(Opcode::Subtract, a, Some(b), span)),
        }
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_subtract_int(&mut self, span: Span) -> WalrusResult<()> {
        let b = self.pop_unchecked();
        let a = self.pop_unchecked();
        if let (Value::Int(a), Value::Int(b)) = (a, b) {
            self.push(Value::Int(a - b));
            Ok(())
        } else {
            Err(WalrusError::Exception {
                message: "SubtractInt requires integers".to_string(),
                span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            })
        }
    }

    #[inline(always)]
    pub(crate) fn handle_multiply(&mut self, span: Span) -> WalrusResult<()> {
        let b = self.pop_unchecked();
        let a = self.pop_unchecked();

        match (a, b) {
            (Value::Int(a), Value::Int(b)) => {
                self.push(Value::Int(a * b));
            }
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                self.push(Value::Float(FloatOrd(a * b)));
            }
            (Value::Int(a), Value::Float(FloatOrd(b))) => {
                self.push(Value::Float(FloatOrd(a as f64 * b)));
            }
            (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                self.push(Value::Float(FloatOrd(a * b as f64)));
            }
            (Value::List(a), Value::Int(b)) | (Value::Int(b), Value::List(a)) => {
                let a = self.get_heap().get_list(a)?;
                let mut list = Vec::with_capacity(a.len() * b as usize);

                for _ in 0..b {
                    list.extend(a);
                }

                let value = self.get_heap_mut().push(HeapValue::List(list));
                self.push(value);
            }
            (Value::String(a), Value::Int(b)) | (Value::Int(b), Value::String(a)) => {
                let a = self.get_heap().get_string(a)?;
                let mut s = String::with_capacity(a.len() * b as usize);

                for _ in 0..b {
                    s.push_str(a);
                }

                let value = self.get_heap_mut().push(HeapValue::String(&s));
                self.push(value);
            }
            _ => return Err(self.construct_err(Opcode::Multiply, a, Some(b), span)),
        }
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_divide(&mut self, span: Span) -> WalrusResult<()> {
        let b = self.pop(Opcode::Divide, span)?;
        let a = self.pop(Opcode::Divide, span)?;

        match (a, b) {
            (Value::Int(a), Value::Int(b)) => {
                if b == 0 {
                    return Err(WalrusError::DivisionByZero {
                        span,
                        src: self.source_ref.source().to_string(),
                        filename: self.source_ref.filename().to_string(),
                    });
                }
                self.push(Value::Int(a / b));
            }
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                self.push(Value::Float(FloatOrd(a / b)));
            }
            (Value::Int(a), Value::Float(FloatOrd(b))) => {
                self.push(Value::Float(FloatOrd(a as f64 / b)));
            }
            (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                self.push(Value::Float(FloatOrd(a / b as f64)));
            }
            _ => return Err(self.construct_err(Opcode::Divide, a, Some(b), span)),
        }
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_power(&mut self, span: Span) -> WalrusResult<()> {
        let b = self.pop(Opcode::Power, span)?;
        let a = self.pop(Opcode::Power, span)?;

        match (a, b) {
            (Value::Int(a), Value::Int(b)) => {
                if b < 0 {
                    let result = (a as f64).powf(b as f64);
                    self.push(Value::Float(FloatOrd(result)));
                } else {
                    self.push(Value::Int(a.pow(b as u32)));
                }
            }
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                self.push(Value::Float(FloatOrd(a.powf(b))));
            }
            (Value::Int(a), Value::Float(FloatOrd(b))) => {
                self.push(Value::Float(FloatOrd((a as f64).powf(b))));
            }
            (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                self.push(Value::Float(FloatOrd(a.powf(b as f64))));
            }
            _ => return Err(self.construct_err(Opcode::Power, a, Some(b), span)),
        }
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_modulo(&mut self, span: Span) -> WalrusResult<()> {
        let b = self.pop(Opcode::Modulo, span)?;
        let a = self.pop(Opcode::Modulo, span)?;

        match (a, b) {
            (Value::Int(a), Value::Int(b)) => {
                if b == 0 {
                    return Err(WalrusError::DivisionByZero {
                        span,
                        src: self.source_ref.source().to_string(),
                        filename: self.source_ref.filename().to_string(),
                    });
                }
                self.push(Value::Int(a % b));
            }
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                self.push(Value::Float(FloatOrd(a % b)));
            }
            (Value::Int(a), Value::Float(FloatOrd(b))) => {
                self.push(Value::Float(FloatOrd(a as f64 % b)));
            }
            (Value::Float(FloatOrd(a)), Value::Int(b)) => {
                self.push(Value::Float(FloatOrd(a % b as f64)));
            }
            _ => return Err(self.construct_err(Opcode::Modulo, a, Some(b), span)),
        }
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_negate(&mut self, span: Span) -> WalrusResult<()> {
        let a = self.pop(Opcode::Negate, span)?;

        match a {
            Value::Int(a) => {
                self.push(Value::Int(-a));
            }
            Value::Float(FloatOrd(a)) => {
                self.push(Value::Float(FloatOrd(-a)));
            }
            _ => return Err(self.construct_err(Opcode::Negate, a, None, span)),
        }
        Ok(())
    }
}
