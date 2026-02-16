//! Comparison operation handlers: Equal, NotEqual, Greater, GreaterEqual, Less, LessEqual

use float_ord::FloatOrd;

use crate::error::WalrusError;
use crate::span::Span;
use crate::value::Value;
use crate::vm::opcode::Opcode;
use crate::vm::VM;
use crate::WalrusResult;

impl<'a> VM<'a> {
    #[inline(always)]
    pub(crate) fn handle_equal(&mut self, span: Span) -> WalrusResult<()> {
        let b = self.pop(Opcode::Equal, span)?;
        let a = self.pop(Opcode::Equal, span)?;

        match (a, b) {
            (Value::List(a), Value::List(b)) => {
                let a = self.get_heap().get_list(a)?;
                let b = self.get_heap().get_list(b)?;
                self.push(Value::Bool(a == b));
            }
            (Value::Dict(a), Value::Dict(b)) => {
                let a = self.get_heap().get_dict(a)?;
                let b = self.get_heap().get_dict(b)?;
                self.push(Value::Bool(a == b));
            }
            (Value::Function(a), Value::Function(b)) => {
                let a_func = self.get_heap().get_function(a)?;
                let b_func = self.get_heap().get_function(b)?;
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
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_not_equal(&mut self, span: Span) -> WalrusResult<()> {
        let b = self.pop(Opcode::NotEqual, span)?;
        let a = self.pop(Opcode::NotEqual, span)?;

        match (a, b) {
            (Value::List(a), Value::List(b)) => {
                let a = self.get_heap().get_list(a)?;
                let b = self.get_heap().get_list(b)?;
                self.push(Value::Bool(a != b));
            }
            (Value::Dict(a), Value::Dict(b)) => {
                let a = self.get_heap().get_dict(a)?;
                let b = self.get_heap().get_dict(b)?;
                self.push(Value::Bool(a != b));
            }
            (Value::Function(a), Value::Function(b)) => {
                let a_func = self.get_heap().get_function(a)?;
                let b_func = self.get_heap().get_function(b)?;
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
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_greater(&mut self, span: Span) -> WalrusResult<()> {
        let b = self.pop(Opcode::Greater, span)?;
        let a = self.pop(Opcode::Greater, span)?;

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
                self.push(Value::Bool((a as f64) > b));
            }
            _ => return Err(self.construct_err(Opcode::Greater, a, Some(b), span)),
        }
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_greater_equal(&mut self, span: Span) -> WalrusResult<()> {
        let b = self.pop(Opcode::GreaterEqual, span)?;
        let a = self.pop(Opcode::GreaterEqual, span)?;

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
                self.push(Value::Bool((a as f64) >= b));
            }
            _ => return Err(self.construct_err(Opcode::GreaterEqual, a, Some(b), span)),
        }
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_less(&mut self, span: Span) -> WalrusResult<()> {
        let b = self.pop_unchecked();
        let a = self.pop_unchecked();

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
                self.push(Value::Bool((a as f64) < b));
            }
            _ => return Err(self.construct_err(Opcode::Less, a, Some(b), span)),
        }
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_less_equal(&mut self, span: Span) -> WalrusResult<()> {
        let b = self.pop(Opcode::LessEqual, span)?;
        let a = self.pop(Opcode::LessEqual, span)?;

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
                self.push(Value::Bool((a as f64) <= b));
            }
            _ => return Err(self.construct_err(Opcode::LessEqual, a, Some(b), span)),
        }
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_less_int(&mut self, span: Span) -> WalrusResult<()> {
        let b = self.pop_unchecked();
        let a = self.pop_unchecked();
        if let (Value::Int(a), Value::Int(b)) = (a, b) {
            self.push(Value::Bool(a < b));
            Ok(())
        } else {
            Err(WalrusError::TypeMismatch {
                expected: "int and int".to_string(),
                found: format!("{} and {}", a.get_type(), b.get_type()),
                span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            })
        }
    }
}
