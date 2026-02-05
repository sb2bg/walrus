//! Logic operation handlers: And, Or, Not

use crate::WalrusResult;
use crate::span::Span;
use crate::value::Value;
use crate::vm::opcode::Opcode;
use crate::vm::VM;

impl<'a> VM<'a> {
    #[inline(always)]
    pub(crate) fn handle_not(&mut self, span: Span) -> WalrusResult<()> {
        let a = self.pop(Opcode::Not, span)?;

        match a {
            Value::Bool(a) => {
                self.push(Value::Bool(!a));
            }
            _ => return Err(self.construct_err(Opcode::Not, a, None, span)),
        }
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_and(&mut self, span: Span) -> WalrusResult<()> {
        let b = self.pop(Opcode::And, span)?;
        let a = self.pop(Opcode::And, span)?;

        match (a, b) {
            (Value::Bool(a), Value::Bool(b)) => {
                self.push(Value::Bool(a && b));
            }
            _ => return Err(self.construct_err(Opcode::And, a, Some(b), span)),
        }
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_or(&mut self, span: Span) -> WalrusResult<()> {
        let b = self.pop(Opcode::Or, span)?;
        let a = self.pop(Opcode::Or, span)?;

        match (a, b) {
            (Value::Bool(a), Value::Bool(b)) => {
                self.push(Value::Bool(a || b));
            }
            _ => return Err(self.construct_err(Opcode::Or, a, Some(b), span)),
        }
        Ok(())
    }
}
