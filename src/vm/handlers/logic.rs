//! Logic operation handlers: And, Or, Not

use crate::WalrusResult;
use crate::span::Span;
use crate::value::Value;
use crate::vm::VM;
use crate::vm::opcode::Opcode;

impl<'a> VM<'a> {
    #[inline(always)]
    pub(crate) fn handle_not(&mut self, span: Span) -> WalrusResult<()> {
        let a = self.pop(Opcode::Not, span)?;
        self.push(Value::Bool(!self.get_heap().is_truthy(a)?));
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_and(&mut self, span: Span) -> WalrusResult<()> {
        let b = self.pop(Opcode::And, span)?;
        let a = self.pop(Opcode::And, span)?;
        self.push(Value::Bool(
            self.get_heap().is_truthy(a)? && self.get_heap().is_truthy(b)?,
        ));
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_or(&mut self, span: Span) -> WalrusResult<()> {
        let b = self.pop(Opcode::Or, span)?;
        let a = self.pop(Opcode::Or, span)?;
        self.push(Value::Bool(
            self.get_heap().is_truthy(a)? || self.get_heap().is_truthy(b)?,
        ));
        Ok(())
    }
}
