//! Stack manipulation handlers: LoadConst, Load, Store, Pop, Dup, Swap

use crate::WalrusResult;
use crate::error::WalrusError;
use crate::span::Span;
use crate::value::Value;
use crate::vm::opcode::Opcode;
use crate::vm::VM;

impl<'a> VM<'a> {
    #[inline(always)]
    pub(crate) fn handle_load_const(&mut self, index: u32) {
        self.push(self.current_frame().instructions.get_constant(index));
    }

    #[inline(always)]
    pub(crate) fn handle_load_const_0(&mut self) {
        self.push(self.current_frame().instructions.get_constant(0));
    }

    #[inline(always)]
    pub(crate) fn handle_load_const_1(&mut self) {
        self.push(self.current_frame().instructions.get_constant(1));
    }

    #[inline(always)]
    pub(crate) fn handle_load(&mut self, index: u32) {
        let fp = self.frame_pointer();
        self.push(self.locals[fp + index as usize]);
    }

    #[inline(always)]
    pub(crate) fn handle_load_local_0(&mut self) {
        let fp = self.frame_pointer();
        self.push(self.locals[fp]);
    }

    #[inline(always)]
    pub(crate) fn handle_load_local_1(&mut self) {
        let fp = self.frame_pointer();
        self.push(self.locals[fp + 1]);
    }

    #[inline(always)]
    pub(crate) fn handle_load_local_2(&mut self) {
        let fp = self.frame_pointer();
        self.push(self.locals[fp + 2]);
    }

    #[inline(always)]
    pub(crate) fn handle_load_local_3(&mut self) {
        let fp = self.frame_pointer();
        self.push(self.locals[fp + 3]);
    }

    #[inline(always)]
    pub(crate) fn handle_load_global(&mut self, index: u32) {
        let value = {
            let globals = self.globals.borrow();
            globals[index as usize]
        };
        self.push(value);
    }

    #[inline(always)]
    pub(crate) fn handle_load_global_0(&mut self) {
        let value = {
            let globals = self.globals.borrow();
            globals[0]
        };
        self.push(value);
    }

    #[inline(always)]
    pub(crate) fn handle_load_global_1(&mut self) {
        let value = {
            let globals = self.globals.borrow();
            globals[1]
        };
        self.push(value);
    }

    #[inline(always)]
    pub(crate) fn handle_load_global_2(&mut self) {
        let value = {
            let globals = self.globals.borrow();
            globals[2]
        };
        self.push(value);
    }

    #[inline(always)]
    pub(crate) fn handle_load_global_3(&mut self) {
        let value = {
            let globals = self.globals.borrow();
            globals[3]
        };
        self.push(value);
    }

    #[inline(always)]
    pub(crate) fn handle_store(&mut self, span: Span) -> WalrusResult<()> {
        let value = self.pop(Opcode::Store, span)?;
        self.locals.push(value);
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_store_at(&mut self, index: u32, span: Span) -> WalrusResult<()> {
        let value = self.pop(Opcode::StoreAt(index), span)?;
        let fp = self.frame_pointer();
        let abs_index = fp + index as usize;

        if abs_index == self.locals.len() {
            self.locals.push(value);
        } else {
            self.locals[abs_index] = value;
        }
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_store_global(&mut self, index: u32, span: Span) -> WalrusResult<()> {
        let value = self.pop(Opcode::StoreGlobal(index), span)?;
        let index = index as usize;
        let mut globals = self.globals.borrow_mut();

        if index == globals.len() {
            globals.push(value);
        } else {
            globals[index] = value;
        }
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_reassign(&mut self, index: u32, span: Span) -> WalrusResult<()> {
        let value = self.pop(Opcode::Reassign(index), span)?;
        let fp = self.frame_pointer();
        self.locals[fp + index as usize] = value;
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_reassign_global(&mut self, index: u32, span: Span) -> WalrusResult<()> {
        let value = self.pop(Opcode::ReassignGlobal(index), span)?;
        let mut globals = self.globals.borrow_mut();
        globals[index as usize] = value;
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_pop(&mut self, span: Span) -> WalrusResult<()> {
        self.pop(Opcode::Pop, span)?;
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_pop_local(&mut self, num: u32) {
        for _ in 0..num {
            self.locals.pop();
        }
    }

    #[inline(always)]
    pub(crate) fn handle_pop2(&mut self, span: Span) -> WalrusResult<()> {
        self.pop(Opcode::Pop2, span)?;
        self.pop(Opcode::Pop2, span)?;
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_pop3(&mut self, span: Span) -> WalrusResult<()> {
        self.pop(Opcode::Pop3, span)?;
        self.pop(Opcode::Pop3, span)?;
        self.pop(Opcode::Pop3, span)?;
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_dup(&mut self, span: Span) -> WalrusResult<()> {
        let a = self.pop(Opcode::Dup, span)?;
        self.push(a);
        self.push(a);
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_swap(&mut self, span: Span) -> WalrusResult<()> {
        let b = self.pop(Opcode::Swap, span)?;
        let a = self.pop(Opcode::Swap, span)?;
        self.push(b);
        self.push(a);
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_true(&mut self) {
        self.push(Value::Bool(true));
    }

    #[inline(always)]
    pub(crate) fn handle_false(&mut self) {
        self.push(Value::Bool(false));
    }

    #[inline(always)]
    pub(crate) fn handle_void(&mut self) {
        self.push(Value::Void);
    }

    #[inline(always)]
    pub(crate) fn handle_increment_local(&mut self, index: u32, span: Span) -> WalrusResult<()> {
        let fp = self.frame_pointer();
        let idx = fp + index as usize;
        if let Value::Int(v) = self.locals[idx] {
            self.locals[idx] = Value::Int(v + 1);
            Ok(())
        } else {
            Err(WalrusError::TypeMismatch {
                expected: "int".to_string(),
                found: self.locals[idx].get_type().to_string(),
                span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            })
        }
    }

    #[inline(always)]
    pub(crate) fn handle_decrement_local(&mut self, index: u32, span: Span) -> WalrusResult<()> {
        let fp = self.frame_pointer();
        let idx = fp + index as usize;
        if let Value::Int(v) = self.locals[idx] {
            self.locals[idx] = Value::Int(v - 1);
            Ok(())
        } else {
            Err(WalrusError::TypeMismatch {
                expected: "int".to_string(),
                found: self.locals[idx].get_type().to_string(),
                span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            })
        }
    }
}
