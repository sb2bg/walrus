//! Loop handlers: GetIter, IterNext, ForRangeInit, ForRangeNext

use std::ptr::NonNull;

use log::debug;

use crate::WalrusResult;
use crate::error::WalrusError;
use crate::iter::ValueIterator;
use crate::span::Span;
use crate::value::Value;
use crate::vm::opcode::Opcode;
use crate::vm::VM;

impl<'a> VM<'a> {
    #[inline(always)]
    pub(crate) fn handle_get_iter(&mut self, span: Span) -> WalrusResult<()> {
        let value = self.pop(Opcode::GetIter, span)?;
        let iter = self.get_heap_mut().value_to_iter(value)?;
        self.push(iter);
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_iter_next(&mut self, offset: u32, span: Span) -> WalrusResult<()> {
        // JIT PROFILING: This is also a loop header for iterator-based loops
        if self.profiling_enabled {
            let loop_header_ip = self.ip - 1;

            // Dynamic loop registration for iterator loops
            if !self.hotspot_detector.is_loop_header(loop_header_ip) {
                self.hotspot_detector.register_loop(
                    loop_header_ip,
                    loop_header_ip,
                    offset as usize,
                );
            }

            if self.hotspot_detector.record_loop_iteration(loop_header_ip) {
                debug!("Hot iterator loop detected at IP {}", loop_header_ip);
            }
        }

        let iter = self.pop(Opcode::IterNext(offset), span)?;

        match iter {
            Value::Iter(key) => unsafe {
                let mut ptr = NonNull::from(self.get_heap_mut());
                let iter = ptr.as_mut().get_mut_iter(key)?;

                if let Some(value) = iter.next(self.get_heap_mut()) {
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
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_for_range_init(&mut self, local_idx: u32) {
        let end = self.pop_unchecked();
        let start = self.pop_unchecked();
        let fp = self.frame_pointer();
        let idx = fp + local_idx as usize;
        // Ensure we have space for both values
        while self.locals.len() <= idx + 1 {
            self.locals.push(Value::Void);
        }
        self.locals[idx] = start;
        self.locals[idx + 1] = end;
    }

    #[inline(always)]
    pub(crate) fn handle_for_range_next(
        &mut self,
        jump_target: u32,
        local_idx: u32,
        span: Span,
    ) -> WalrusResult<bool> {
        let loop_header_ip = self.ip - 1;
        let exit_ip = jump_target as usize;

        // Profile for hotspot detection
        self.profile_loop_iteration(loop_header_ip, exit_ip);

        // Try JIT execution if available
        #[cfg(feature = "jit")]
        if let Some(jit_exit) = self.try_jit_range_loop(loop_header_ip, local_idx as u16, jump_target as u16) {
            self.ip = jit_exit;
            return Ok(true); // Signal to continue in outer loop
        }

        // Try to compile hot loops
        #[cfg(feature = "jit")]
        self.try_compile_hot_range_loop(loop_header_ip, exit_ip);

        // Standard interpreted execution
        let fp = self.frame_pointer();
        let idx = fp + local_idx as usize;
        if let (Value::Int(current), Value::Int(end)) = (self.locals[idx], self.locals[idx + 1]) {
            if current < end {
                self.push(Value::Int(current));
                self.locals[idx] = Value::Int(current + 1);
            } else {
                self.ip = jump_target as usize;
            }
            Ok(false)
        } else {
            Err(WalrusError::TypeMismatch {
                expected: "int and int (range bounds)".to_string(),
                found: format!(
                    "{} and {}",
                    self.locals[idx].get_type(),
                    self.locals[idx + 1].get_type()
                ),
                span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            })
        }
    }
}
