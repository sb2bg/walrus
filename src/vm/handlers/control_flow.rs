//! Control flow handlers: Jump, JumpIfFalse, Call, TailCall, Return

use std::rc::Rc;

use log::debug;

use crate::WalrusResult;
use crate::error::WalrusError;
use crate::function::WalrusFunction;
use crate::jit::WalrusType;
use crate::span::Span;
use crate::value::Value;
use crate::vm::opcode::Opcode;
use crate::vm::{CallFrame, VM};

/// Result of a call handler - either continue execution or return a value
pub enum CallResult {
    Continue,
    Return(Value),
}

impl<'a> VM<'a> {
    #[inline(always)]
    pub(crate) fn handle_jump(&mut self, offset: u32) {
        // JIT PROFILING: Backward jumps indicate loops (while loops)
        if self.profiling_enabled && (offset as usize) < self.ip {
            let loop_header_ip = offset as usize;

            // Register while loop dynamically
            if !self.hotspot_detector.is_loop_header(loop_header_ip) {
                self.hotspot_detector.register_loop(
                    loop_header_ip,
                    self.ip - 1,
                    self.ip,
                );
            }

            if self.hotspot_detector.record_loop_iteration(loop_header_ip) {
                debug!("Hot while loop detected at IP {}", loop_header_ip);
            }
        }

        self.ip = offset as usize;
    }

    #[inline(always)]
    pub(crate) fn handle_jump_if_false(&mut self, offset: u32, span: Span) -> WalrusResult<()> {
        let value = self.pop(Opcode::JumpIfFalse(offset), span)?;

        if let Value::Bool(false) = value {
            self.ip = offset as usize;
        }
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_call(&mut self, args: u32, span: Span) -> WalrusResult<CallResult> {
        let arg_count = args as usize;
        let func = self.pop(Opcode::Call(args), span)?;
        let args = self.pop_n(arg_count, Opcode::Call(args), span)?;

        // JIT PROFILING: Track function calls and argument types
        if self.profiling_enabled {
            if let Value::Function(key) = func {
                if let Ok(func_ref) = self.get_heap().get_function(key) {
                    let name = match func_ref {
                        WalrusFunction::Vm(f) => f.name.clone(),
                        WalrusFunction::Rust(f) => f.name.clone(),
                        _ => String::new(),
                    };
                    if !name.is_empty() {
                        if self.hotspot_detector.record_function_call(&name) {
                            debug!("Hot function detected: {}", name);
                        }
                        // Track argument types
                        for (i, arg) in args.iter().enumerate() {
                            let arg_type = WalrusType::from_value(arg);
                            self.type_profile.observe(self.ip - 1 + i, arg_type);
                        }
                    }
                }
            }
        }

        match func {
            Value::Function(key) => {
                let func = self.get_heap().get_function(key)?;

                match func {
                    WalrusFunction::Rust(func) => {
                        if args.len() != func.args {
                            return Err(WalrusError::InvalidArgCount {
                                name: func.name.clone(),
                                expected: func.args,
                                got: args.len(),
                                span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            });
                        }
                        let result = func.call(args, self.source_ref, span)?;
                        self.push(result);
                    }
                    WalrusFunction::Native(native_fn) => {
                        let result = self.call_native(*native_fn, args, span)?;
                        self.push(result);
                    }
                    WalrusFunction::Vm(func) => {
                        if args.len() != func.arity {
                            return Err(WalrusError::InvalidArgCount {
                                name: func.name.clone(),
                                expected: func.arity,
                                got: args.len(),
                                span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            });
                        }

                        let new_frame = CallFrame {
                            return_ip: self.ip,
                            frame_pointer: self.locals.len(),
                            stack_pointer: self.stack.len(),
                            instructions: Rc::clone(&func.code),
                            function_name: format!("fn<{}>", func.name),
                        };

                        self.call_stack.push(new_frame);

                        for arg in args {
                            self.locals.push(arg);
                        }

                        self.ip = 0;
                    }
                    _ => {
                        return Err(WalrusError::NodeFunctionNotSupportedInVm {
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
        Ok(CallResult::Continue)
    }

    #[inline(always)]
    pub(crate) fn handle_tail_call(&mut self, args: u32, span: Span) -> WalrusResult<CallResult> {
        let arg_count = args as usize;
        let func = self.pop(Opcode::TailCall(args), span)?;
        let args = self.pop_n(arg_count, Opcode::TailCall(args), span)?;

        match func {
            Value::Function(key) => {
                let func = self.get_heap().get_function(key)?;

                match func {
                    WalrusFunction::Rust(func) => {
                        if args.len() != func.args {
                            return Err(WalrusError::InvalidArgCount {
                                name: func.name.clone(),
                                expected: func.args,
                                got: args.len(),
                                span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            });
                        }
                        let result = func.call(args, self.source_ref, span)?;

                        let frame = self
                            .call_stack
                            .pop()
                            .expect("Call stack should never be empty on tail call");

                        if self.call_stack.is_empty() {
                            return Ok(CallResult::Return(result));
                        }

                        self.locals.truncate(frame.frame_pointer);
                        self.ip = frame.return_ip;
                        self.push(result);
                    }
                    WalrusFunction::Native(native_fn) => {
                        let result = self.call_native(*native_fn, args, span)?;

                        let frame = self
                            .call_stack
                            .pop()
                            .expect("Call stack should never be empty on tail call");

                        if self.call_stack.is_empty() {
                            return Ok(CallResult::Return(result));
                        }

                        self.locals.truncate(frame.frame_pointer);
                        self.ip = frame.return_ip;
                        self.push(result);
                    }
                    WalrusFunction::Vm(func) => {
                        if args.len() != func.arity {
                            return Err(WalrusError::InvalidArgCount {
                                name: func.name.clone(),
                                expected: func.arity,
                                got: args.len(),
                                span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            });
                        }

                        let new_instructions = Rc::clone(&func.code);
                        let new_name = format!("fn<{}>", func.name);

                        let frame_pointer = self.frame_pointer();
                        self.locals.truncate(frame_pointer);

                        for arg in args {
                            self.locals.push(arg);
                        }

                        if let Some(current_frame) = self.call_stack.last_mut() {
                            current_frame.instructions = new_instructions;
                            current_frame.function_name = new_name;
                        }

                        self.ip = 0;
                    }
                    _ => {
                        return Err(WalrusError::NodeFunctionNotSupportedInVm {
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        });
                    }
                }
            }
            _ => {
                return Err(WalrusError::NotCallable {
                    value: func.get_type().to_string(),
                    span,
                    src: self.source_ref.source().into(),
                    filename: self.source_ref.filename().into(),
                });
            }
        }
        Ok(CallResult::Continue)
    }

    #[inline(always)]
    pub(crate) fn handle_return(&mut self, span: Span) -> WalrusResult<CallResult> {
        let return_value = self.pop(Opcode::Return, span)?;

        let frame = self
            .call_stack
            .pop()
            .expect("Call stack should never be empty on return");

        if self.call_stack.is_empty() {
            return Ok(CallResult::Return(return_value));
        }

        self.locals.truncate(frame.frame_pointer);
        self.stack.truncate(frame.stack_pointer);
        self.ip = frame.return_ip;

        self.push(return_value);
        Ok(CallResult::Continue)
    }
}
