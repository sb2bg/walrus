//! Struct handlers: MakeStruct, GetMethod, CallMethod

use std::rc::Rc;

use crate::WalrusResult;
use crate::arenas::HeapValue;
use crate::error::WalrusError;
use crate::function::WalrusFunction;
use crate::span::Span;
use crate::value::Value;
use crate::vm::opcode::Opcode;
use crate::vm::{methods, CallFrame, VM};

/// Result of CallMethod - whether to continue normally or skip push
pub enum MethodCallResult {
    /// Method returned a value that was pushed to stack
    Pushed,
    /// Method was a VM function that set up its own frame
    FrameCreated,
}

impl<'a> VM<'a> {
    #[inline(always)]
    pub(crate) fn handle_make_struct(&mut self, span: Span) -> WalrusResult<()> {
        let struct_def_value = self.pop(Opcode::MakeStruct, span)?;

        if let Value::StructDef(struct_def_key) = struct_def_value {
            let struct_def = self.get_heap().get_struct_def(struct_def_key)?;
            let struct_name = struct_def.name().to_string();

            let instance = crate::structs::StructInstance::new(struct_name);
            let instance_value = self.get_heap_mut().push(HeapValue::StructInst(instance));

            self.push(instance_value);
            Ok(())
        } else {
            Err(WalrusError::TypeMismatch {
                expected: "struct definition".to_string(),
                found: struct_def_value.get_type().to_string(),
                span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            })
        }
    }

    #[inline(always)]
    pub(crate) fn handle_get_method(&mut self, span: Span) -> WalrusResult<()> {
        let member_name_value = self.pop(Opcode::GetMethod, span)?;
        let object_value = self.pop(Opcode::GetMethod, span)?;

        match (member_name_value, object_value) {
            (Value::String(method_name_sym), Value::StructDef(struct_def_key)) => {
                let method_name = self.get_heap().get_string(method_name_sym)?.to_string();
                let method_clone = {
                    let struct_def = self.get_heap().get_struct_def(struct_def_key)?;
                    if let Some(method) = struct_def.get_method(&method_name) {
                        method.clone()
                    } else {
                        return Err(WalrusError::MethodNotFound {
                            type_name: struct_def.name().to_string(),
                            method: method_name,
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        });
                    }
                };

                let func_value = self.get_heap_mut().push(HeapValue::Function(method_clone));
                self.push(func_value);
            }
            (Value::String(member_name_sym), Value::Dict(dict_key)) => {
                let dict = self.get_heap().get_dict(dict_key)?;
                if let Some(&value) = dict.get(&Value::String(member_name_sym)) {
                    self.push(value);
                } else {
                    let member_name = self.get_heap().get_string(member_name_sym)?;
                    return Err(WalrusError::MemberNotFound {
                        type_name: "module/dict".to_string(),
                        member: member_name.to_string(),
                        span,
                        src: self.source_ref.source().into(),
                        filename: self.source_ref.filename().into(),
                    });
                }
            }
            (member, object) => {
                return Err(WalrusError::InvalidMemberAccessTarget {
                    object_type: object.get_type().to_string(),
                    member_type: member.get_type().to_string(),
                    span,
                    src: self.source_ref.source().into(),
                    filename: self.source_ref.filename().into(),
                });
            }
        }
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_call_method(
        &mut self,
        arg_count: u32,
        span: Span,
    ) -> WalrusResult<MethodCallResult> {
        let method_name_val = self.pop(Opcode::CallMethod(arg_count), span)?;
        let method_name = match method_name_val {
            Value::String(sym) => self.get_heap().get_string(sym)?.to_string(),
            other => {
                return Err(WalrusError::TypeMismatch {
                    expected: "string".to_string(),
                    found: other.get_type().to_string(),
                    span,
                    src: self.source_ref.source().into(),
                    filename: self.source_ref.filename().into(),
                });
            }
        };

        let args = self.pop_n(arg_count as usize, Opcode::CallMethod(arg_count), span)?;
        let object = self.pop(Opcode::CallMethod(arg_count), span)?;

        let src = self.source_ref.source().to_string();
        let filename = self.source_ref.filename().to_string();

        let result = match object {
            Value::List(key) => {
                let r = methods::dispatch_list_method(
                    self.get_heap_mut(),
                    key,
                    &method_name,
                    args,
                    span,
                    &src,
                    &filename,
                )?;
                self.push(r);
                return Ok(MethodCallResult::Pushed);
            }
            Value::String(key) => {
                let r = methods::dispatch_string_method(
                    self.get_heap_mut(),
                    key,
                    &method_name,
                    args,
                    span,
                    &src,
                    &filename,
                )?;
                self.push(r);
                return Ok(MethodCallResult::Pushed);
            }
            Value::Dict(key) => {
                let method_key = self.get_heap_mut().push(HeapValue::String(&method_name));
                let dict = self.get_heap().get_dict(key)?;

                if let Some(func_val) = dict.get(&method_key).copied() {
                    if let Value::Function(func_key) = func_val {
                        let func = self.get_heap().get_function(func_key)?.clone();
                        if let WalrusFunction::Native(native) = func {
                            let result = self.call_native(native, args, span)?;
                            self.push(result);
                            return Ok(MethodCallResult::Pushed);
                        }
                    }
                }

                let r = methods::dispatch_dict_method(
                    self.get_heap_mut(),
                    key,
                    &method_name,
                    args,
                    span,
                    &src,
                    &filename,
                )?;
                self.push(r);
                return Ok(MethodCallResult::Pushed);
            }
            Value::StructDef(key) => {
                let method = {
                    let struct_def = self.get_heap().get_struct_def(key)?;
                    if let Some(method) = struct_def.get_method(&method_name) {
                        method.clone()
                    } else {
                        return Err(WalrusError::MethodNotFound {
                            type_name: struct_def.name().to_string(),
                            method: method_name.clone(),
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        });
                    }
                };

                if let WalrusFunction::Vm(func) = method {
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
                    return Ok(MethodCallResult::FrameCreated);
                } else {
                    return Err(WalrusError::StructMethodMustBeVmFunction {
                        span,
                        src: self.source_ref.source().into(),
                        filename: self.source_ref.filename().into(),
                    });
                }
            }
            other => {
                return Err(WalrusError::InvalidMethodReceiver {
                    method: method_name,
                    type_name: other.get_type().to_string(),
                    span,
                    src: self.source_ref.source().into(),
                    filename: self.source_ref.filename().into(),
                });
            }
        };
    }
}
