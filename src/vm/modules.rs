use super::*;

impl<'a> VM<'a> {
    pub fn new_with_module_binding(
        source_ref: SourceRef<'a>,
        is: InstructionSet,
        module_key: DictKey,
    ) -> WalrusResult<Self> {
        let mut vm = Self::new(source_ref, is);
        let binding = vm.build_module_binding(module_key)?;
        vm.call_stack
            .last_mut()
            .expect("main frame should exist")
            .module_binding = Some(binding);
        Ok(vm)
    }

    #[inline(always)]
    pub(super) fn current_module_binding(&self) -> Option<Rc<VmModuleBinding>> {
        self.call_stack
            .last()
            .and_then(|frame| frame.module_binding.clone())
    }

    pub(super) fn build_module_binding(
        &mut self,
        module_key: DictKey,
    ) -> WalrusResult<Rc<VmModuleBinding>> {
        let global_names = self.global_names.clone();
        let mut global_values = Vec::with_capacity(global_names.len());

        for name in &global_names {
            let key = Value::String(self.get_heap_mut().push_ident(name));
            let value = self
                .get_heap()
                .get_module(module_key)?
                .get(&key)
                .copied()
                .unwrap_or(Value::Void);
            global_values.push(value);
        }

        Ok(Rc::new(VmModuleBinding {
            module_key,
            global_names: Rc::new(global_names),
            global_values: Rc::new(global_values),
            source: Rc::new(self.source_ref.source().to_string()),
            filename: Rc::new(self.source_ref.filename().to_string()),
        }))
    }

    pub(super) fn undefined_global_error(
        &self,
        index: usize,
        span: Span,
        binding: Option<&VmModuleBinding>,
    ) -> WalrusError {
        let name = binding
            .and_then(|ctx| ctx.global_names.get(index))
            .or_else(|| self.global_names.get(index))
            .cloned()
            .unwrap_or_else(|| format!("<global[{index}]>"));

        let (src, filename) = if let Some(ctx) = binding {
            (ctx.source.to_string(), ctx.filename.to_string())
        } else {
            (
                self.source_ref.source().to_string(),
                self.source_ref.filename().to_string(),
            )
        };

        WalrusError::UndefinedVariable {
            name,
            span,
            src,
            filename,
        }
    }

    pub(super) fn load_global_value(&mut self, index: usize, span: Span) -> WalrusResult<Value> {
        if let Some(binding) = self.current_module_binding() {
            let Some(name) = binding.global_names.get(index) else {
                return Err(self.undefined_global_error(index, span, Some(binding.as_ref())));
            };

            let key = Value::String(self.get_heap_mut().push_ident(name));
            let module = self.get_heap().get_module(binding.module_key)?;
            if let Some(value) = module.get(&key).copied() {
                return Ok(value);
            }

            Ok(binding
                .global_values
                .get(index)
                .copied()
                .unwrap_or(Value::Void))
        } else {
            if index < self.globals.len() {
                Ok(unsafe { *self.globals.get_unchecked(index) })
            } else {
                Err(self.undefined_global_error(index, span, None))
            }
        }
    }

    #[inline(always)]
    pub(super) fn load_global_value_fast(
        &mut self,
        index: usize,
        span: Span,
    ) -> WalrusResult<Value> {
        if self.current_frame().module_binding.is_none() {
            if index < self.globals.len() {
                Ok(unsafe { *self.globals.get_unchecked(index) })
            } else {
                Err(self.undefined_global_error(index, span, None))
            }
        } else {
            self.load_global_value(index, span)
        }
    }

    pub(super) fn store_global_value(
        &mut self,
        index: usize,
        value: Value,
        span: Span,
    ) -> WalrusResult<()> {
        if let Some(binding) = self.current_module_binding() {
            let Some(name) = binding.global_names.get(index) else {
                return Err(self.undefined_global_error(index, span, Some(binding.as_ref())));
            };

            let key = Value::String(self.get_heap_mut().push_ident(name));
            let value = self.bind_exported_value_to_module(value, &binding)?;
            self.get_heap_mut()
                .get_mut_module(binding.module_key)?
                .insert(key, value);
            Ok(())
        } else {
            if index >= self.globals.len() {
                self.globals.resize(index + 1, Value::Void);
            }
            self.globals[index] = value;
            Ok(())
        }
    }

    pub(super) fn bind_exported_value_to_module(
        &mut self,
        value: Value,
        binding: &Rc<VmModuleBinding>,
    ) -> WalrusResult<Value> {
        match value {
            Value::Function(func_key) => {
                let function = self.get_heap().get_function(func_key)?.clone();
                match function {
                    WalrusFunction::Vm(mut vm_func) => {
                        vm_func.module_binding = Some(binding.clone());
                        Ok(self
                            .get_heap_mut()
                            .push(HeapValue::Function(WalrusFunction::Vm(vm_func))))
                    }
                    _ => Ok(value),
                }
            }
            Value::StructDef(struct_key) => {
                let original = self.get_heap().get_struct_def(struct_key)?.clone();
                let mut rebound =
                    crate::structs::StructDefinition::new(original.name().to_string());

                for (method_name, method_fn) in original.methods() {
                    let bound_method = match method_fn.clone() {
                        WalrusFunction::Vm(mut vm_func) => {
                            vm_func.module_binding = Some(binding.clone());
                            WalrusFunction::Vm(vm_func)
                        }
                        other => other,
                    };
                    rebound.add_method(method_name.clone(), bound_method);
                }

                Ok(self.get_heap_mut().push(HeapValue::StructDef(rebound)))
            }
            _ => Ok(value),
        }
    }

    pub fn export_globals_as_module(&mut self) -> WalrusResult<Value> {
        let global_names = self.global_names.clone();
        let global_values = self.globals.clone();
        let mut exports = FxHashMap::default();

        for (index, name) in global_names.iter().enumerate() {
            if name == "_" {
                continue;
            }

            let value = match self.globals.get(index).copied() {
                Some(value) => value,
                None => continue,
            };

            let key = self.get_heap_mut().push(HeapValue::String(name));
            exports.insert(key, value);
        }

        let module_key = match self.get_heap_mut().push(HeapValue::Module(exports)) {
            Value::Module(key) => key,
            _ => unreachable!("module export must allocate a module"),
        };

        let binding = Rc::new(VmModuleBinding {
            module_key,
            global_names: Rc::new(global_names),
            global_values: Rc::new(global_values),
            source: Rc::new(self.source_ref.source().to_string()),
            filename: Rc::new(self.source_ref.filename().to_string()),
        });

        // Rebind exported values that contain VM functions so global accesses
        // resolve against this module, not the importing VM's global vector.
        let names = binding.global_names.clone();
        for name in names.iter() {
            if name == "_" {
                continue;
            }

            let entry_value = {
                let key = Value::String(self.get_heap_mut().push_ident(name));
                let module = self.get_heap().get_module(module_key)?;
                module.get(&key).copied()
            };

            let Some(entry_value) = entry_value else {
                continue;
            };

            let bound_value = self.bind_exported_value_to_module(entry_value, &binding)?;

            let key = Value::String(self.get_heap_mut().push_ident(name));
            self.get_heap_mut()
                .get_mut_module(module_key)?
                .insert(key, bound_value);
        }

        Ok(Value::Module(module_key))
    }
}
