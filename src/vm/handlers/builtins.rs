//! Runtime helper handlers: Print, Println, Str, Import

use rustc_hash::FxHashMap;

use crate::WalrusResult;
use crate::arenas::HeapValue;
use crate::error::WalrusError;
use crate::function::WalrusFunction;
use crate::span::Span;
use crate::value::Value;
use crate::vm::VM;
use crate::vm::opcode::Opcode;

impl<'a> VM<'a> {
    #[inline(always)]
    pub(crate) fn handle_print(&mut self, span: Span) -> WalrusResult<()> {
        let a = self.pop(Opcode::Print, span)?;
        print!("{}", self.stringify_value(a)?);
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_println(&mut self, span: Span) -> WalrusResult<()> {
        let a = self.pop(Opcode::Println, span)?;
        println!("{}", self.stringify_value(a)?);
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_str(&mut self, span: Span) -> WalrusResult<()> {
        let a = self.pop(Opcode::Str, span)?;
        let s = self.stringify_value(a)?;
        let value = self.get_heap_mut().push(HeapValue::String(&s));
        self.push(value);
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_import(&mut self, span: Span) -> WalrusResult<()> {
        let module_name = self.pop(Opcode::Import, span)?;
        match module_name {
            Value::String(name_key) => {
                let name_str = self.get_heap().get_string(name_key)?;
                if let Some(functions) = crate::stdlib::get_module_functions(name_str) {
                    let mut dict = FxHashMap::default();
                    for native_fn in functions {
                        let key = self
                            .get_heap_mut()
                            .push(HeapValue::String(native_fn.name()));
                        let func = self
                            .get_heap_mut()
                            .push(HeapValue::Function(WalrusFunction::Native(native_fn)));
                        dict.insert(key, func);
                    }
                    let module = self.get_heap_mut().push(HeapValue::Module(dict));
                    self.push(module);
                    Ok(())
                } else {
                    Err(WalrusError::ModuleNotFound {
                        module: name_str.to_string(),
                        span,
                        src: self.source_ref.source().into(),
                        filename: self.source_ref.filename().into(),
                    })
                }
            }
            _ => Err(WalrusError::TypeMismatch {
                expected: "string".to_string(),
                found: module_name.get_type().to_string(),
                span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            }),
        }
    }
}
