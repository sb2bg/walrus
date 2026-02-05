//! Builtin operation handlers: Print, Println, Len, Str, Type, Gc, HeapStats, Import

use rustc_hash::FxHashMap;

use crate::WalrusResult;
use crate::arenas::HeapValue;
use crate::error::WalrusError;
use crate::function::WalrusFunction;
use crate::span::Span;
use crate::value::Value;
use crate::vm::opcode::Opcode;
use crate::vm::VM;

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
    pub(crate) fn handle_len(&mut self, span: Span) -> WalrusResult<()> {
        let a = self.pop(Opcode::Len, span)?;

        match a {
            Value::String(key) => {
                let s = self.get_heap().get_string(key)?;
                self.push(Value::Int(s.len() as i64));
            }
            Value::List(key) => {
                let list = self.get_heap().get_list(key)?;
                self.push(Value::Int(list.len() as i64));
            }
            Value::Dict(key) => {
                let dict = self.get_heap().get_dict(key)?;
                self.push(Value::Int(dict.len() as i64));
            }
            _ => {
                return Err(WalrusError::NoLength {
                    type_name: a.get_type().to_string(),
                    span,
                    src: self.source_ref.source().to_string(),
                    filename: self.source_ref.filename().to_string(),
                });
            }
        }
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
    pub(crate) fn handle_type(&mut self, span: Span) -> WalrusResult<()> {
        let a = self.pop(Opcode::Type, span)?;
        let type_name = a.get_type();
        let value = self.get_heap_mut().push(HeapValue::String(type_name));
        self.push(value);
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_gc(&mut self) {
        let roots = self.collect_roots();
        let result = self.get_heap_mut().force_collect(&roots);

        let mut dict = FxHashMap::default();
        let heap = self.get_heap_mut();

        let key_freed = heap.push(HeapValue::String("objects_freed"));
        let key_before = heap.push(HeapValue::String("objects_before"));
        let key_after = heap.push(HeapValue::String("objects_after"));
        let key_collections = heap.push(HeapValue::String("total_collections"));

        dict.insert(key_freed, Value::Int(result.objects_freed as i64));
        dict.insert(key_before, Value::Int(result.objects_before as i64));
        dict.insert(key_after, Value::Int(result.objects_after as i64));
        dict.insert(key_collections, Value::Int(result.collections_total as i64));

        let result_dict = self.get_heap_mut().push(HeapValue::Dict(dict));
        self.push(result_dict);
    }

    #[inline(always)]
    pub(crate) fn handle_heap_stats(&mut self) {
        let stats = self.get_heap().heap_stats();
        let gc_info = self.get_heap().gc_stats();

        let mut dict = FxHashMap::default();
        let heap = self.get_heap_mut();

        let key_lists = heap.push(HeapValue::String("lists"));
        let key_tuples = heap.push(HeapValue::String("tuples"));
        let key_dicts = heap.push(HeapValue::String("dicts"));
        let key_functions = heap.push(HeapValue::String("functions"));
        let key_iterators = heap.push(HeapValue::String("iterators"));
        let key_struct_defs = heap.push(HeapValue::String("struct_defs"));
        let key_struct_insts = heap.push(HeapValue::String("struct_instances"));
        let key_total = heap.push(HeapValue::String("total_objects"));

        let key_alloc_count = heap.push(HeapValue::String("allocation_count"));
        let key_bytes = heap.push(HeapValue::String("bytes_allocated"));
        let key_bytes_freed = heap.push(HeapValue::String("total_bytes_freed"));
        let key_collections = heap.push(HeapValue::String("total_collections"));
        let key_threshold = heap.push(HeapValue::String("allocation_threshold"));
        let key_mem_threshold = heap.push(HeapValue::String("memory_threshold"));

        dict.insert(key_lists, Value::Int(stats.lists as i64));
        dict.insert(key_tuples, Value::Int(stats.tuples as i64));
        dict.insert(key_dicts, Value::Int(stats.dicts as i64));
        dict.insert(key_functions, Value::Int(stats.functions as i64));
        dict.insert(key_iterators, Value::Int(stats.iterators as i64));
        dict.insert(key_struct_defs, Value::Int(stats.struct_defs as i64));
        dict.insert(key_struct_insts, Value::Int(stats.struct_instances as i64));
        dict.insert(key_total, Value::Int(stats.total_objects() as i64));

        dict.insert(key_alloc_count, Value::Int(gc_info.allocation_count as i64));
        dict.insert(key_bytes, Value::Int(gc_info.bytes_allocated as i64));
        dict.insert(
            key_bytes_freed,
            Value::Int(gc_info.total_bytes_freed as i64),
        );
        dict.insert(
            key_collections,
            Value::Int(gc_info.total_collections as i64),
        );
        dict.insert(
            key_threshold,
            Value::Int(gc_info.allocation_threshold as i64),
        );
        dict.insert(
            key_mem_threshold,
            Value::Int(gc_info.memory_threshold as i64),
        );

        let result_dict = self.get_heap_mut().push(HeapValue::Dict(dict));
        self.push(result_dict);
    }

    #[inline(always)]
    pub(crate) fn handle_gc_config(&mut self, span: Span) -> WalrusResult<()> {
        let threshold = self.pop(Opcode::GcConfig, span)?;
        match threshold {
            Value::Int(n) if n > 0 => {
                let old = crate::gc::set_allocation_threshold(n as usize);
                self.push(Value::Int(old as i64));
                Ok(())
            }
            _ => Err(WalrusError::InvalidGcThresholdArg {
                span,
                src: self.source_ref.source().to_string(),
                filename: self.source_ref.filename().to_string(),
            }),
        }
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
                        let func = self.get_heap_mut().push(HeapValue::Function(
                            WalrusFunction::Native(native_fn),
                        ));
                        dict.insert(key, func);
                    }
                    let module = self.get_heap_mut().push(HeapValue::Dict(dict));
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
