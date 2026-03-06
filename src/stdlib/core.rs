use std::io::Write;

use rustc_hash::FxHashMap;

use crate::WalrusResult;
use crate::arenas::{HeapValue, Resolve};
use crate::error::WalrusError;
use crate::function::{RustFunction, WalrusFunction};
use crate::source_ref::SourceRef;
use crate::span::Span;
use crate::value::Value;

pub fn interpreter_module(module: &str) -> Option<Value> {
    match module {
        "std/core" => Some(interpreter_core_module()),
        _ => None,
    }
}

fn interpreter_core_module() -> Value {
    fn add(
        dict: &mut FxHashMap<Value, Value>,
        name: &'static str,
        arity: usize,
        func: fn(Vec<Value>, SourceRef, Span) -> WalrusResult<Value>,
    ) {
        let key = HeapValue::String(name).alloc();
        let val = HeapValue::Function(WalrusFunction::Rust(RustFunction::new(
            name.to_string(),
            arity,
            func,
        )))
        .alloc();
        dict.insert(key, val);
    }

    fn core_len(args: Vec<Value>, source_ref: SourceRef, span: Span) -> WalrusResult<Value> {
        match args[0] {
            Value::String(key) => Ok(Value::Int(key.resolve()?.len() as i64)),
            Value::List(key) => Ok(Value::Int(key.resolve()?.len() as i64)),
            Value::Dict(key) => Ok(Value::Int(key.resolve()?.len() as i64)),
            Value::Module(key) => Ok(Value::Int(key.resolve()?.len() as i64)),
            other => Err(WalrusError::NoLength {
                type_name: other.get_type().to_string(),
                span,
                src: source_ref.source().to_string(),
                filename: source_ref.filename().to_string(),
            }),
        }
    }

    fn core_str(args: Vec<Value>, _source_ref: SourceRef, _span: Span) -> WalrusResult<Value> {
        let rendered = args[0].stringify()?;
        Ok(HeapValue::String(&rendered).alloc())
    }

    fn core_type(args: Vec<Value>, _source_ref: SourceRef, _span: Span) -> WalrusResult<Value> {
        Ok(HeapValue::String(args[0].get_type()).alloc())
    }

    fn core_input(args: Vec<Value>, _source_ref: SourceRef, _span: Span) -> WalrusResult<Value> {
        print!("{}", args[0].stringify()?);
        std::io::stdout()
            .flush()
            .map_err(|source| WalrusError::IOError { source })?;

        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(|source| WalrusError::IOError { source })?;
        Ok(HeapValue::String(&input).alloc())
    }

    fn core_gc_unavailable(
        _args: Vec<Value>,
        _source_ref: SourceRef,
        _span: Span,
    ) -> WalrusResult<Value> {
        Err(WalrusError::GenericError {
            message: "std/core.gc is only available in VM mode".to_string(),
        })
    }

    fn core_heap_stats_unavailable(
        _args: Vec<Value>,
        _source_ref: SourceRef,
        _span: Span,
    ) -> WalrusResult<Value> {
        Err(WalrusError::GenericError {
            message: "std/core.heap_stats is only available in VM mode".to_string(),
        })
    }

    fn core_gc_threshold_unavailable(
        _args: Vec<Value>,
        _source_ref: SourceRef,
        _span: Span,
    ) -> WalrusResult<Value> {
        Err(WalrusError::GenericError {
            message: "std/core.gc_threshold is only available in VM mode".to_string(),
        })
    }

    let mut dict = FxHashMap::default();
    add(&mut dict, "len", 1, core_len);
    add(&mut dict, "str", 1, core_str);
    add(&mut dict, "type", 1, core_type);
    add(&mut dict, "input", 1, core_input);
    add(&mut dict, "gc", 0, core_gc_unavailable);
    add(&mut dict, "heap_stats", 0, core_heap_stats_unavailable);
    add(&mut dict, "gc_threshold", 1, core_gc_threshold_unavailable);

    HeapValue::Module(dict).alloc()
}
