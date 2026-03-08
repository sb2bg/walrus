use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use rustc_hash::FxHashMap;

use crate::WalrusResult;
use crate::arenas::HeapValue;
use crate::error::WalrusError;
use crate::function::NativeFunction;
use crate::span::Span;
use crate::value::Value;
use crate::vm::VM;

use super::{NativeSpec, define_native_spec};

pub static SPECS: &[NativeSpec] = &[
    define_native_spec(
        NativeFunction::CoreLen,
        "std/core",
        "len",
        &["value"],
        "Return length of a string, list, dict, or module.",
        native_core_len,
    ),
    define_native_spec(
        NativeFunction::CoreStr,
        "std/core",
        "str",
        &["value"],
        "Convert any value to a string.",
        native_core_str,
    ),
    define_native_spec(
        NativeFunction::CoreType,
        "std/core",
        "type",
        &["value"],
        "Return the runtime type name for a value.",
        native_core_type,
    ),
    define_native_spec(
        NativeFunction::CoreInput,
        "std/core",
        "input",
        &["prompt"],
        "Print a prompt and read one line from stdin.",
        native_core_input,
    ),
    define_native_spec(
        NativeFunction::CoreGc,
        "std/core",
        "gc",
        &[],
        "Trigger garbage collection and return collection stats.",
        native_core_gc,
    ),
    define_native_spec(
        NativeFunction::CoreHeapStats,
        "std/core",
        "heap_stats",
        &[],
        "Return heap and GC statistics.",
        native_core_heap_stats,
    ),
    define_native_spec(
        NativeFunction::CoreGcThreshold,
        "std/core",
        "gc_threshold",
        &["threshold"],
        "Set GC allocation threshold and return previous value.",
        native_core_gc_threshold,
    ),
    define_native_spec(
        NativeFunction::CoreTimestamp,
        "std/core",
        "timestamp",
        &[],
        "Return the current Unix epoch time in milliseconds.",
        native_core_timestamp,
    ),
];

fn native_core_len(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    match args[0] {
        Value::String(key) => {
            let s = vm.get_heap().get_string(key)?;
            Ok(Value::Int(s.len() as i64))
        }
        Value::List(key) => {
            let list = vm.get_heap().get_list(key)?;
            Ok(Value::Int(list.len() as i64))
        }
        Value::Dict(key) => {
            let dict = vm.get_heap().get_dict(key)?;
            Ok(Value::Int(dict.len() as i64))
        }
        Value::Module(key) => {
            let module = vm.get_heap().get_module(key)?;
            Ok(Value::Int(module.len() as i64))
        }
        other => Err(WalrusError::NoLength {
            type_name: other.get_type().to_string(),
            span,
            src: vm.source_ref().source().to_string(),
            filename: vm.source_ref().filename().to_string(),
        }),
    }
}

fn native_core_str(vm: &mut VM<'_>, args: &[Value], _span: Span) -> WalrusResult<Value> {
    let rendered = vm.get_heap().stringify(args[0])?;
    Ok(vm.get_heap_mut().push_string_owned(rendered))
}

fn native_core_type(vm: &mut VM<'_>, args: &[Value], _span: Span) -> WalrusResult<Value> {
    Ok(vm
        .get_heap_mut()
        .push(HeapValue::String(args[0].get_type())))
}

fn native_core_input(vm: &mut VM<'_>, args: &[Value], _span: Span) -> WalrusResult<Value> {
    let prompt = vm.get_heap().stringify(args[0])?;
    print!("{prompt}");
    std::io::stdout()
        .flush()
        .map_err(|source| WalrusError::IOError { source })?;

    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .map_err(|source| WalrusError::IOError { source })?;

    Ok(vm.get_heap_mut().push_string_owned(input))
}

fn native_core_gc(vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    let roots = vm.collect_roots();
    let result = vm.get_heap_mut().force_collect(&roots);

    let mut dict = FxHashMap::default();
    let heap = vm.get_heap_mut();

    let key_freed = heap.push(HeapValue::String("objects_freed"));
    let key_before = heap.push(HeapValue::String("objects_before"));
    let key_after = heap.push(HeapValue::String("objects_after"));
    let key_collections = heap.push(HeapValue::String("total_collections"));

    dict.insert(key_freed, Value::Int(result.objects_freed as i64));
    dict.insert(key_before, Value::Int(result.objects_before as i64));
    dict.insert(key_after, Value::Int(result.objects_after as i64));
    dict.insert(key_collections, Value::Int(result.collections_total as i64));

    Ok(vm.get_heap_mut().push(HeapValue::Dict(dict)))
}

fn native_core_heap_stats(vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    let stats = vm.get_heap().heap_stats();
    let gc_info = vm.get_heap().gc_stats();

    let mut dict = FxHashMap::default();
    let heap = vm.get_heap_mut();

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

    Ok(vm.get_heap_mut().push(HeapValue::Dict(dict)))
}

fn native_core_gc_threshold(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let n = vm.value_to_int(args[0], span)?;
    if n <= 0 {
        return Err(WalrusError::InvalidGcThresholdArg {
            span,
            src: vm.source_ref().source().to_string(),
            filename: vm.source_ref().filename().to_string(),
        });
    }

    let old = crate::gc::set_allocation_threshold(n as usize);
    Ok(Value::Int(old as i64))
}

fn native_core_timestamp(_vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;
    Ok(Value::Int(millis))
}
