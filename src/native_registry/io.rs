use crate::WalrusResult;
use crate::arenas::HeapValue;
use crate::function::NativeFunction;
use crate::span::Span;
use crate::value::Value;
use crate::vm::VM;

use super::{NativeSpec, define_native_spec};

pub static SPECS: &[NativeSpec] = &[
    define_native_spec(
        NativeFunction::FileOpen,
        "std/io",
        "file_open",
        &["path", "mode"],
        "Open a file and return a handle.",
        native_file_open,
    ),
    define_native_spec(
        NativeFunction::FileRead,
        "std/io",
        "file_read",
        &["handle"],
        "Read entire contents from an open file handle.",
        native_file_read,
    ),
    define_native_spec(
        NativeFunction::FileReadLine,
        "std/io",
        "file_read_line",
        &["handle"],
        "Read one line from an open file handle.",
        native_file_read_line,
    ),
    define_native_spec(
        NativeFunction::FileWrite,
        "std/io",
        "file_write",
        &["handle", "content"],
        "Write content to an open file handle.",
        native_file_write,
    ),
    define_native_spec(
        NativeFunction::FileClose,
        "std/io",
        "file_close",
        &["handle"],
        "Close an open file handle.",
        native_file_close,
    ),
    define_native_spec(
        NativeFunction::FileExists,
        "std/io",
        "file_exists",
        &["path"],
        "Return true if a file exists at the given path.",
        native_file_exists,
    ),
    define_native_spec(
        NativeFunction::ReadFile,
        "std/io",
        "read_file",
        &["path"],
        "Read an entire file into a string.",
        native_read_file,
    ),
    define_native_spec(
        NativeFunction::WriteFile,
        "std/io",
        "write_file",
        &["path", "content"],
        "Write a full string to a file, replacing existing contents.",
        native_write_file,
    ),
];

fn native_file_open(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let path = vm.value_to_string(args[0], span)?;
    let mode = vm.value_to_string(args[1], span)?;
    crate::stdlib::file_open(&path, &mode, span)
}

fn native_file_read(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let handle = vm.value_to_int(args[0], span)?;
    let content = crate::stdlib::file_read(handle, span)?;
    Ok(vm.get_heap_mut().push(HeapValue::String(&content)))
}

fn native_file_read_line(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let handle = vm.value_to_int(args[0], span)?;
    match crate::stdlib::file_read_line(handle, span)? {
        Some(line) => Ok(vm.get_heap_mut().push(HeapValue::String(&line))),
        None => Ok(Value::Void),
    }
}

fn native_file_write(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let handle = vm.value_to_int(args[0], span)?;
    let content = vm.value_to_string(args[1], span)?;
    let bytes = crate::stdlib::file_write(handle, &content, span)?;
    Ok(Value::Int(bytes))
}

fn native_file_close(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let handle = vm.value_to_int(args[0], span)?;
    crate::stdlib::file_close(handle, span)?;
    Ok(Value::Void)
}

fn native_file_exists(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let path = vm.value_to_string(args[0], span)?;
    Ok(Value::Bool(crate::stdlib::file_exists(&path)))
}

fn native_read_file(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let path = vm.value_to_string(args[0], span)?;
    let content = crate::stdlib::read_file(&path, span)?;
    Ok(vm.get_heap_mut().push(HeapValue::String(&content)))
}

fn native_write_file(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let path = vm.value_to_string(args[0], span)?;
    let content = vm.value_to_string(args[1], span)?;
    crate::stdlib::write_file(&path, &content, span)?;
    Ok(Value::Void)
}
