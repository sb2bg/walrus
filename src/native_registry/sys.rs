use crate::WalrusResult;
use crate::arenas::HeapValue;
use crate::function::NativeFunction;
use crate::span::Span;
use crate::value::Value;
use crate::vm::VM;

use super::{NativeSpec, define_native_spec};

pub static SPECS: &[NativeSpec] = &[
    define_native_spec(
        NativeFunction::EnvGet,
        "std/sys",
        "env_get",
        &["name"],
        "Read an environment variable; returns void when unset.",
        native_env_get,
    ),
    define_native_spec(
        NativeFunction::Args,
        "std/sys",
        "args",
        &[],
        "Return command-line arguments.",
        native_args,
    ),
    define_native_spec(
        NativeFunction::Cwd,
        "std/sys",
        "cwd",
        &[],
        "Return the current working directory.",
        native_cwd,
    ),
    define_native_spec(
        NativeFunction::Exit,
        "std/sys",
        "exit",
        &["code"],
        "Exit the process immediately with the given code.",
        native_exit,
    ),
];

fn native_env_get(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let name = vm.value_to_string(args[0], span)?;
    match crate::stdlib::env_get(&name) {
        Some(value) => Ok(vm.get_heap_mut().push(HeapValue::String(&value))),
        None => Ok(Value::Void),
    }
}

fn native_args(vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    let cli_args = crate::stdlib::args();
    let mut list = Vec::with_capacity(cli_args.len());
    for arg in cli_args {
        let s = vm.get_heap_mut().push(HeapValue::String(&arg));
        list.push(s);
    }
    Ok(vm.get_heap_mut().push(HeapValue::List(list)))
}

fn native_cwd(vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    match crate::stdlib::cwd() {
        Some(path) => Ok(vm.get_heap_mut().push(HeapValue::String(&path))),
        None => Ok(Value::Void),
    }
}

fn native_exit(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let code = vm.value_to_int(args[0], span)?;
    std::process::exit(code as i32);
}
