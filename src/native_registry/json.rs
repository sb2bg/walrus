use crate::WalrusResult;
use crate::function::NativeFunction;
use crate::span::Span;
use crate::value::Value;
use crate::vm::VM;

use super::helpers::{heap_string, serde_json_to_value, value_to_serde_json};
use super::{NativeSpec, define_native_spec};

pub static SPECS: &[NativeSpec] = &[
    define_native_spec(
        NativeFunction::JsonEncode,
        "std/json",
        "encode",
        &["value"],
        "Encode a value into a JSON string.",
        native_json_encode,
    ),
    define_native_spec(
        NativeFunction::JsonDecode,
        "std/json",
        "decode",
        &["string"],
        "Decode a JSON string into a value.",
        native_json_decode,
    ),
];

fn native_json_encode(vm: &mut VM<'_>, args: &[Value], _span: Span) -> WalrusResult<Value> {
    let json_value = value_to_serde_json(vm, args[0])
        .map_err(|msg| crate::error::WalrusError::GenericError { message: msg })?;
    let out = serde_json::to_string(&json_value).map_err(|e| {
        crate::error::WalrusError::GenericError {
            message: format!("json.encode: {e}"),
        }
    })?;
    Ok(heap_string(vm, &out))
}

fn native_json_decode(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let input = vm.value_to_string(args[0], span)?;
    let json_value: serde_json::Value =
        serde_json::from_str(&input).map_err(|e| crate::error::WalrusError::GenericError {
            message: format!("json.decode: {e}"),
        })?;
    Ok(serde_json_to_value(vm, json_value))
}
