use float_ord::FloatOrd;
use rustc_hash::FxHashMap;

use crate::WalrusResult;
use crate::arenas::{DictValue, HeapValue};
use crate::error::WalrusError;
use crate::span::Span;
use crate::value::Value;
use crate::vm::VM;

pub(crate) fn value_sequence(vm: &VM<'_>, value: Value, span: Span) -> WalrusResult<Vec<Value>> {
    match value {
        Value::List(key) => Ok(vm.get_heap().get_list(key)?.to_vec()),
        Value::Tuple(key) => Ok(vm.get_heap().get_tuple(key)?.to_vec()),
        other => Err(WalrusError::TypeMismatch {
            expected: "list or tuple".to_string(),
            found: other.get_type().to_string(),
            span,
            src: vm.source_ref().source().into(),
            filename: vm.source_ref().filename().into(),
        }),
    }
}

pub(crate) fn value_sequence_or_void(
    vm: &VM<'_>,
    value: Value,
    span: Span,
) -> WalrusResult<Vec<Value>> {
    if matches!(value, Value::Void) {
        return Ok(Vec::new());
    }
    value_sequence(vm, value, span)
}

pub(crate) fn non_negative_millis(
    vm: &VM<'_>,
    value: Value,
    span: Span,
    name: &str,
) -> WalrusResult<u64> {
    let ms = vm.value_to_int(value, span)?;
    if ms < 0 {
        return Err(WalrusError::GenericError {
            message: format!("{name} must be >= 0"),
        });
    }
    Ok(ms as u64)
}

pub(crate) fn task_key_from_value(
    vm: &VM<'_>,
    value: Value,
    span: Span,
) -> WalrusResult<crate::arenas::TaskKey> {
    match value {
        Value::Task(task_key) => Ok(task_key),
        other => Err(WalrusError::TypeMismatch {
            expected: "task".to_string(),
            found: other.get_type().to_string(),
            span,
            src: vm.source_ref().source().into(),
            filename: vm.source_ref().filename().into(),
        }),
    }
}

pub(crate) trait DictSink {
    fn insert_value(&mut self, key: Value, value: Value);
}

impl DictSink for DictValue {
    fn insert_value(&mut self, key: Value, value: Value) {
        self.insert(key, value);
    }
}

impl DictSink for FxHashMap<Value, Value> {
    fn insert_value(&mut self, key: Value, value: Value) {
        self.insert(key, value);
    }
}

pub(crate) fn insert_key_value(vm: &mut VM<'_>, dict: &mut impl DictSink, key: &str, value: Value) {
    let key_value = vm.get_heap_mut().push(HeapValue::String(key));
    dict.insert_value(key_value, value);
}

pub(crate) fn heap_string(vm: &mut VM<'_>, text: &str) -> Value {
    vm.get_heap_mut().push(HeapValue::String(text))
}

pub(crate) fn type_mismatch_error(
    vm: &VM<'_>,
    expected: &str,
    found: Value,
    span: Span,
) -> WalrusError {
    WalrusError::TypeMismatch {
        expected: expected.to_string(),
        found: found.get_type().to_string(),
        span,
        src: vm.source_ref().source().to_string(),
        filename: vm.source_ref().filename().to_string(),
    }
}

pub(crate) fn http_error_dict(vm: &mut VM<'_>, message: &str) -> Value {
    let mut dict = FxHashMap::default();
    insert_key_value(vm, &mut dict, "ok", Value::Bool(false));
    let message_value = heap_string(vm, message);
    insert_key_value(vm, &mut dict, "error", message_value);
    vm.get_heap_mut().push(HeapValue::Dict(dict))
}

pub(crate) fn value_to_serde_json(vm: &VM<'_>, value: Value) -> Result<serde_json::Value, String> {
    match value {
        Value::Int(n) => Ok(serde_json::Value::Number(n.into())),
        Value::Float(f) => {
            let f = f.0;
            serde_json::Number::from_f64(f)
                .map(serde_json::Value::Number)
                .ok_or_else(|| "json.encode: cannot encode NaN or Infinity".to_string())
        }
        Value::Bool(b) => Ok(serde_json::Value::Bool(b)),
        Value::Void => Ok(serde_json::Value::Null),
        Value::String(key) => {
            let s = vm.get_heap().get_string(key).map_err(|e| format!("{e}"))?;
            Ok(serde_json::Value::String(s.to_string()))
        }
        Value::List(key) => {
            let items = vm
                .get_heap()
                .get_list(key)
                .map_err(|e| format!("{e}"))?
                .clone();
            let arr: Result<Vec<_>, _> =
                items.iter().map(|v| value_to_serde_json(vm, *v)).collect();
            Ok(serde_json::Value::Array(arr?))
        }
        Value::Tuple(key) => {
            let items = vm
                .get_heap()
                .get_tuple(key)
                .map_err(|e| format!("{e}"))?
                .clone();
            let arr: Result<Vec<_>, _> =
                items.iter().map(|v| value_to_serde_json(vm, *v)).collect();
            Ok(serde_json::Value::Array(arr?))
        }
        Value::Dict(key) => {
            let dict = vm
                .get_heap()
                .get_dict(key)
                .map_err(|e| format!("{e}"))?
                .clone();
            let mut map = serde_json::Map::new();
            for (k, v) in &dict {
                if let Value::String(sk) = k {
                    let key_str = vm.get_heap().get_string(*sk).map_err(|e| format!("{e}"))?;
                    map.insert(key_str.to_string(), value_to_serde_json(vm, *v)?);
                } else {
                    return Err(format!(
                        "json.encode: dict keys must be strings, found '{}'",
                        k.get_type()
                    ));
                }
            }
            Ok(serde_json::Value::Object(map))
        }
        other => Err(format!(
            "json.encode: cannot encode type '{}'",
            other.get_type()
        )),
    }
}

pub(crate) fn serde_json_to_value(vm: &mut VM<'_>, json: serde_json::Value) -> Value {
    match json {
        serde_json::Value::Null => Value::Void,
        serde_json::Value::Bool(b) => Value::Bool(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else {
                Value::Float(FloatOrd(n.as_f64().unwrap_or(0.0)))
            }
        }
        serde_json::Value::String(s) => heap_string(vm, &s),
        serde_json::Value::Array(arr) => {
            let items: Vec<Value> = arr
                .into_iter()
                .map(|v| serde_json_to_value(vm, v))
                .collect();
            vm.get_heap_mut().push(HeapValue::List(items))
        }
        serde_json::Value::Object(map) => {
            let mut dict = FxHashMap::default();
            for (k, v) in map {
                let key_val = heap_string(vm, &k);
                let val = serde_json_to_value(vm, v);
                dict.insert(key_val, val);
            }
            vm.get_heap_mut().push(HeapValue::Dict(dict))
        }
    }
}
