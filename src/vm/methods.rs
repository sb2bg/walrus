//! Built-in methods for Walrus types.
//!
//! This module defines methods that can be called on built-in types like lists,
//! strings, and dicts. To add a new method:
//!
//! 1. Add the method to the appropriate `dispatch_*_method` function
//! 2. Implement the method logic
//!
//! Example: Adding a `clear` method to lists:
//! ```rust
//! "clear" => {
//!     check_arity("clear", 0, args.len(), span, src, filename)?;
//!     list.clear();
//!     Ok(Value::Void)
//! }
//! ```

use rustc_hash::FxHashMap;

use crate::WalrusResult;
use crate::arenas::{DictKey, HeapValue, ListKey, StringKey, ValueHolder};
use crate::error::WalrusError;
use crate::span::Span;
use crate::value::Value;

/// Check that the correct number of arguments were passed to a method.
fn check_arity(
    method: &str,
    expected: usize,
    got: usize,
    span: Span,
    src: &str,
    filename: &str,
) -> WalrusResult<()> {
    if got != expected {
        Err(WalrusError::InvalidArgCount {
            name: method.to_string(),
            expected,
            got,
            span,
            src: src.into(),
            filename: filename.into(),
        })
    } else {
        Ok(())
    }
}

/// Dispatch a method call on a list.
pub fn dispatch_list_method(
    heap: &mut ValueHolder,
    key: ListKey,
    method: &str,
    args: Vec<Value>,
    span: Span,
    src: &str,
    filename: &str,
) -> WalrusResult<Value> {
    match method {
        "push" => {
            check_arity("push", 1, args.len(), span, src, filename)?;
            let list = heap.get_mut_list(key)?;
            list.push(args[0]);
            Ok(Value::Void)
        }
        "pop" => {
            check_arity("pop", 0, args.len(), span, src, filename)?;
            let list = heap.get_mut_list(key)?;
            list.pop().ok_or_else(|| WalrusError::EmptyListPop {
                span,
                src: src.to_string(),
                filename: filename.to_string(),
            })
        }
        "len" => {
            check_arity("len", 0, args.len(), span, src, filename)?;
            let list = heap.get_list(key)?;
            Ok(Value::Int(list.len() as i64))
        }
        "clear" => {
            check_arity("clear", 0, args.len(), span, src, filename)?;
            let list = heap.get_mut_list(key)?;
            list.clear();
            Ok(Value::Void)
        }
        "reverse" => {
            check_arity("reverse", 0, args.len(), span, src, filename)?;
            let list = heap.get_mut_list(key)?;
            list.reverse();
            Ok(Value::Void)
        }
        "contains" => {
            check_arity("contains", 1, args.len(), span, src, filename)?;
            let list = heap.get_list(key)?;
            let found = list.contains(&args[0]);
            Ok(Value::Bool(found))
        }
        "insert" => {
            check_arity("insert", 2, args.len(), span, src, filename)?;
            let index = match args[0] {
                Value::Int(i) => i,
                _ => {
                    return Err(WalrusError::TypeMismatch {
                        expected: "int".to_string(),
                        found: args[0].get_type().to_string(),
                        span,
                        src: src.into(),
                        filename: filename.into(),
                    });
                }
            };
            let list = heap.get_mut_list(key)?;
            let len = list.len() as i64;
            let index = if index < 0 { len + index } else { index };
            if index < 0 || index > len {
                return Err(WalrusError::IndexOutOfBounds {
                    index,
                    len: len as usize,
                    span,
                    src: src.into(),
                    filename: filename.into(),
                });
            }
            list.insert(index as usize, args[1]);
            Ok(Value::Void)
        }
        "remove" => {
            check_arity("remove", 1, args.len(), span, src, filename)?;
            let index = match args[0] {
                Value::Int(i) => i,
                _ => {
                    return Err(WalrusError::TypeMismatch {
                        expected: "int".to_string(),
                        found: args[0].get_type().to_string(),
                        span,
                        src: src.into(),
                        filename: filename.into(),
                    });
                }
            };
            let list = heap.get_mut_list(key)?;
            let len = list.len() as i64;
            let index = if index < 0 { len + index } else { index };
            if index < 0 || index >= len {
                return Err(WalrusError::IndexOutOfBounds {
                    index,
                    len: len as usize,
                    span,
                    src: src.into(),
                    filename: filename.into(),
                });
            }
            let removed = list.remove(index as usize);
            Ok(removed)
        }
        _ => Err(WalrusError::MethodNotFound {
            type_name: "list".to_string(),
            method: method.to_string(),
            span,
            src: src.into(),
            filename: filename.into(),
        }),
    }
}

/// Dispatch a method call on a string.
pub fn dispatch_string_method(
    heap: &mut ValueHolder,
    key: StringKey,
    method: &str,
    args: Vec<Value>,
    span: Span,
    src: &str,
    filename: &str,
) -> WalrusResult<Value> {
    match method {
        "len" => {
            check_arity("len", 0, args.len(), span, src, filename)?;
            let s = heap.get_string(key)?;
            Ok(Value::Int(s.len() as i64))
        }
        "upper" => {
            check_arity("upper", 0, args.len(), span, src, filename)?;
            let s = heap.get_string(key)?;
            let upper = s.to_uppercase();
            Ok(heap.push(HeapValue::String(&upper)))
        }
        "lower" => {
            check_arity("lower", 0, args.len(), span, src, filename)?;
            let s = heap.get_string(key)?;
            let lower = s.to_lowercase();
            Ok(heap.push(HeapValue::String(&lower)))
        }
        "trim" => {
            check_arity("trim", 0, args.len(), span, src, filename)?;
            let s = heap.get_string(key)?;
            let trimmed = s.trim().to_string();
            Ok(heap.push(HeapValue::String(&trimmed)))
        }
        "split" => {
            check_arity("split", 1, args.len(), span, src, filename)?;
            let delimiter = match args[0] {
                Value::String(sym) => heap.get_string(sym)?.to_string(),
                _ => {
                    return Err(WalrusError::TypeMismatch {
                        expected: "string".to_string(),
                        found: args[0].get_type().to_string(),
                        span,
                        src: src.into(),
                        filename: filename.into(),
                    });
                }
            };
            let s = heap.get_string(key)?.to_string();
            let parts: Vec<&str> = s.split(&delimiter).collect();
            let part_values: Vec<Value> = parts
                .into_iter()
                .map(|part| heap.push(HeapValue::String(part)))
                .collect();
            Ok(heap.push(HeapValue::List(part_values)))
        }
        "starts_with" => {
            check_arity("starts_with", 1, args.len(), span, src, filename)?;
            let prefix = match args[0] {
                Value::String(sym) => heap.get_string(sym)?.to_string(),
                _ => {
                    return Err(WalrusError::TypeMismatch {
                        expected: "string".to_string(),
                        found: args[0].get_type().to_string(),
                        span,
                        src: src.into(),
                        filename: filename.into(),
                    });
                }
            };
            let s = heap.get_string(key)?;
            Ok(Value::Bool(s.starts_with(&prefix)))
        }
        "ends_with" => {
            check_arity("ends_with", 1, args.len(), span, src, filename)?;
            let suffix = match args[0] {
                Value::String(sym) => heap.get_string(sym)?.to_string(),
                _ => {
                    return Err(WalrusError::TypeMismatch {
                        expected: "string".to_string(),
                        found: args[0].get_type().to_string(),
                        span,
                        src: src.into(),
                        filename: filename.into(),
                    });
                }
            };
            let s = heap.get_string(key)?;
            Ok(Value::Bool(s.ends_with(&suffix)))
        }
        "contains" => {
            check_arity("contains", 1, args.len(), span, src, filename)?;
            let needle = match args[0] {
                Value::String(sym) => heap.get_string(sym)?.to_string(),
                _ => {
                    return Err(WalrusError::TypeMismatch {
                        expected: "string".to_string(),
                        found: args[0].get_type().to_string(),
                        span,
                        src: src.into(),
                        filename: filename.into(),
                    });
                }
            };
            let s = heap.get_string(key)?;
            Ok(Value::Bool(s.contains(&needle)))
        }
        "replace" => {
            check_arity("replace", 2, args.len(), span, src, filename)?;
            let from = match args[0] {
                Value::String(sym) => heap.get_string(sym)?.to_string(),
                _ => {
                    return Err(WalrusError::TypeMismatch {
                        expected: "string".to_string(),
                        found: args[0].get_type().to_string(),
                        span,
                        src: src.into(),
                        filename: filename.into(),
                    });
                }
            };
            let to = match args[1] {
                Value::String(sym) => heap.get_string(sym)?.to_string(),
                _ => {
                    return Err(WalrusError::TypeMismatch {
                        expected: "string".to_string(),
                        found: args[1].get_type().to_string(),
                        span,
                        src: src.into(),
                        filename: filename.into(),
                    });
                }
            };
            let s = heap.get_string(key)?;
            let replaced = s.replace(&from, &to);
            Ok(heap.push(HeapValue::String(&replaced)))
        }
        _ => Err(WalrusError::MethodNotFound {
            type_name: "string".to_string(),
            method: method.to_string(),
            span,
            src: src.into(),
            filename: filename.into(),
        }),
    }
}

/// Dispatch a method call on a dict.
pub fn dispatch_dict_method(
    heap: &mut ValueHolder,
    key: DictKey,
    method: &str,
    args: Vec<Value>,
    span: Span,
    src: &str,
    filename: &str,
) -> WalrusResult<Value> {
    match method {
        "len" => {
            check_arity("len", 0, args.len(), span, src, filename)?;
            let dict = heap.get_dict(key)?;
            Ok(Value::Int(dict.len() as i64))
        }
        "keys" => {
            check_arity("keys", 0, args.len(), span, src, filename)?;
            let dict = heap.get_dict(key)?;
            let keys: Vec<Value> = dict.keys().copied().collect();
            Ok(heap.push(HeapValue::List(keys)))
        }
        "values" => {
            check_arity("values", 0, args.len(), span, src, filename)?;
            let dict = heap.get_dict(key)?;
            let values: Vec<Value> = dict.values().copied().collect();
            Ok(heap.push(HeapValue::List(values)))
        }
        "contains" => {
            check_arity("contains", 1, args.len(), span, src, filename)?;
            let dict = heap.get_dict(key)?;
            let found = dict.contains_key(&args[0]);
            Ok(Value::Bool(found))
        }
        "get" => {
            check_arity("get", 2, args.len(), span, src, filename)?;
            let dict = heap.get_dict(key)?;
            match dict.get(&args[0]) {
                Some(value) => Ok(*value),
                None => Ok(args[1]), // Return default
            }
        }
        "clear" => {
            check_arity("clear", 0, args.len(), span, src, filename)?;
            let dict = heap.get_mut_dict(key)?;
            dict.clear();
            Ok(Value::Void)
        }
        "remove" => {
            check_arity("remove", 1, args.len(), span, src, filename)?;
            let dict = heap.get_mut_dict(key)?;
            match dict.remove(&args[0]) {
                Some(value) => Ok(value),
                None => Ok(Value::Void),
            }
        }
        _ => Err(WalrusError::MethodNotFound {
            type_name: "dict".to_string(),
            method: method.to_string(),
            span,
            src: src.into(),
            filename: filename.into(),
        }),
    }
}
