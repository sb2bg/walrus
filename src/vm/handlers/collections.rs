//! Collection handlers: List, Dict, Range, Index, StoreIndex

use rustc_hash::FxHashMap;

use crate::arenas::HeapValue;
use crate::error::WalrusError;
use crate::range::RangeValue;
use crate::span::Span;
use crate::value::Value;
use crate::vm::opcode::Opcode;
use crate::vm::VM;
use crate::WalrusResult;

impl<'a> VM<'a> {
    #[inline(always)]
    pub(crate) fn handle_list(&mut self, cap: u32, span: Span) -> WalrusResult<()> {
        let cap = cap as usize;

        if self.stack.len() < cap {
            return Err(WalrusError::StackUnderflow {
                op: Opcode::List(cap as u32),
                span,
                src: self.source_ref.source().to_string(),
                filename: self.source_ref.filename().to_string(),
            });
        }

        let list = self.stack.split_off(self.stack.len() - cap);
        let value = self.get_heap_mut().push(HeapValue::List(list));
        self.push(value);
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_dict(&mut self, cap: u32, span: Span) -> WalrusResult<()> {
        let cap = cap as usize;
        let mut dict = FxHashMap::with_capacity_and_hasher(cap, Default::default());

        for _ in 0..cap {
            let value = self.pop(Opcode::Dict(cap as u32), span)?;
            let key = self.pop(Opcode::Dict(cap as u32), span)?;
            dict.insert(key, value);
        }

        let value = self.get_heap_mut().push(HeapValue::Dict(dict));
        self.push(value);
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_range(&mut self, span: Span) -> WalrusResult<()> {
        let left = self.pop(Opcode::Range, span)?;
        let right = self.pop(Opcode::Range, span)?;

        match (left, right) {
            (Value::Void, Value::Void) => {
                self.push(Value::Range(RangeValue::new(0, span, -1, span)));
            }
            (Value::Void, Value::Int(right)) => {
                self.push(Value::Range(RangeValue::new(0, span, right, span)));
            }
            (Value::Int(left), Value::Void) => {
                self.push(Value::Range(RangeValue::new(left, span, -1, span)));
            }
            (Value::Int(left), Value::Int(right)) => {
                self.push(Value::Range(RangeValue::new(left, span, right, span)));
            }
            (left, right) => {
                return Err(WalrusError::TypeMismatch {
                    expected: "type: todo".to_string(),
                    found: format!("{} and {}", left.get_type(), right.get_type()),
                    span,
                    src: self.source_ref.source().into(),
                    filename: self.source_ref.filename().into(),
                });
            }
        }
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_index(&mut self, span: Span) -> WalrusResult<()> {
        let b = self.pop(Opcode::Index, span)?;
        let a = self.pop(Opcode::Index, span)?;

        match (a, b) {
            (Value::List(a), Value::Int(b)) => {
                let a = self.get_heap().get_list(a)?;
                let mut b = b;
                let original = b;

                if b < 0 {
                    b += a.len() as i64;
                }

                if b < 0 || b >= a.len() as i64 {
                    return Err(WalrusError::IndexOutOfBounds {
                        index: original,
                        len: a.len(),
                        span,
                        src: self.source_ref.source().to_string(),
                        filename: self.source_ref.filename().to_string(),
                    });
                }

                self.push(a[b as usize]);
            }
            (Value::String(a), Value::Int(b)) => {
                let a = self.get_heap().get_string(a)?;
                let char_len = a.chars().count();
                let original = b;

                let Some(char_idx) = Self::normalize_index(b, char_len) else {
                    return Err(WalrusError::IndexOutOfBounds {
                        index: original,
                        len: char_len,
                        span,
                        src: self.source_ref.source().to_string(),
                        filename: self.source_ref.filename().to_string(),
                    });
                };

                let res = a
                    .chars()
                    .nth(char_idx)
                    .map(|ch| ch.to_string())
                    .ok_or_else(|| WalrusError::IndexOutOfBounds {
                        index: original,
                        len: char_len,
                        span,
                        src: self.source_ref.source().to_string(),
                        filename: self.source_ref.filename().to_string(),
                    })?;
                let value = self.get_heap_mut().push(HeapValue::String(&res));

                self.push(value);
            }
            (Value::Dict(a), b) => {
                let a = self.get_heap().get_dict(a)?;

                if let Some(value) = a.get(&b) {
                    self.push(*value);
                } else {
                    let b_str = b.stringify()?;

                    return Err(WalrusError::KeyNotFound {
                        key: b_str,
                        span,
                        src: self.source_ref.source().to_string(),
                        filename: self.source_ref.filename().to_string(),
                    });
                }
            }
            (Value::String(a), Value::Range(range)) => {
                let a = self.get_heap().get_string(a)?;
                let a_len = a.chars().count();

                let start = if range.start < 0 {
                    a_len as i64 + range.start + 1
                } else {
                    range.start
                };

                let end = if range.end < 0 {
                    a_len as i64 + range.end + 1
                } else {
                    range.end
                };

                if start < 0 || end < 0 || start as usize > a_len || end as usize > a_len {
                    return Err(WalrusError::IndexOutOfBounds {
                        index: range.start,
                        len: a_len,
                        span,
                        src: self.source_ref.source().to_string(),
                        filename: self.source_ref.filename().to_string(),
                    });
                }

                if start > end {
                    return Err(WalrusError::InvalidRange {
                        start: range.start,
                        end: range.end,
                        span,
                        src: self.source_ref.source().to_string(),
                        filename: self.source_ref.filename().to_string(),
                    });
                }

                let start = start as usize;
                let end = end as usize;

                let start_byte = Self::char_to_byte_offset(a, start).ok_or_else(|| {
                    WalrusError::IndexOutOfBounds {
                        index: range.start,
                        len: a_len,
                        span,
                        src: self.source_ref.source().to_string(),
                        filename: self.source_ref.filename().to_string(),
                    }
                })?;
                let end_byte = Self::char_to_byte_offset(a, end).ok_or_else(|| {
                    WalrusError::IndexOutOfBounds {
                        index: range.end,
                        len: a_len,
                        span,
                        src: self.source_ref.source().to_string(),
                        filename: self.source_ref.filename().to_string(),
                    }
                })?;

                let res = a[start_byte..end_byte].to_string();
                let value = self.get_heap_mut().push(HeapValue::String(&res));

                self.push(value);
            }
            (Value::List(a), Value::Range(range)) => {
                let a = self.get_heap().get_list(a)?;
                let a_len = a.len();

                let start = if range.start < 0 {
                    a_len as i64 + range.start + 1
                } else {
                    range.start
                };

                let end = if range.end < 0 {
                    a_len as i64 + range.end + 1
                } else {
                    range.end
                };

                if start < 0 || end < 0 || start as usize > a_len || end as usize > a_len {
                    return Err(WalrusError::IndexOutOfBounds {
                        index: range.start,
                        len: a.len(),
                        span,
                        src: self.source_ref.source().to_string(),
                        filename: self.source_ref.filename().to_string(),
                    });
                }

                if start > end {
                    return Err(WalrusError::InvalidRange {
                        start: range.start,
                        end: range.end,
                        span,
                        src: self.source_ref.source().to_string(),
                        filename: self.source_ref.filename().to_string(),
                    });
                }

                let res = a[start as usize..end as usize].to_vec();
                let value = self.get_heap_mut().push(HeapValue::List(res));

                self.push(value);
            }
            _ => return Err(self.construct_err(Opcode::Index, a, Some(b), span)),
        }
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn handle_store_index(&mut self, span: Span) -> WalrusResult<()> {
        let value = self.pop(Opcode::StoreIndex, span)?;
        let index = self.pop(Opcode::StoreIndex, span)?;
        let object = self.pop(Opcode::StoreIndex, span)?;

        match (object, index) {
            (Value::List(list_key), Value::Int(idx)) => {
                let list = self.get_heap_mut().get_mut_list(list_key)?;
                let mut idx = idx;
                let original = idx;

                if idx < 0 {
                    idx += list.len() as i64;
                }

                if idx < 0 || idx >= list.len() as i64 {
                    return Err(WalrusError::IndexOutOfBounds {
                        index: original,
                        len: list.len(),
                        span,
                        src: self.source_ref.source().to_string(),
                        filename: self.source_ref.filename().to_string(),
                    });
                }

                list[idx as usize] = value;
                self.push(Value::Void);
            }
            (Value::Dict(dict_key), key) => {
                let dict = self.get_heap_mut().get_mut_dict(dict_key)?;
                dict.insert(key, value);
                self.push(Value::Void);
            }
            _ => {
                return Err(WalrusError::InvalidIndexType {
                    non_indexable: object.get_type().to_string(),
                    index_type: index.get_type().to_string(),
                    span,
                    src: self.source_ref.source().to_string(),
                    filename: self.source_ref.filename().to_string(),
                });
            }
        }
        Ok(())
    }
}
