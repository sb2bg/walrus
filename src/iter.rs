use std::fmt::Debug;
use std::ops::Range;

use rustc_hash::FxHashMap;

use crate::arenas::{DictKey, HeapValue, ListKey, StringKey, TupleKey, ValueHolder};
use crate::range::RangeValue;
use crate::value::Value;

pub trait ValueIterator: Debug {
    fn next(&mut self, arena: &mut ValueHolder) -> Option<Value>;
}

#[derive(Debug, Clone)]
pub struct RangeIter {
    range: Range<i64>,
}

impl RangeIter {
    pub fn new(range: RangeValue) -> Self {
        Self {
            range: range.start..range.end,
        }
    }
}

impl ValueIterator for RangeIter {
    fn next(&mut self, _: &mut ValueHolder) -> Option<Value> {
        Some(Value::Int(self.range.next()?))
    }
}

#[derive(Debug, Clone)]
pub struct StrIter {
    string: StringKey,
    byte_index: usize,
}

impl StrIter {
    pub fn new(string: StringKey) -> Self {
        Self {
            string,
            byte_index: 0,
        }
    }

    pub fn source_value(&self) -> Value {
        Value::String(self.string)
    }
}

impl ValueIterator for StrIter {
    fn next(&mut self, arena: &mut ValueHolder) -> Option<Value> {
        let (ch, next_byte_index) = {
            let s = arena.get_string(self.string).ok()?;
            if self.byte_index >= s.len() {
                return None;
            }

            let ch = s[self.byte_index..].chars().next()?;
            (ch, self.byte_index + ch.len_utf8())
        };

        self.byte_index = next_byte_index;

        // Avoid per-character String allocation by encoding into a stack buffer.
        let mut buf = [0u8; 4];
        let ch_str = ch.encode_utf8(&mut buf);
        Some(arena.push(HeapValue::String(ch_str)))
    }
}

#[derive(Debug, Clone)]
pub struct DictIter {
    dict: DictKey,
    keys: Vec<Value>,
    index: usize,
}

impl DictIter {
    pub fn new(dict_key: DictKey, dict: &FxHashMap<Value, Value>) -> Self {
        Self {
            dict: dict_key,
            keys: dict.keys().copied().collect(),
            index: 0,
        }
    }

    pub fn source_value(&self) -> Value {
        Value::Dict(self.dict)
    }
}

impl ValueIterator for DictIter {
    fn next(&mut self, arena: &mut ValueHolder) -> Option<Value> {
        while self.index < self.keys.len() {
            let key = self.keys[self.index];
            self.index += 1;

            let value = {
                let dict = arena.get_dict(self.dict).ok()?;
                dict.get(&key).copied()
            };

            if let Some(value) = value {
                return Some(arena.push(HeapValue::Tuple(&[key, value])));
            }
        }

        None
    }
}

#[derive(Debug, Clone, Copy)]
enum CollectionSource {
    List(ListKey),
    Tuple(TupleKey),
}

#[derive(Debug, Clone)]
pub struct CollectionIter {
    source: CollectionSource,
    index: usize,
}

impl CollectionIter {
    pub fn from_list(list: ListKey) -> Self {
        Self {
            source: CollectionSource::List(list),
            index: 0,
        }
    }

    pub fn from_tuple(tuple: TupleKey) -> Self {
        Self {
            source: CollectionSource::Tuple(tuple),
            index: 0,
        }
    }

    pub fn source_value(&self) -> Value {
        match self.source {
            CollectionSource::List(key) => Value::List(key),
            CollectionSource::Tuple(key) => Value::Tuple(key),
        }
    }
}

impl ValueIterator for CollectionIter {
    fn next(&mut self, arena: &mut ValueHolder) -> Option<Value> {
        let value = match self.source {
            CollectionSource::List(key) => {
                let list = arena.get_list(key).ok()?;
                list.get(self.index).copied()
            }
            CollectionSource::Tuple(key) => {
                let tuple = arena.get_tuple(key).ok()?;
                tuple.get(self.index).copied()
            }
        };

        if value.is_some() {
            self.index += 1;
        }

        value
    }
}
