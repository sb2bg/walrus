use std::collections::VecDeque;
use std::ops::Range;

use rustc_hash::FxHashMap;

use crate::arenas::{HeapValue, ValueHolder};
use crate::range::RangeValue;
use crate::value::Value;

pub trait ValueIterator {
    fn next(&mut self, arena: &mut ValueHolder) -> Option<Value>;
}

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

pub struct StrIter {
    chars: VecDeque<char>,
}

impl StrIter {
    pub fn new(string: &str) -> Self {
        Self {
            chars: string.chars().collect(),
        }
    }
}

impl ValueIterator for StrIter {
    fn next(&mut self, arena: &mut ValueHolder) -> Option<Value> {
        self.chars
            .pop_front()
            .map(|c| arena.push(HeapValue::String(&c.to_string())))
    }
}

pub struct DictIter {
    dict: Vec<(Value, Value)>,
}

impl DictIter {
    pub fn new(dict: &FxHashMap<Value, Value>) -> Self {
        Self {
            dict: dict.iter().map(|(k, v)| (*k, *v)).collect(),
        }
    }
}

impl ValueIterator for DictIter {
    fn next(&mut self, arena: &mut ValueHolder) -> Option<Value> {
        self.dict
            .pop()
            .map(|(k, v)| arena.push(HeapValue::Tuple(&vec![k, v])))
    }
}

pub struct CollectionIter {
    collection: VecDeque<Value>,
}

impl CollectionIter {
    pub fn new(collection: &Vec<Value>) -> Self {
        Self {
            collection: collection.iter().copied().collect(),
        }
    }
}

impl ValueIterator for CollectionIter {
    fn next(&mut self, _: &mut ValueHolder) -> Option<Value> {
        self.collection.pop_front()
    }
}
