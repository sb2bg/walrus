use float_ord::FloatOrd;
use log::debug;
use once_cell::sync::Lazy;
use rustc_hash::FxHashMap;
use slotmap::{DenseSlotMap, new_key_type};
use strena::{Interner, Symbol};

use crate::WalrusResult;
use crate::error::WalrusError;
use crate::function::WalrusFunction;
use crate::gc::{
    estimate_dict_size, estimate_function_size, estimate_list_size, estimate_struct_instance_size,
    estimate_tuple_size, get_allocation_threshold, GcState,
};
use crate::iter::{CollectionIter, DictIter, RangeIter, StrIter};
use crate::structs::{StructDefinition, StructInstance};
use crate::value::{Value, ValueIter};

pub static mut ARENA: Lazy<ValueHolder> = Lazy::new(ValueHolder::new);

// todo: maybe instead of this, we can use a single slotmap
// and use a different enum to differentiate between the
// types stored by value and the types stored by key
// todo: maybe use a different arena library, DenseSlotMap is mid-performance
#[derive(Default, Debug, Clone)]
pub struct ValueHolder {
    dicts: DenseSlotMap<DictKey, FxHashMap<Value, Value>>,
    lists: DenseSlotMap<ListKey, Vec<Value>>,
    tuples: DenseSlotMap<TupleKey, Vec<Value>>, // todo: use slice?
    strings: Interner,
    functions: DenseSlotMap<FuncKey, WalrusFunction>,
    iterators: DenseSlotMap<IterKey, ValueIter>,
    struct_defs: DenseSlotMap<StructDefKey, StructDefinition>,
    struct_instances: DenseSlotMap<StructInstKey, StructInstance>,
    gc: GcState,
}

impl ValueHolder {
    pub fn new() -> Self {
        Self {
            dicts: DenseSlotMap::with_key(),
            lists: DenseSlotMap::with_key(),
            tuples: DenseSlotMap::with_key(),
            strings: Interner::default(), // todo: use FxHasher
            functions: DenseSlotMap::with_key(),
            iterators: DenseSlotMap::with_key(),
            struct_defs: DenseSlotMap::with_key(),
            struct_instances: DenseSlotMap::with_key(),
            gc: GcState::new(),
        }
    }

    pub fn free(&mut self, key: Value) -> bool {
        match key {
            Value::Dict(key) => self.dicts.remove(key).is_some(),
            Value::List(key) => self.lists.remove(key).is_some(),
            Value::String(_key) => {
                // fixme: for now, no way to free strings
                // self.string_interner.remove(key).is_some()
                false
            }
            Value::Function(key) => self.functions.remove(key).is_some(),
            Value::Tuple(key) => self.tuples.remove(key).is_some(),
            Value::Iter(key) => self.iterators.remove(key).is_some(),
            Value::StructDef(key) => self.struct_defs.remove(key).is_some(),
            Value::StructInst(key) => self.struct_instances.remove(key).is_some(),
            _ => false,
        }
    }

    // todo: split this into multiple functions for each type
    pub fn push(&mut self, value: HeapValue) -> Value {
        // Estimate allocation size and track for GC
        let bytes = match &value {
            HeapValue::List(list) => estimate_list_size(list.len(), list.capacity()),
            HeapValue::Tuple(tuple) => estimate_tuple_size(tuple.len()),
            HeapValue::Dict(dict) => estimate_dict_size(dict.len()),
            HeapValue::Function(func) => {
                let (bc_len, const_len) = match func {
                    WalrusFunction::Vm(f) => (f.code.instructions.len(), f.code.constants.len()),
                    _ => (0, 0),
                };
                estimate_function_size(bc_len, const_len)
            }
            HeapValue::String(s) => s.len() + 24, // String + overhead
            HeapValue::Iter(_) => 64,             // Rough estimate
            HeapValue::StructDef(_) => 128,       // Rough estimate
            HeapValue::StructInst(inst) => estimate_struct_instance_size(inst.fields().len()),
        };
        self.gc.record_allocation(bytes);

        match value {
            HeapValue::List(list) => Value::List(self.lists.insert(list)),
            HeapValue::Tuple(tuple) => Value::Tuple(self.tuples.insert(tuple.to_vec())),
            HeapValue::Dict(dict) => Value::Dict(self.dicts.insert(dict)),
            HeapValue::Function(func) => Value::Function(self.functions.insert(func)),
            HeapValue::String(string) => Value::String(self.strings.get_or_insert(string)),
            HeapValue::Iter(iter) => Value::Iter(self.iterators.insert(iter)),
            HeapValue::StructDef(def) => Value::StructDef(self.struct_defs.insert(def)),
            HeapValue::StructInst(inst) => Value::StructInst(self.struct_instances.insert(inst)),
        }
    }

    /// Check if garbage collection should be triggered
    #[inline]
    pub fn should_collect(&self) -> bool {
        self.gc.allocation_count() >= get_allocation_threshold()
            || self.gc.bytes_allocated() >= crate::gc::get_memory_threshold()
    }

    /// Mark a value and all values it references as reachable
    pub fn mark(&mut self, value: Value) {
        // If not newly marked (already marked or primitive), skip
        if !self.gc.mark(value) {
            return;
        }

        // Trace contained values
        self.trace(value);
    }

    /// Trace and mark all values contained within a heap object
    fn trace(&mut self, value: Value) {
        match value {
            Value::List(key) => {
                if let Some(list) = self.lists.get(key) {
                    // Clone to avoid borrow issues
                    let items: Vec<Value> = list.clone();
                    for item in items {
                        self.mark(item);
                    }
                }
            }
            Value::Tuple(key) => {
                if let Some(tuple) = self.tuples.get(key) {
                    let items: Vec<Value> = tuple.clone();
                    for item in items {
                        self.mark(item);
                    }
                }
            }
            Value::Dict(key) => {
                if let Some(dict) = self.dicts.get(key) {
                    let entries: Vec<(Value, Value)> =
                        dict.iter().map(|(&k, &v)| (k, v)).collect();
                    for (k, v) in entries {
                        self.mark(k);
                        self.mark(v);
                    }
                }
            }
            Value::StructInst(key) => {
                if let Some(inst) = self.struct_instances.get(key) {
                    let fields: Vec<Value> = inst.fields().values().copied().collect();
                    for field in fields {
                        self.mark(field);
                    }
                }
            }
            // Functions, iterators, struct defs, and primitives don't contain traceable values
            _ => {}
        }
    }

    /// Mark all values in a slice as reachable (for marking roots)
    pub fn mark_roots(&mut self, roots: &[Value]) {
        for &value in roots {
            self.mark(value);
        }
    }

    /// Sweep (free) all unmarked heap objects
    /// Returns the number of objects freed
    pub fn sweep(&mut self) -> usize {
        let mut freed = 0;

        // Sweep lists
        let unmarked_lists: Vec<ListKey> = self
            .lists
            .keys()
            .filter(|&k| !self.gc.is_list_marked(k))
            .collect();
        for key in unmarked_lists {
            self.lists.remove(key);
            freed += 1;
        }

        // Sweep tuples
        let unmarked_tuples: Vec<TupleKey> = self
            .tuples
            .keys()
            .filter(|&k| !self.gc.is_tuple_marked(k))
            .collect();
        for key in unmarked_tuples {
            self.tuples.remove(key);
            freed += 1;
        }

        // Sweep dicts
        let unmarked_dicts: Vec<DictKey> = self
            .dicts
            .keys()
            .filter(|&k| !self.gc.is_dict_marked(k))
            .collect();
        for key in unmarked_dicts {
            self.dicts.remove(key);
            freed += 1;
        }

        // Sweep functions
        let unmarked_funcs: Vec<FuncKey> = self
            .functions
            .keys()
            .filter(|&k| !self.gc.is_function_marked(k))
            .collect();
        for key in unmarked_funcs {
            self.functions.remove(key);
            freed += 1;
        }

        // Sweep iterators
        let unmarked_iters: Vec<IterKey> = self
            .iterators
            .keys()
            .filter(|&k| !self.gc.is_iter_marked(k))
            .collect();
        for key in unmarked_iters {
            self.iterators.remove(key);
            freed += 1;
        }

        // Sweep struct definitions
        let unmarked_struct_defs: Vec<StructDefKey> = self
            .struct_defs
            .keys()
            .filter(|&k| !self.gc.is_struct_def_marked(k))
            .collect();
        for key in unmarked_struct_defs {
            self.struct_defs.remove(key);
            freed += 1;
        }

        // Sweep struct instances
        let unmarked_struct_insts: Vec<StructInstKey> = self
            .struct_instances
            .keys()
            .filter(|&k| !self.gc.is_struct_inst_marked(k))
            .collect();
        for key in unmarked_struct_insts {
            self.struct_instances.remove(key);
            freed += 1;
        }

        // Clear marks and reset for next cycle
        self.gc.clear_marks();
        // Estimate bytes freed (rough estimate based on object count)
        let bytes_freed = freed * 64; // Average object size estimate
        self.gc.finish_collection(bytes_freed);

        freed
    }

    /// Run a full garbage collection cycle
    /// Returns the number of objects freed
    pub fn collect_garbage(&mut self, roots: &[Value]) -> usize {
        debug!("GC: Starting collection with {} roots", roots.len());

        // Mark phase
        self.mark_roots(roots);

        // Sweep phase
        let freed = self.sweep();

        debug!("GC: Freed {} objects", freed);
        freed
    }

    /// Force a garbage collection (for manual triggering via builtin)
    pub fn force_collect(&mut self, roots: &[Value]) -> GcResult {
        let before_objects = self.total_objects();
        let before_bytes = self.gc.bytes_allocated();

        let freed = self.collect_garbage(roots);

        GcResult {
            objects_freed: freed,
            objects_before: before_objects,
            objects_after: self.total_objects(),
            bytes_before: before_bytes,
            collections_total: self.gc.total_collections(),
        }
    }

    /// Get total number of heap objects
    pub fn total_objects(&self) -> usize {
        self.lists.len()
            + self.tuples.len()
            + self.dicts.len()
            + self.functions.len()
            + self.iterators.len()
            + self.struct_defs.len()
            + self.struct_instances.len()
    }

    /// Get GC statistics for introspection
    pub fn gc_stats(&self) -> GcInfo {
        GcInfo {
            allocation_count: self.gc.allocation_count(),
            bytes_allocated: self.gc.bytes_allocated(),
            total_bytes_freed: self.gc.total_bytes_freed(),
            total_collections: self.gc.total_collections(),
            allocation_threshold: get_allocation_threshold(),
            memory_threshold: crate::gc::get_memory_threshold(),
        }
    }

    /// Get heap statistics
    pub fn heap_stats(&self) -> HeapStats {
        HeapStats {
            lists: self.lists.len(),
            tuples: self.tuples.len(),
            dicts: self.dicts.len(),
            functions: self.functions.len(),
            iterators: self.iterators.len(),
            struct_defs: self.struct_defs.len(),
            struct_instances: self.struct_instances.len(),
            allocation_count: self.gc.allocation_count(),
        }
    }

    pub fn is_truthy(&self, value: Value) -> WalrusResult<bool> {
        Ok(match value {
            Value::Bool(b) => b,
            Value::Int(i) => i != 0,
            Value::Float(f) => f != FloatOrd(0.0),
            Value::String(s) => !self.get_string(s)?.is_empty(),
            Value::List(l) => !self.get_list(l)?.is_empty(),
            Value::Dict(d) => !self.get_dict(d)?.is_empty(),
            Value::Tuple(t) => !self.get_tuple(t)?.is_empty(),
            Value::Range(range) => !range.is_empty(),
            Value::Function(_) => true,
            Value::Iter(_) => true,
            Value::StructDef(_) => true,
            Value::StructInst(_) => true,
            Value::Void => false,
        })
    }

    pub fn push_ident(&mut self, ident: &str) -> Symbol {
        self.strings.get_or_insert(ident)
    }

    pub fn get_mut_dict(&mut self, key: DictKey) -> WalrusResult<&mut FxHashMap<Value, Value>> {
        Self::check(self.dicts.get_mut(key))
    }

    pub fn get_dict(&self, key: DictKey) -> WalrusResult<&FxHashMap<Value, Value>> {
        Self::check(self.dicts.get(key))
    }

    pub fn get_mut_list(&mut self, key: ListKey) -> WalrusResult<&mut Vec<Value>> {
        Self::check(self.lists.get_mut(key))
    }

    pub fn get_list(&self, key: ListKey) -> WalrusResult<&Vec<Value>> {
        Self::check(self.lists.get(key))
    }

    pub fn get_tuple(&self, key: TupleKey) -> WalrusResult<&Vec<Value>> {
        Self::check(self.tuples.get(key))
    }

    pub fn get_string(&self, key: Symbol) -> WalrusResult<&str> {
        Self::check(self.strings.resolve(key))
    }

    pub fn get_function(&self, key: FuncKey) -> WalrusResult<&WalrusFunction> {
        Self::check(self.functions.get(key))
    }

    pub fn get_iter(&self, key: IterKey) -> WalrusResult<&ValueIter> {
        Self::check(self.iterators.get(key))
    }

    pub fn get_mut_iter(&mut self, key: IterKey) -> WalrusResult<&mut ValueIter> {
        Self::check(self.iterators.get_mut(key))
    }

    pub fn get_struct_def(&self, key: StructDefKey) -> WalrusResult<&StructDefinition> {
        Self::check(self.struct_defs.get(key))
    }

    pub fn get_struct_inst(&self, key: StructInstKey) -> WalrusResult<&StructInstance> {
        Self::check(self.struct_instances.get(key))
    }

    pub fn get_mut_struct_inst(&mut self, key: StructInstKey) -> WalrusResult<&mut StructInstance> {
        Self::check(self.struct_instances.get_mut(key))
    }

    pub fn value_to_iter(&mut self, value: Value) -> WalrusResult<Value> {
        let key = match value {
            Value::List(list) => {
                let iter = CollectionIter::new(self.get_list(list)?);
                self.push(HeapValue::Iter(ValueIter::Collection(iter)))
            }
            Value::Tuple(tuple) => {
                let iter = CollectionIter::new(self.get_tuple(tuple)?);
                self.push(HeapValue::Iter(ValueIter::Collection(iter)))
            }
            Value::Dict(dict) => {
                let iter = DictIter::new(self.get_dict(dict)?);
                self.push(HeapValue::Iter(ValueIter::Dict(iter)))
            }
            Value::String(string) => {
                let iter = StrIter::new(self.get_string(string)?);
                self.push(HeapValue::Iter(ValueIter::Str(iter)))
            }
            Value::Range(range) => {
                self.push(HeapValue::Iter(ValueIter::Range(RangeIter::new(range))))
            }
            Value::Iter(iter) => Value::Iter(iter),
            _ => {
                return Err(WalrusError::TodoError {
                    //todo
                    message: "TODO: implement error for not iterable".into(),
                });
            }
        };

        Ok(key)
    }

    fn check<T>(result: Option<T>) -> WalrusResult<T> {
        result.ok_or(WalrusError::UnknownError {
            message: "Attempt to access released memory".into(), // fixme: use correct AccessReleasedMemory error
        })
    }

    // todo: speed this up
    pub fn stringify(&self, value: Value) -> WalrusResult<String> {
        Ok(match value {
            Value::Int(i) => i.to_string(),
            Value::Float(FloatOrd(f)) => f.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Range(r) => r.to_string(),
            Value::String(s) => self.get_string(s)?.to_string(),
            Value::List(l) => {
                let list = self.get_list(l)?;
                let mut s = String::new();

                s.push('[');

                for (i, &item) in list.iter().enumerate() {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    s.push_str(&self.stringify(item)?);
                }

                s.push(']');
                s
            }
            Value::Tuple(t) => {
                let tuple = self.get_tuple(t)?;
                let mut s = String::new();

                s.push('(');

                for (i, &item) in tuple.iter().enumerate() {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    s.push_str(&self.stringify(item)?);
                }

                s.push(')');
                s
            }
            Value::Dict(d) => {
                let dict = self.get_dict(d)?;
                let mut s = String::new();

                s.push('{');

                for (i, (&key, &value)) in dict.iter().enumerate() {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    s.push_str(&self.stringify(key)?);
                    s.push_str(": ");
                    s.push_str(&self.stringify(value)?);
                }

                s.push('}');
                s
            }
            Value::Function(f) => {
                let func = self.get_function(f)?;
                func.to_string()
            }
            Value::Iter(_) => "<iter object>".to_string(),
            Value::StructDef(s) => {
                let def = self.get_struct_def(s)?;
                def.to_string()
            }
            Value::StructInst(s) => {
                let inst = self.get_struct_inst(s)?;
                inst.to_string()
            }
            Value::Void => "void".to_string(),
        })
    }
}

/// Statistics about the heap
#[derive(Debug, Clone)]
pub struct HeapStats {
    pub lists: usize,
    pub tuples: usize,
    pub dicts: usize,
    pub functions: usize,
    pub iterators: usize,
    pub struct_defs: usize,
    pub struct_instances: usize,
    pub allocation_count: usize,
}

impl HeapStats {
    pub fn total_objects(&self) -> usize {
        self.lists
            + self.tuples
            + self.dicts
            + self.functions
            + self.iterators
            + self.struct_defs
            + self.struct_instances
    }
}

/// Result of a garbage collection cycle
#[derive(Debug, Clone)]
pub struct GcResult {
    pub objects_freed: usize,
    pub objects_before: usize,
    pub objects_after: usize,
    pub bytes_before: usize,
    pub collections_total: usize,
}

/// GC statistics for introspection
#[derive(Debug, Clone)]
pub struct GcInfo {
    pub allocation_count: usize,
    pub bytes_allocated: usize,
    pub total_bytes_freed: usize,
    pub total_collections: usize,
    pub allocation_threshold: usize,
    pub memory_threshold: usize,
}

pub enum HeapValue<'a> {
    List(Vec<Value>),
    Tuple(&'a [Value]),
    Dict(FxHashMap<Value, Value>),
    Function(WalrusFunction),
    String(&'a str),
    Iter(ValueIter),
    StructDef(StructDefinition),
    StructInst(StructInstance),
}

impl HeapValue<'_> {
    pub fn alloc(self) -> Value {
        unsafe { (*std::ptr::addr_of_mut!(ARENA)).push(self) }
    }
}

new_key_type! {
    pub struct ListKey;
    pub struct DictKey;
    pub struct FuncKey;
    pub struct TupleKey;
    pub struct IterKey;
    pub struct StructDefKey;
    pub struct StructInstKey;
}

pub trait Free {
    fn free(&mut self) -> bool;
}

pub trait Resolve<'a> {
    type Output;
    fn resolve(self) -> WalrusResult<Self::Output>;
}

pub trait ResolveMut<'a> {
    type Output;
    fn resolve_mut(self) -> WalrusResult<Self::Output>;
}

impl Free for Value {
    fn free(&mut self) -> bool {
        unsafe { (*std::ptr::addr_of_mut!(ARENA)).free(*self) }
    }
}

impl<'a> Resolve<'a> for ListKey {
    type Output = &'a Vec<Value>;

    fn resolve(self) -> WalrusResult<Self::Output> {
        unsafe { (*std::ptr::addr_of_mut!(ARENA)).get_list(self) }
    }
}

impl<'a> Resolve<'a> for DictKey {
    type Output = &'a FxHashMap<Value, Value>;

    fn resolve(self) -> WalrusResult<Self::Output> {
        unsafe { (*std::ptr::addr_of_mut!(ARENA)).get_dict(self) }
    }
}

impl<'a> Resolve<'a> for FuncKey {
    type Output = &'a WalrusFunction;

    fn resolve(self) -> WalrusResult<Self::Output> {
        unsafe { (*std::ptr::addr_of_mut!(ARENA)).get_function(self) }
    }
}

impl<'a> Resolve<'a> for Symbol {
    type Output = &'a str;

    fn resolve(self) -> WalrusResult<Self::Output> {
        unsafe { (*std::ptr::addr_of_mut!(ARENA)).get_string(self) }
    }
}

impl<'a> ResolveMut<'a> for ListKey {
    type Output = &'a mut Vec<Value>;

    fn resolve_mut(self) -> WalrusResult<Self::Output> {
        unsafe { (*std::ptr::addr_of_mut!(ARENA)).get_mut_list(self) }
    }
}

impl<'a> ResolveMut<'a> for DictKey {
    type Output = &'a mut FxHashMap<Value, Value>;

    fn resolve_mut(self) -> WalrusResult<Self::Output> {
        unsafe { (*std::ptr::addr_of_mut!(ARENA)).get_mut_dict(self) }
    }
}

impl<'a> Resolve<'a> for TupleKey {
    type Output = &'a Vec<Value>;

    fn resolve(self) -> WalrusResult<Self::Output> {
        unsafe { (*std::ptr::addr_of_mut!(ARENA)).get_tuple(self) }
    }
}

impl<'a> Resolve<'a> for StructDefKey {
    type Output = &'a StructDefinition;

    fn resolve(self) -> WalrusResult<Self::Output> {
        unsafe { (*std::ptr::addr_of_mut!(ARENA)).get_struct_def(self) }
    }
}

impl<'a> Resolve<'a> for StructInstKey {
    type Output = &'a StructInstance;

    fn resolve(self) -> WalrusResult<Self::Output> {
        unsafe { (*std::ptr::addr_of_mut!(ARENA)).get_struct_inst(self) }
    }
}

impl<'a> ResolveMut<'a> for StructInstKey {
    type Output = &'a mut StructInstance;

    fn resolve_mut(self) -> WalrusResult<Self::Output> {
        unsafe { (*std::ptr::addr_of_mut!(ARENA)).get_mut_struct_inst(self) }
    }
}
