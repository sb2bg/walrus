//! Mark-and-Sweep Garbage Collector for Walrus
//!
//! This module implements a simple tracing garbage collector that:
//! 1. Marks all values reachable from roots (stack, locals, globals)
//! 2. Sweeps (frees) all unmarked heap objects
//!
//! The GC can be triggered:
//! - Automatically after a configurable number of allocations
//! - Automatically when estimated memory exceeds a threshold
//! - Manually via the `__gc__()` builtin

use std::mem::size_of;
use std::sync::atomic::{AtomicUsize, Ordering};

use rustc_hash::FxHashSet;

use crate::arenas::{DictKey, FuncKey, IterKey, ListKey, StringKey, StructDefKey, StructInstKey, TupleKey};
use crate::value::Value;

/// Default allocation count threshold before GC is triggered
pub const DEFAULT_ALLOCATION_THRESHOLD: usize = 1024;

/// Default memory threshold (bytes) before GC is triggered (8 MB)
pub const DEFAULT_MEMORY_THRESHOLD: usize = 8 * 1024 * 1024;

/// Global configurable threshold - can be changed via `__gc_threshold__()` builtin
static ALLOCATION_THRESHOLD: AtomicUsize = AtomicUsize::new(DEFAULT_ALLOCATION_THRESHOLD);
static MEMORY_THRESHOLD: AtomicUsize = AtomicUsize::new(DEFAULT_MEMORY_THRESHOLD);

/// Get the current allocation threshold
pub fn get_allocation_threshold() -> usize {
    ALLOCATION_THRESHOLD.load(Ordering::Relaxed)
}

/// Set the allocation threshold (returns the old value)
pub fn set_allocation_threshold(threshold: usize) -> usize {
    ALLOCATION_THRESHOLD.swap(threshold, Ordering::Relaxed)
}

/// Get the current memory threshold
pub fn get_memory_threshold() -> usize {
    MEMORY_THRESHOLD.load(Ordering::Relaxed)
}

/// Set the memory threshold in bytes (returns the old value)
pub fn set_memory_threshold(threshold: usize) -> usize {
    MEMORY_THRESHOLD.swap(threshold, Ordering::Relaxed)
}

/// Size of a Value on the stack (used for estimating container overhead)
pub const VALUE_SIZE: usize = size_of::<Value>();

/// Estimate memory used by a list (in bytes)
pub fn estimate_list_size(len: usize, capacity: usize) -> usize {
    // Vec overhead + capacity * element size
    size_of::<Vec<Value>>() + capacity * VALUE_SIZE
}

/// Estimate memory used by a tuple (in bytes)
pub fn estimate_tuple_size(len: usize) -> usize {
    size_of::<Vec<Value>>() + len * VALUE_SIZE
}

/// Estimate memory used by a dict entry
pub fn estimate_dict_size(len: usize) -> usize {
    // HashMap has complex memory layout, rough estimate
    // Each entry is roughly 2 Values (key + value) + bucket overhead
    size_of::<rustc_hash::FxHashMap<Value, Value>>() + len * (2 * VALUE_SIZE + 16)
}

/// Estimate memory for a function (bytecode + metadata)
pub fn estimate_function_size(bytecode_len: usize, constants_len: usize) -> usize {
    // Rough estimate: struct overhead + bytecode + constants
    64 + bytecode_len * 16 + constants_len * VALUE_SIZE
}

/// Estimate memory for a struct instance
pub fn estimate_struct_instance_size(field_count: usize) -> usize {
    // Name string + hashmap of fields
    32 + estimate_dict_size(field_count)
}

/// Tracks which heap objects are marked as reachable
#[derive(Debug, Default, Clone)]
pub struct GcState {
    /// Set of marked list keys
    marked_lists: FxHashSet<ListKey>,
    /// Set of marked tuple keys
    marked_tuples: FxHashSet<TupleKey>,
    /// Set of marked dict keys
    marked_dicts: FxHashSet<DictKey>,
    /// Set of marked function keys
    marked_functions: FxHashSet<FuncKey>,
    /// Set of marked iterator keys
    marked_iters: FxHashSet<IterKey>,
    /// Set of marked struct definition keys
    marked_struct_defs: FxHashSet<StructDefKey>,
    /// Set of marked struct instance keys
    marked_struct_insts: FxHashSet<StructInstKey>,
    /// Set of marked string keys
    marked_strings: FxHashSet<StringKey>,
    /// Number of allocations since last collection
    allocation_count: usize,
    /// Estimated bytes allocated since last collection
    bytes_allocated: usize,
    /// Total bytes freed by GC (lifetime)
    total_bytes_freed: usize,
    /// Total number of GC cycles run
    total_collections: usize,
}

impl GcState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an allocation with estimated byte size.
    /// Returns true if GC should be triggered.
    #[inline]
    pub fn record_allocation(&mut self, bytes: usize) -> bool {
        self.allocation_count += 1;
        self.bytes_allocated += bytes;

        self.allocation_count >= get_allocation_threshold()
            || self.bytes_allocated >= get_memory_threshold()
    }

    /// Record an allocation without byte tracking (backwards compat)
    #[inline]
    pub fn record_allocation_simple(&mut self) -> bool {
        self.record_allocation(0)
    }

    /// Reset counters after GC, recording stats
    pub fn finish_collection(&mut self, bytes_freed: usize) {
        self.allocation_count = 0;
        self.bytes_allocated = 0;
        self.total_bytes_freed += bytes_freed;
        self.total_collections += 1;
    }

    /// Get current allocation count
    pub fn allocation_count(&self) -> usize {
        self.allocation_count
    }

    /// Get estimated bytes allocated since last collection
    pub fn bytes_allocated(&self) -> usize {
        self.bytes_allocated
    }

    /// Get total bytes freed over all collections
    pub fn total_bytes_freed(&self) -> usize {
        self.total_bytes_freed
    }

    /// Get total number of collections run
    pub fn total_collections(&self) -> usize {
        self.total_collections
    }

    /// Clear all marks in preparation for a new GC cycle
    pub fn clear_marks(&mut self) {
        self.marked_lists.clear();
        self.marked_tuples.clear();
        self.marked_dicts.clear();
        self.marked_functions.clear();
        self.marked_iters.clear();
        self.marked_struct_defs.clear();
        self.marked_struct_insts.clear();
        self.marked_strings.clear();
    }

    /// Mark a value as reachable. Returns true if it was newly marked.
    pub fn mark(&mut self, value: Value) -> bool {
        match value {
            Value::List(key) => self.marked_lists.insert(key),
            Value::Tuple(key) => self.marked_tuples.insert(key),
            Value::Dict(key) => self.marked_dicts.insert(key),
            Value::Function(key) => self.marked_functions.insert(key),
            Value::Iter(key) => self.marked_iters.insert(key),
            Value::StructDef(key) => self.marked_struct_defs.insert(key),
            Value::StructInst(key) => self.marked_struct_insts.insert(key),
            Value::String(key) => self.marked_strings.insert(key),
            // Primitives don't need marking (stored inline, not on heap)
            Value::Int(_)
            | Value::Float(_)
            | Value::Bool(_)
            | Value::Range(_)
            | Value::Void => false,
        }
    }

    /// Check if a list is marked
    pub fn is_list_marked(&self, key: ListKey) -> bool {
        self.marked_lists.contains(&key)
    }

    /// Check if a tuple is marked
    pub fn is_tuple_marked(&self, key: TupleKey) -> bool {
        self.marked_tuples.contains(&key)
    }

    /// Check if a dict is marked
    pub fn is_dict_marked(&self, key: DictKey) -> bool {
        self.marked_dicts.contains(&key)
    }

    /// Check if a function is marked
    pub fn is_function_marked(&self, key: FuncKey) -> bool {
        self.marked_functions.contains(&key)
    }

    /// Check if an iterator is marked
    pub fn is_iter_marked(&self, key: IterKey) -> bool {
        self.marked_iters.contains(&key)
    }

    /// Check if a struct definition is marked
    pub fn is_struct_def_marked(&self, key: StructDefKey) -> bool {
        self.marked_struct_defs.contains(&key)
    }

    /// Check if a struct instance is marked
    pub fn is_struct_inst_marked(&self, key: StructInstKey) -> bool {
        self.marked_struct_insts.contains(&key)
    }

    /// Check if a string is marked
    pub fn is_string_marked(&self, key: StringKey) -> bool {
        self.marked_strings.contains(&key)
    }
}
