//! Type tracking and profiling for JIT compilation.
//!
//! Since Walrus is dynamically typed, we need to observe types at runtime
//! to generate specialized native code. This module tracks:
//! - Observed types at each bytecode location
//! - Type stability (monomorphic vs polymorphic)
//! - Specialized type information for containers

use std::fmt::{self, Display};

use rustc_hash::FxHashMap;

use crate::arenas::StructDefKey;
use crate::value::Value;

/// Represents the type of a Walrus value for JIT purposes.
///
/// Unlike the runtime `Value` enum which holds actual data,
/// `WalrusType` represents just the type information needed
/// for specialization decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WalrusType {
    /// 64-bit signed integer
    Int,
    /// 64-bit floating point
    Float,
    /// Boolean value
    Bool,
    /// Interned string
    String,
    /// Homogeneous list (element type if known)
    List(Option<WalrusTypeId>),
    /// Tuple (fixed size, heterogeneous)
    Tuple,
    /// Dictionary
    Dict,
    /// Range value (start..end)
    Range,
    /// Function reference
    Function,
    /// Iterator
    Iter,
    /// Struct definition
    StructDef,
    /// Struct instance (with optional def key for specialization)
    StructInst(Option<StructDefKey>),
    /// Void/None/Unit
    Void,
    /// Type is not yet known (no observations)
    Unknown,
    /// Multiple different types observed (polymorphic)
    Polymorphic,
}

/// A compact type ID for recursive type references (e.g., List element types)
/// This avoids infinite recursion in type representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WalrusTypeId(u8);

impl WalrusTypeId {
    pub const INT: WalrusTypeId = WalrusTypeId(0);
    pub const FLOAT: WalrusTypeId = WalrusTypeId(1);
    pub const BOOL: WalrusTypeId = WalrusTypeId(2);
    pub const STRING: WalrusTypeId = WalrusTypeId(3);
    pub const UNKNOWN: WalrusTypeId = WalrusTypeId(255);

    pub fn from_type(ty: WalrusType) -> Self {
        match ty {
            WalrusType::Int => Self::INT,
            WalrusType::Float => Self::FLOAT,
            WalrusType::Bool => Self::BOOL,
            WalrusType::String => Self::STRING,
            _ => Self::UNKNOWN,
        }
    }

    pub fn to_type(self) -> WalrusType {
        match self.0 {
            0 => WalrusType::Int,
            1 => WalrusType::Float,
            2 => WalrusType::Bool,
            3 => WalrusType::String,
            _ => WalrusType::Unknown,
        }
    }
}

impl WalrusType {
    /// Extract the type from a runtime Value
    pub fn from_value(value: &Value) -> Self {
        match value {
            Value::Int(_) => WalrusType::Int,
            Value::Float(_) => WalrusType::Float,
            Value::Bool(_) => WalrusType::Bool,
            Value::String(_) => WalrusType::String,
            Value::List(_) => WalrusType::List(None), // Element type tracked separately
            Value::Tuple(_) => WalrusType::Tuple,
            Value::Dict(_) => WalrusType::Dict,
            Value::Range(_) => WalrusType::Range,
            Value::Function(_) => WalrusType::Function,
            Value::Iter(_) => WalrusType::Iter,
            Value::StructDef(_) => WalrusType::StructDef,
            Value::StructInst(_) => WalrusType::StructInst(None),
            Value::Void => WalrusType::Void,
        }
    }

    /// Check if this type is suitable for JIT compilation
    ///
    /// Monomorphic types (single observed type) are ideal for JIT.
    /// Polymorphic code paths should stay in the interpreter.
    pub fn is_jit_candidate(&self) -> bool {
        !matches!(self, WalrusType::Unknown | WalrusType::Polymorphic)
    }

    /// Check if this is a numeric type (good for arithmetic specialization)
    pub fn is_numeric(&self) -> bool {
        matches!(self, WalrusType::Int | WalrusType::Float)
    }

    /// Check if this type can be unboxed (stored directly, not via heap key)
    pub fn is_unboxable(&self) -> bool {
        matches!(
            self,
            WalrusType::Int | WalrusType::Float | WalrusType::Bool | WalrusType::Void
        )
    }

    /// Merge two types, returning the combined type
    ///
    /// Used when multiple observations exist at the same program point:
    /// - Same type → keep that type (monomorphic)
    /// - Different types → Polymorphic
    pub fn merge(self, other: WalrusType) -> WalrusType {
        if self == other {
            self
        } else if self == WalrusType::Unknown {
            other
        } else if other == WalrusType::Unknown {
            self
        } else {
            // Special case: Int + Float → Float (numeric widening)
            if self.is_numeric() && other.is_numeric() {
                WalrusType::Float
            } else {
                WalrusType::Polymorphic
            }
        }
    }
}

impl Display for WalrusType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WalrusType::Int => write!(f, "int"),
            WalrusType::Float => write!(f, "float"),
            WalrusType::Bool => write!(f, "bool"),
            WalrusType::String => write!(f, "string"),
            WalrusType::List(elem) => {
                if let Some(elem_id) = elem {
                    write!(f, "list[{}]", elem_id.to_type())
                } else {
                    write!(f, "list")
                }
            }
            WalrusType::Tuple => write!(f, "tuple"),
            WalrusType::Dict => write!(f, "dict"),
            WalrusType::Range => write!(f, "range"),
            WalrusType::Function => write!(f, "function"),
            WalrusType::Iter => write!(f, "iter"),
            WalrusType::StructDef => write!(f, "struct_def"),
            WalrusType::StructInst(_) => write!(f, "struct"),
            WalrusType::Void => write!(f, "void"),
            WalrusType::Unknown => write!(f, "?"),
            WalrusType::Polymorphic => write!(f, "poly"),
        }
    }
}

/// Type feedback for a single bytecode location.
///
/// Tracks all observed types at a specific instruction pointer,
/// along with observation count for confidence.
#[derive(Debug, Clone)]
pub struct TypeFeedback {
    /// The merged type from all observations
    pub observed_type: WalrusType,
    /// Number of times this location has been executed
    pub observation_count: u32,
    /// Whether the type has been stable (same type every time)
    pub is_stable: bool,
}

impl Default for TypeFeedback {
    fn default() -> Self {
        Self {
            observed_type: WalrusType::Unknown,
            observation_count: 0,
            is_stable: true,
        }
    }
}

impl TypeFeedback {
    /// Record a new type observation
    pub fn observe(&mut self, ty: WalrusType) {
        if self.observation_count == 0 {
            self.observed_type = ty;
        } else if self.observed_type != ty {
            self.is_stable = false;
            self.observed_type = self.observed_type.merge(ty);
        }
        self.observation_count = self.observation_count.saturating_add(1);
    }

    /// Check if we have enough observations to be confident about the type
    pub fn is_confident(&self) -> bool {
        self.observation_count >= 10 && self.is_stable
    }

    /// Check if this feedback suggests JIT compilation would be beneficial
    pub fn should_jit(&self) -> bool {
        self.is_confident() && self.observed_type.is_jit_candidate()
    }
}

/// Type profile for an entire function or code region.
///
/// Maps instruction pointers to their type feedback.
#[derive(Debug, Default, Clone)]
pub struct TypeProfile {
    /// Type feedback indexed by instruction pointer
    feedback: FxHashMap<usize, TypeFeedback>,
    /// Stack type profile at key points (loop headers, call sites)
    stack_profiles: FxHashMap<usize, Vec<WalrusType>>,
}

impl TypeProfile {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a type observation at a specific instruction
    pub fn observe(&mut self, ip: usize, ty: WalrusType) {
        self.feedback.entry(ip).or_default().observe(ty);
    }

    /// Record the stack state at a specific instruction
    pub fn observe_stack(&mut self, ip: usize, stack_types: Vec<WalrusType>) {
        self.stack_profiles.insert(ip, stack_types);
    }

    /// Get type feedback for an instruction
    pub fn get(&self, ip: usize) -> Option<&TypeFeedback> {
        self.feedback.get(&ip)
    }

    /// Get the stack profile at an instruction
    pub fn get_stack(&self, ip: usize) -> Option<&[WalrusType]> {
        self.stack_profiles.get(&ip).map(|v| v.as_slice())
    }

    /// Check if a code region (start..end) is a good JIT candidate
    pub fn region_is_jit_candidate(&self, start: usize, end: usize) -> bool {
        // All observed types in the region should be stable and JIT-able
        self.feedback
            .iter()
            .filter(|&(&ip, _)| ip >= start && ip < end)
            .all(|(_, fb)| fb.should_jit())
    }

    /// Get a summary of types observed in a region
    pub fn summarize_region(&self, start: usize, end: usize) -> Vec<(usize, &TypeFeedback)> {
        let mut result: Vec<_> = self
            .feedback
            .iter()
            .filter(|&(&ip, _)| ip >= start && ip < end)
            .map(|(&ip, fb)| (ip, fb))
            .collect();
        result.sort_by_key(|(ip, _)| *ip);
        result
    }

    /// Clear all profiling data (useful after JIT compilation)
    pub fn clear(&mut self) {
        self.feedback.clear();
        self.stack_profiles.clear();
    }

    /// Number of profiled locations
    pub fn len(&self) -> usize {
        self.feedback.len()
    }

    pub fn is_empty(&self) -> bool {
        self.feedback.is_empty()
    }
}
