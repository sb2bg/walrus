//! JIT Compilation Infrastructure for Walrus
//!
//! This module provides the foundation for Just-In-Time compilation:
//! - Type tracking and profiling
//! - Hot-spot detection for loops and functions
//! - Type specialization information for the JIT compiler
//! - Cranelift-based native code generation (when `jit` feature is enabled)
//!
//! Phase 1: Type tracking and hot-spot detection
//! Phase 2: Cranelift-based JIT compilation (current)

#[cfg(feature = "jit")]
pub mod compiler;
pub mod hotspot;
pub mod types;

#[cfg(feature = "jit")]
pub use compiler::{CompiledFunction, CompiledPattern, JitCompiler, JitError, JitResult, JitStats};
pub use hotspot::{HotSpotDetector, HotSpotKind, HotSpotStats};
pub use types::{TypeFeedback, TypeProfile, WalrusType};
