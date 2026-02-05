//! Opcode handlers for the Walrus VM.
//!
//! This module organizes opcode handlers by category for better maintainability.
//! Each handler is marked `#[inline(always)]` to ensure no performance regression.

mod arithmetic;
mod builtins;
mod collections;
mod comparison;
mod control_flow;
mod logic;
mod loops;
mod stack;
mod structs;

pub use arithmetic::*;
pub use builtins::*;
pub use collections::*;
pub use comparison::*;
pub use control_flow::*;
pub use logic::*;
pub use loops::*;
pub use stack::*;
pub use structs::*;
