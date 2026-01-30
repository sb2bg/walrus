//! Compile-time optimizations for the Walrus bytecode compiler.
//!
//! This module contains optimization passes that can be applied during compilation:
//! - Constant folding: Evaluate constant expressions at compile time
//! - Increment/decrement detection: Recognize `x = x + 1` patterns
//! - (Future) Dead code elimination
//! - (Future) Strength reduction

use float_ord::FloatOrd;

use crate::ast::{Node, NodeKind};
use crate::value::Value;
use crate::vm::opcode::Opcode;

/// Result of analyzing a reassignment for optimization opportunities
#[derive(Debug)]
pub enum ReassignOptimization {
    /// Can use IncrementLocal opcode
    Increment,
    /// Can use DecrementLocal opcode
    Decrement,
    /// No optimization available
    None,
}

/// Analyze a reassignment to see if it can be optimized to increment/decrement.
/// Detects patterns like:
/// - `x += 1` (op is Add, node is Int(1))
/// - `x -= 1` (op is Subtract, node is Int(1))
/// - `x = x + 1` (op is Equal, node is BinOp(Ident(x), Add, Int(1)))
/// - `x = 1 + x` (op is Equal, node is BinOp(Int(1), Add, Ident(x)))
/// - `x = x - 1` (op is Equal, node is BinOp(Ident(x), Subtract, Int(1)))
pub fn analyze_reassign_for_increment(
    var_name: &str,
    node: &Node,
    op: Opcode,
) -> ReassignOptimization {
    // Check for x += 1 or x -= 1
    if let NodeKind::Int(1) = node.kind() {
        match op {
            Opcode::Add => return ReassignOptimization::Increment,
            Opcode::Subtract => return ReassignOptimization::Decrement,
            _ => {}
        }
    }

    // Check for x = x + 1, x = 1 + x, x = x - 1 patterns
    if op == Opcode::Equal {
        if let NodeKind::BinOp(lhs, bin_op, rhs) = node.kind() {
            // Check for x = x + 1 or x = 1 + x
            if *bin_op == Opcode::Add {
                let is_increment = match (lhs.kind(), rhs.kind()) {
                    (NodeKind::Ident(var), NodeKind::Int(1)) if var == var_name => true,
                    (NodeKind::Int(1), NodeKind::Ident(var)) if var == var_name => true,
                    _ => false,
                };
                if is_increment {
                    return ReassignOptimization::Increment;
                }
            }
            // Check for x = x - 1
            if *bin_op == Opcode::Subtract {
                if let (NodeKind::Ident(var), NodeKind::Int(1)) = (lhs.kind(), rhs.kind()) {
                    if var == var_name {
                        return ReassignOptimization::Decrement;
                    }
                }
            }
        }
    }

    ReassignOptimization::None
}

/// Try to evaluate a binary operation at compile time (constant folding).
/// Returns Some(Value) if both operands are constants and the operation can be folded,
/// None otherwise.
pub fn try_fold_binop(left: &Node, op: Opcode, right: &Node) -> Option<Value> {
    let left_val = try_get_constant(left)?;
    let right_val = try_get_constant(right)?;

    fold_binary_op(left_val, op, right_val)
}

/// Fold a binary operation on two constant values.
fn fold_binary_op(left: Value, op: Opcode, right: Value) -> Option<Value> {
    match (left, op, right) {
        // Integer arithmetic
        (Value::Int(a), Opcode::Add, Value::Int(b)) => Some(Value::Int(a.wrapping_add(b))),
        (Value::Int(a), Opcode::Subtract, Value::Int(b)) => Some(Value::Int(a.wrapping_sub(b))),
        (Value::Int(a), Opcode::Multiply, Value::Int(b)) => Some(Value::Int(a.wrapping_mul(b))),
        (Value::Int(a), Opcode::Divide, Value::Int(b)) if b != 0 => Some(Value::Int(a / b)),
        (Value::Int(a), Opcode::Modulo, Value::Int(b)) if b != 0 => Some(Value::Int(a % b)),
        (Value::Int(a), Opcode::Power, Value::Int(b)) if b >= 0 => {
            Some(Value::Int(a.wrapping_pow(b as u32)))
        }

        // Float arithmetic
        (Value::Float(FloatOrd(a)), Opcode::Add, Value::Float(FloatOrd(b))) => {
            Some(Value::Float(FloatOrd(a + b)))
        }
        (Value::Float(FloatOrd(a)), Opcode::Subtract, Value::Float(FloatOrd(b))) => {
            Some(Value::Float(FloatOrd(a - b)))
        }
        (Value::Float(FloatOrd(a)), Opcode::Multiply, Value::Float(FloatOrd(b))) => {
            Some(Value::Float(FloatOrd(a * b)))
        }
        (Value::Float(FloatOrd(a)), Opcode::Divide, Value::Float(FloatOrd(b))) => {
            Some(Value::Float(FloatOrd(a / b)))
        }

        // Mixed int/float arithmetic
        (Value::Int(a), Opcode::Add, Value::Float(FloatOrd(b))) => {
            Some(Value::Float(FloatOrd(a as f64 + b)))
        }
        (Value::Float(FloatOrd(a)), Opcode::Add, Value::Int(b)) => {
            Some(Value::Float(FloatOrd(a + b as f64)))
        }
        (Value::Int(a), Opcode::Subtract, Value::Float(FloatOrd(b))) => {
            Some(Value::Float(FloatOrd(a as f64 - b)))
        }
        (Value::Float(FloatOrd(a)), Opcode::Subtract, Value::Int(b)) => {
            Some(Value::Float(FloatOrd(a - b as f64)))
        }
        (Value::Int(a), Opcode::Multiply, Value::Float(FloatOrd(b))) => {
            Some(Value::Float(FloatOrd(a as f64 * b)))
        }
        (Value::Float(FloatOrd(a)), Opcode::Multiply, Value::Int(b)) => {
            Some(Value::Float(FloatOrd(a * b as f64)))
        }
        (Value::Int(a), Opcode::Divide, Value::Float(FloatOrd(b))) => {
            Some(Value::Float(FloatOrd(a as f64 / b)))
        }
        (Value::Float(FloatOrd(a)), Opcode::Divide, Value::Int(b)) => {
            Some(Value::Float(FloatOrd(a / b as f64)))
        }

        // Integer comparisons
        (Value::Int(a), Opcode::Equal, Value::Int(b)) => Some(Value::Bool(a == b)),
        (Value::Int(a), Opcode::NotEqual, Value::Int(b)) => Some(Value::Bool(a != b)),
        (Value::Int(a), Opcode::Less, Value::Int(b)) => Some(Value::Bool(a < b)),
        (Value::Int(a), Opcode::LessEqual, Value::Int(b)) => Some(Value::Bool(a <= b)),
        (Value::Int(a), Opcode::Greater, Value::Int(b)) => Some(Value::Bool(a > b)),
        (Value::Int(a), Opcode::GreaterEqual, Value::Int(b)) => Some(Value::Bool(a >= b)),

        // Float comparisons
        (Value::Float(FloatOrd(a)), Opcode::Equal, Value::Float(FloatOrd(b))) => {
            Some(Value::Bool(a == b))
        }
        (Value::Float(FloatOrd(a)), Opcode::NotEqual, Value::Float(FloatOrd(b))) => {
            Some(Value::Bool(a != b))
        }
        (Value::Float(FloatOrd(a)), Opcode::Less, Value::Float(FloatOrd(b))) => {
            Some(Value::Bool(a < b))
        }
        (Value::Float(FloatOrd(a)), Opcode::LessEqual, Value::Float(FloatOrd(b))) => {
            Some(Value::Bool(a <= b))
        }
        (Value::Float(FloatOrd(a)), Opcode::Greater, Value::Float(FloatOrd(b))) => {
            Some(Value::Bool(a > b))
        }
        (Value::Float(FloatOrd(a)), Opcode::GreaterEqual, Value::Float(FloatOrd(b))) => {
            Some(Value::Bool(a >= b))
        }

        // Boolean operations
        (Value::Bool(a), Opcode::And, Value::Bool(b)) => Some(Value::Bool(a && b)),
        (Value::Bool(a), Opcode::Or, Value::Bool(b)) => Some(Value::Bool(a || b)),
        (Value::Bool(a), Opcode::Equal, Value::Bool(b)) => Some(Value::Bool(a == b)),
        (Value::Bool(a), Opcode::NotEqual, Value::Bool(b)) => Some(Value::Bool(a != b)),

        _ => None,
    }
}

/// Try to extract a constant value from a node (for constant folding).
/// Recursively folds nested constant expressions.
pub fn try_get_constant(node: &Node) -> Option<Value> {
    match node.kind() {
        NodeKind::Int(v) => Some(Value::Int(*v)),
        NodeKind::Float(v) => Some(Value::Float(*v)),
        NodeKind::Bool(v) => Some(Value::Bool(*v)),

        // Recursively fold nested binary operations
        NodeKind::BinOp(left, op, right) => try_fold_binop(left, *op, right),

        // Fold unary operations
        NodeKind::UnaryOp(Opcode::Negate, inner) => match try_get_constant(inner)? {
            Value::Int(v) => Some(Value::Int(-v)),
            Value::Float(FloatOrd(v)) => Some(Value::Float(FloatOrd(-v))),
            _ => None,
        },
        NodeKind::UnaryOp(Opcode::Not, inner) => match try_get_constant(inner)? {
            Value::Bool(v) => Some(Value::Bool(!v)),
            _ => None,
        },

        _ => None,
    }
}

/// Try to fold a unary operation at compile time.
pub fn try_fold_unary(op: Opcode, operand: &Node) -> Option<Value> {
    let val = try_get_constant(operand)?;

    match (op, val) {
        (Opcode::Negate, Value::Int(v)) => Some(Value::Int(-v)),
        (Opcode::Negate, Value::Float(FloatOrd(v))) => Some(Value::Float(FloatOrd(-v))),
        (Opcode::Not, Value::Bool(v)) => Some(Value::Bool(!v)),
        _ => None,
    }
}
