use crate::ast::{Node, Op};
use crate::error::InterpreterError;
use float_ord::FloatOrd;
use itertools::join;
use std::collections::BTreeMap;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};

#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Clone)]
pub enum Value {
    Int(i64),
    Float(FloatOrd<f64>),
    Bool(bool),
    String(String),
    List(Vec<Value>),
    Dict(BTreeMap<Value, Value>),
    Function(String, Vec<String>, Node),
    Void,
}

type OperationResult = Result<Value, InterpreterError>;

impl Value {
    // todo: consider implicit conversion from int to float
    pub fn add(self, other: Self) -> OperationResult {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                Ok(Value::Float(FloatOrd(a + b)))
            }
            (Value::String(a), Value::String(b)) => {
                let mut a = a;
                a.push_str(&b);

                Ok(Value::String(a))
            }
            (Value::List(a), Value::List(b)) => {
                let mut a = a;
                a.extend(b);

                Ok(Value::List(a))
            }
            (Value::Dict(a), Value::Dict(b)) => {
                let mut a = a;
                a.extend(b);

                Ok(Value::Dict(a))
            }
            (a, b) => Err(InterpreterError::InvalidOperation { op: Op::Add, a, b }),
        }
    }

    pub fn sub(self, other: Self) -> OperationResult {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                Ok(Value::Float(FloatOrd(a - b)))
            }
            (a, b) => Err(InterpreterError::InvalidOperation { op: Op::Sub, a, b }),
        }
    }

    pub fn mul(self, other: Self) -> OperationResult {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                Ok(Value::Float(FloatOrd(a * b)))
            }
            (a, b) => Err(InterpreterError::InvalidOperation { op: Op::Mul, a, b }),
        }
    }

    pub fn div(self, other: Self) -> OperationResult {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a / b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                Ok(Value::Float(FloatOrd(a / b)))
            }
            (a, b) => Err(InterpreterError::InvalidOperation { op: Op::Div, a, b }),
        }
    }

    pub fn rem(self, other: Self) -> OperationResult {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a % b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                Ok(Value::Float(FloatOrd(a % b)))
            }
            (a, b) => Err(InterpreterError::InvalidOperation { op: Op::Mod, a, b }),
        }
    }

    pub fn pow(self, other: Self) -> OperationResult {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.pow(b as u32))),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => {
                Ok(Value::Float(FloatOrd(a.powf(b))))
            }
            (a, b) => Err(InterpreterError::InvalidOperation { op: Op::Pow, a, b }),
        }
    }

    pub fn neg(self) -> OperationResult {
        match self {
            Value::Int(a) => Ok(Value::Int(-a)),
            Value::Float(FloatOrd(a)) => Ok(Value::Float(FloatOrd(-a))),
            value => Err(InterpreterError::InvalidUnaryOperation { op: Op::Sub, value }),
        }
    }

    pub fn eq(self, other: Self) -> OperationResult {
        Ok(Value::Bool(self == other))
    }

    pub fn ne(self, other: Self) -> OperationResult {
        Ok(Value::Bool(self != other))
    }

    pub fn lt(self, other: Self) -> OperationResult {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a < b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => Ok(Value::Bool(a < b)),
            (a, b) => Err(InterpreterError::InvalidOperation { op: Op::Less, a, b }),
        }
    }

    pub fn le(self, other: Self) -> OperationResult {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a <= b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => Ok(Value::Bool(a <= b)),
            (a, b) => Err(InterpreterError::InvalidOperation {
                op: Op::LessEqual,
                a,
                b,
            }),
        }
    }

    pub fn gt(self, other: Self) -> OperationResult {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a > b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => Ok(Value::Bool(a > b)),
            (a, b) => Err(InterpreterError::InvalidOperation {
                op: Op::Greater,
                a,
                b,
            }),
        }
    }

    pub fn ge(self, other: Self) -> OperationResult {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a >= b)),
            (Value::Float(FloatOrd(a)), Value::Float(FloatOrd(b))) => Ok(Value::Bool(a >= b)),
            (a, b) => Err(InterpreterError::InvalidOperation {
                op: Op::GreaterEqual,
                a,
                b,
            }),
        }
    }

    pub fn and(self, other: Self) -> OperationResult {
        match (self, other) {
            (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a && b)),
            (a, b) => Err(InterpreterError::InvalidOperation { op: Op::And, a, b }),
        }
    }

    pub fn or(self, other: Self) -> OperationResult {
        match (self, other) {
            (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a || b)),
            (a, b) => Err(InterpreterError::InvalidOperation { op: Op::Or, a, b }),
        }
    }

    pub fn not(self) -> OperationResult {
        match self {
            Value::Bool(a) => Ok(Value::Bool(!a)),
            value => Err(InterpreterError::InvalidUnaryOperation { op: Op::Not, value }),
        }
    }

    pub fn get_type(&self) -> &str {
        match self {
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Bool(_) => "bool",
            Value::String(_) => "string",
            Value::List(_) => "list",
            Value::Dict(_) => "dict",
            Value::Function(..) => "function",
            Value::Void => "void",
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(i) => write!(f, "{}", i),
            Value::Float(FloatOrd(fl)) => write!(f, "{}", fl),
            Value::Bool(b) => write!(f, "{}", b),
            Value::String(s) => write!(f, "{}", s),
            Value::List(l) => write!(f, "[{}]", vec_to_string(l)),
            Value::Dict(d) => write!(f, "{{{}}}", map_to_string(d)),
            Value::Function(name, args, _) => write!(f, "[{}({})]", name, vec_to_string(args)),
            Value::Void => write!(f, "void"),
        }
    }
}

fn vec_to_string<T: Display>(vec: &Vec<T>) -> String {
    join(vec.iter(), ", ")
}

fn map_to_string(map: &BTreeMap<Value, Value>) -> String {
    join(map.iter().map(|(k, v)| format!("{}: {}", k, v)), ", ")
}
