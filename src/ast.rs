use crate::span::Span;
use get_size::GetSize;
use std::fmt::Display;

#[derive(Debug, PartialEq, GetSize)]
pub enum Node {
    Statement(Vec<Box<Node>>),
    Int(i64, Span),
    Float(f64, Span),
    String(String, Span),
    List(Vec<Box<Node>>, Span),
    Bool(bool, Span),
    Dict(Vec<(Box<Node>, Box<Node>)>, Span),
    BinOp(Box<Node>, Op, Box<Node>, Span),
    UnaryOp(Op, Box<Node>, Span),
    Ident(String, Span),
    Assign(String, Box<Node>, Span),
    FunctionCall(String, Vec<Box<Node>>),
    FunctionDefinition(String, Vec<String>, Box<Node>),
    Return(Box<Node>, Span),
    If(Box<Node>, Box<Node>, Option<Box<Node>>),
    While(Box<Node>, Box<Node>),
    For(String, Box<Node>, Box<Node>, Option<Box<Node>>, Box<Node>),
    Block(Vec<Box<Node>>, Span),
    Void(Span),
}

#[derive(Debug, PartialEq, GetSize)]
pub enum Op {
    Mul,
    Div,
    Add,
    Sub,
    Mod,
    Pow,
    Not,
    Equal,
    NotEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    And,
    Or,
}

impl Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::Mul => write!(f, "*"),
            Op::Div => write!(f, "/"),
            Op::Add => write!(f, "+"),
            Op::Sub => write!(f, "-"),
            Op::Mod => write!(f, "%"),
            Op::Pow => write!(f, "^"),
            Op::Not => write!(f, "!"),
            Op::Equal => write!(f, "=="),
            Op::NotEqual => write!(f, "!="),
            Op::Greater => write!(f, ">"),
            Op::GreaterEqual => write!(f, ">="),
            Op::Less => write!(f, "<"),
            Op::LessEqual => write!(f, "<="),
            Op::And => write!(f, "&&"),
            Op::Or => write!(f, "||"),
        }
    }
}
