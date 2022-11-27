use crate::span::Span;
use float_ord::FloatOrd;
use std::fmt::Display;

#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Clone)]
pub struct Node {
    kind: NodeKind,
    span: Span,
}

#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Clone)]
pub enum NodeKind {
    Statement(Vec<Box<Node>>),
    Int(i64),
    Float(FloatOrd<f64>),
    String(String),
    List(Vec<Box<Node>>),
    Bool(bool),
    Dict(Vec<(Box<Node>, Box<Node>)>),
    BinOp(Box<Node>, Op, Box<Node>),
    UnaryOp(Op, Box<Node>),
    Ident(String),
    Assign(String, Box<Node>),
    FunctionCall(String, Vec<Box<Node>>),
    AnonFunctionDefinition(Vec<String>, Box<Node>),
    FunctionDefinition(String, Vec<String>, Box<Node>),
    Return(Box<Node>),
    If(Box<Node>, Box<Node>, Option<Box<Node>>),
    While(Box<Node>, Box<Node>),
    For(String, Box<Node>, Box<Node>, Option<Box<Node>>, Box<Node>),
    Block(Vec<Box<Node>>),
    Void,
}

impl Node {
    pub fn new(kind: NodeKind, span: Span) -> Self {
        Self { kind, span }
    }

    pub fn kind(&self) -> &NodeKind {
        &self.kind
    }

    pub fn into_kind(self) -> NodeKind {
        self.kind
    }

    pub fn span(&self) -> &Span {
        &self.span
    }
}

#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Clone)]
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
