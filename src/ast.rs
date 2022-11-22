use crate::span::Span;
use get_size::GetSize;

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
    Assign(String, Box<Node>),
    FunctionCall(String, Vec<Box<Node>>),
    FunctionDefinition(String, Vec<String>, Box<Node>),
    Return(Box<Node>),
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
}
