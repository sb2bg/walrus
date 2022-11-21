use std::ops::Range;

#[derive(Debug)]
pub enum Node {
    Statement(Vec<Box<Node>>),
    Int(i64, Range<usize>),
    Float(f64, Range<usize>),
    String(String, Range<usize>),
    List(Vec<Box<Node>>, Range<usize>),
    BinOp(Box<Node>, Op, Box<Node>),
    UnaryOp(Op, Box<Node>),
    Ident(String, Range<usize>),
    Assign(String, Box<Node>),
    FunctionCall(String, Vec<Box<Node>>),
    FunctionDefinition(String, Vec<String>, Box<Node>),
    Return(Box<Node>),
    If(Box<Node>, Box<Node>, Option<Box<Node>>),
    While(Box<Node>, Box<Node>),
    For(String, Box<Node>, Box<Node>, Option<Box<Node>>, Box<Node>),
    Block(Vec<Box<Node>>, Range<usize>),
}

#[derive(Debug)]
pub enum Op {
    Mul,
    Div,
    Add,
    Sub,
    Mod,
    Pow,
    Not,
}
