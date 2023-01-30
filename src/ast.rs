use crate::span::{Span, Spanned};
use crate::vm::opcode::Opcode;
use float_ord::FloatOrd;
use std::fmt::{Debug, Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub kind: NodeKind,
    span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeKind {
    Statements(Vec<Node>),
    Int(i64),
    Float(FloatOrd<f64>),
    String(String),
    List(Vec<Node>),
    Bool(bool),
    Dict(Vec<(Node, Node)>),
    BinOp(Box<Node>, Opcode, Box<Node>),
    UnaryOp(Opcode, Box<Node>),
    Ident(String),
    Assign(String, Box<Node>),
    Reassign(Spanned<String>, Box<Node>, Opcode),
    IndexAssign(Box<Node>, Box<Node>, Box<Node>),
    FunctionCall(Box<Node>, Vec<Node>),
    Index(Box<Node>, Box<Node>),
    AnonFunctionDefinition(Vec<String>, Box<Node>),
    FunctionDefinition(String, Vec<String>, Box<Node>),
    ExternFunctionDefinition(String, Vec<String>),
    Return(Box<Node>),
    If(Box<Node>, Box<Node>, Option<Box<Node>>),
    Ternary(Box<Node>, Box<Node>, Box<Node>),
    While(Box<Node>, Box<Node>),
    For(String, Box<Node>, Box<Node>),
    Block(Vec<Node>),
    PackageImport(String, Option<String>),
    ModuleImport(String, Option<String>),
    Print(Box<Node>),
    Println(Box<Node>),
    Throw(Box<Node>),
    Try(Box<Node>, String, Box<Node>),
    Free(Box<Node>),
    Range(Option<Box<Node>>, Option<Box<Node>>),
    Defer(Box<Node>),
    Break,
    Continue,
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

impl Display for NodeKind {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            NodeKind::Statements(_) => write!(f, "Statements"),
            NodeKind::Int(_) => write!(f, "Int"),
            NodeKind::Float(_) => write!(f, "Float"),
            NodeKind::String(_) => write!(f, "String"),
            NodeKind::List(_) => write!(f, "List"),
            NodeKind::Bool(_) => write!(f, "Bool"),
            NodeKind::Dict(_) => write!(f, "Dict"),
            NodeKind::BinOp(_, _, _) => write!(f, "BinOp"),
            NodeKind::UnaryOp(_, _) => write!(f, "UnaryOp"),
            NodeKind::Ident(_) => write!(f, "Ident"),
            NodeKind::Assign(_, _) => write!(f, "Assign"),
            NodeKind::Reassign(_, _, _) => write!(f, "Reassign"),
            NodeKind::FunctionCall(_, _) => write!(f, "FunctionCall"),
            NodeKind::AnonFunctionDefinition(_, _) => write!(f, "AnonFunctionDefinition"),
            NodeKind::FunctionDefinition(_, _, _) => write!(f, "FunctionDefinition"),
            NodeKind::ExternFunctionDefinition(_, _) => write!(f, "ExternFunctionDefinition"),
            NodeKind::Return(_) => write!(f, "Return"),
            NodeKind::If(_, _, _) => write!(f, "If"),
            NodeKind::Ternary(_, _, _) => write!(f, "Ternary"),
            NodeKind::While(_, _) => write!(f, "While"),
            NodeKind::For(_, _, _) => write!(f, "For"),
            NodeKind::Block(_) => write!(f, "Block"),
            NodeKind::PackageImport(_, _) => write!(f, "PackageImport"),
            NodeKind::ModuleImport(_, _) => write!(f, "ModuleImport"),
            NodeKind::Break => write!(f, "Break"),
            NodeKind::Continue => write!(f, "Continue"),
            NodeKind::Print(_) => write!(f, "Print"),
            NodeKind::Println(_) => write!(f, "Println"),
            NodeKind::Throw(_) => write!(f, "Throw"),
            NodeKind::Try(_, _, _) => write!(f, "Try"),
            NodeKind::Free(_) => write!(f, "Free"),
            NodeKind::Index(_, _) => write!(f, "Index"),
            NodeKind::IndexAssign(_, _, _) => write!(f, "IndexAssign"),
            NodeKind::Range(_, _) => write!(f, "Range"),
            NodeKind::Defer(_) => write!(f, "Defer"),
            NodeKind::Void => write!(f, "Void"),
        }
    }
}
