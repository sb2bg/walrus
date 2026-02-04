use rustc_hash::FxHashSet;

use crate::span::{Span, Spanned};
use crate::vm::opcode::Opcode;
use float_ord::FloatOrd;
use std::fmt::{Debug, Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FStringPart {
    Literal(String),
    Expr(Box<Node>), // Store parsed expression node with proper span
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub kind: NodeKind,
    span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeKind {
    Program(Vec<Node>),
    Statements(Vec<Node>),
    UnscopedStatements(Vec<Node>),
    Int(i64),
    Float(FloatOrd<f64>),
    String(String),
    FString(Vec<FStringPart>),
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
    StructDefinition(String, Vec<Node>),
    StructFunctionDefinition(String, Vec<String>, Box<Node>),
    Return(Box<Node>),
    If(Box<Node>, Box<Node>, Option<Box<Node>>),
    Ternary(Box<Node>, Box<Node>, Box<Node>),
    While(Box<Node>, Box<Node>),
    For(String, Box<Node>, Box<Node>),
    Block(Vec<Node>),
    ExpressionStatement(Box<Node>),
    PackageImport(String, Option<String>),
    ModuleImport(String, Option<String>),
    Print(Box<Node>),
    Println(Box<Node>),
    Throw(Box<Node>),
    Try(Box<Node>, String, Box<Node>),
    Free(Box<Node>),
    Range(Option<Box<Node>>, Box<Node>),
    Defer(Box<Node>),
    MemberAccess(Box<Node>, String),
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
            NodeKind::Program(_) => write!(f, "Program"),
            NodeKind::Statements(_) => write!(f, "Statements"),
            NodeKind::UnscopedStatements(_) => write!(f, "UnscopedStatements"),
            NodeKind::Int(_) => write!(f, "Int"),
            NodeKind::Float(_) => write!(f, "Float"),
            NodeKind::String(_) => write!(f, "String"),
            NodeKind::FString(_) => write!(f, "FString"),
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
            NodeKind::StructDefinition(_, _) => write!(f, "StructDefinition"),
            NodeKind::StructFunctionDefinition(_, _, _) => write!(f, "StructFunctionDefinition"),
            NodeKind::Return(_) => write!(f, "Return"),
            NodeKind::If(_, _, _) => write!(f, "If"),
            NodeKind::Ternary(_, _, _) => write!(f, "Ternary"),
            NodeKind::While(_, _) => write!(f, "While"),
            NodeKind::For(_, _, _) => write!(f, "For"),
            NodeKind::Block(_) => write!(f, "Block"),
            NodeKind::ExpressionStatement(_) => write!(f, "ExpressionStatement"),
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
            NodeKind::MemberAccess(_, _) => write!(f, "MemberAccess"),
            NodeKind::Void => write!(f, "Void"),
        }
    }
}

/// Collect all free variables in a function body.
/// Free variables are those that are referenced but not defined locally.
pub fn collect_free_variables(body: &Node, params: &[String]) -> FxHashSet<String> {
    let mut free_vars = FxHashSet::default();
    let mut defined = FxHashSet::default();

    // Parameters are defined
    for param in params {
        defined.insert(param.clone());
    }

    collect_free_vars_recursive(body, &mut defined, &mut free_vars);
    free_vars
}

fn collect_free_vars_recursive(
    node: &Node,
    defined: &mut FxHashSet<String>,
    free_vars: &mut FxHashSet<String>,
) {
    match &node.kind {
        NodeKind::Ident(name) => {
            if !defined.contains(name) {
                free_vars.insert(name.clone());
            }
        }
        NodeKind::Assign(name, value) => {
            // First collect from value (it might reference the variable before assignment)
            collect_free_vars_recursive(value, defined, free_vars);
            // Then mark as defined
            defined.insert(name.clone());
        }
        NodeKind::Reassign(name, value, _) => {
            // Check if the variable is free (referenced before local definition)
            if !defined.contains(name.value()) {
                free_vars.insert(name.value().clone());
            }
            collect_free_vars_recursive(value, defined, free_vars);
        }
        NodeKind::For(var, iter, body) => {
            collect_free_vars_recursive(iter, defined, free_vars);
            // For loop variable is defined within the body
            let mut inner_defined = defined.clone();
            inner_defined.insert(var.clone());
            collect_free_vars_recursive(body, &mut inner_defined, free_vars);
        }
        NodeKind::FunctionDefinition(name, args, body) => {
            // Function name is defined in outer scope
            defined.insert(name.clone());
            // Don't recurse into the function body - it has its own closure scope
        }
        NodeKind::AnonFunctionDefinition(args, body) => {
            // Anonymous functions capture their environment, but we don't recurse
            // because they'll capture variables when they're created
        }
        NodeKind::Statements(nodes) | NodeKind::UnscopedStatements(nodes) | NodeKind::Program(nodes) => {
            for child in nodes {
                collect_free_vars_recursive(child, defined, free_vars);
            }
        }
        NodeKind::BinOp(left, _, right) => {
            collect_free_vars_recursive(left, defined, free_vars);
            collect_free_vars_recursive(right, defined, free_vars);
        }
        NodeKind::UnaryOp(_, value) => {
            collect_free_vars_recursive(value, defined, free_vars);
        }
        NodeKind::FunctionCall(func, args) => {
            collect_free_vars_recursive(func, defined, free_vars);
            for arg in args {
                collect_free_vars_recursive(arg, defined, free_vars);
            }
        }
        NodeKind::Index(value, index) => {
            collect_free_vars_recursive(value, defined, free_vars);
            collect_free_vars_recursive(index, defined, free_vars);
        }
        NodeKind::IndexAssign(value, index, new_value) => {
            collect_free_vars_recursive(value, defined, free_vars);
            collect_free_vars_recursive(index, defined, free_vars);
            collect_free_vars_recursive(new_value, defined, free_vars);
        }
        NodeKind::Return(value) => {
            collect_free_vars_recursive(value, defined, free_vars);
        }
        NodeKind::If(cond, then_branch, else_branch) => {
            collect_free_vars_recursive(cond, defined, free_vars);
            collect_free_vars_recursive(then_branch, defined, free_vars);
            if let Some(else_br) = else_branch {
                collect_free_vars_recursive(else_br, defined, free_vars);
            }
        }
        NodeKind::Ternary(value, cond, else_value) => {
            collect_free_vars_recursive(value, defined, free_vars);
            collect_free_vars_recursive(cond, defined, free_vars);
            collect_free_vars_recursive(else_value, defined, free_vars);
        }
        NodeKind::While(cond, body) => {
            collect_free_vars_recursive(cond, defined, free_vars);
            collect_free_vars_recursive(body, defined, free_vars);
        }
        NodeKind::List(items) => {
            for item in items {
                collect_free_vars_recursive(item, defined, free_vars);
            }
        }
        NodeKind::Dict(pairs) => {
            for (key, value) in pairs {
                collect_free_vars_recursive(key, defined, free_vars);
                collect_free_vars_recursive(value, defined, free_vars);
            }
        }
        NodeKind::FString(parts) => {
            for part in parts {
                if let FStringPart::Expr(expr) = part {
                    collect_free_vars_recursive(expr, defined, free_vars);
                }
            }
        }
        NodeKind::Print(value) | NodeKind::Println(value) => {
            collect_free_vars_recursive(value, defined, free_vars);
        }
        NodeKind::Throw(value) | NodeKind::Free(value) | NodeKind::Defer(value) => {
            collect_free_vars_recursive(value, defined, free_vars);
        }
        NodeKind::ExpressionStatement(expr) => {
            collect_free_vars_recursive(expr, defined, free_vars);
        }
        NodeKind::Range(start, end) => {
            if let Some(s) = start {
                collect_free_vars_recursive(s, defined, free_vars);
            }
            collect_free_vars_recursive(end, defined, free_vars);
        }
        NodeKind::MemberAccess(value, _) => {
            collect_free_vars_recursive(value, defined, free_vars);
        }
        NodeKind::Try(try_block, catch_var, catch_block) => {
            collect_free_vars_recursive(try_block, defined, free_vars);
            let mut inner_defined = defined.clone();
            inner_defined.insert(catch_var.clone());
            collect_free_vars_recursive(catch_block, &mut inner_defined, free_vars);
        }
        // These don't contain variable references
        NodeKind::Int(_) | NodeKind::Float(_) | NodeKind::String(_) | NodeKind::Bool(_)
        | NodeKind::Break | NodeKind::Continue | NodeKind::Void
        | NodeKind::PackageImport(_, _) | NodeKind::ModuleImport(_, _)
        | NodeKind::ExternFunctionDefinition(_, _) | NodeKind::StructDefinition(_, _)
        | NodeKind::StructFunctionDefinition(_, _, _) | NodeKind::Block(_) => {}
    }
}
