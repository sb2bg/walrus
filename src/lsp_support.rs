use lalrpop_util::ParseError;
use lalrpop_util::lexer::Token;

use crate::ast::{FStringPart, Node, NodeKind};
use crate::error::RecoveredParseError;
use crate::grammar::ProgramParser;
use crate::span::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    Error,
}

#[derive(Debug, Clone)]
pub struct ParseDiagnostic {
    pub message: String,
    pub span: Span,
    pub severity: DiagnosticSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    Function,
    Method,
    Struct,
    Variable,
    Module,
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub span: Span,
    pub kind: SymbolKind,
}

#[derive(Debug, Clone)]
pub struct Definition {
    pub name: String,
    pub span: Span,
    pub kind: SymbolKind,
    pub scope_depth: usize,
}

#[derive(Debug, Clone)]
pub struct Reference {
    pub name: String,
    pub span: Span,
    pub scope_depth: usize,
}

#[derive(Debug, Clone)]
pub struct Analysis {
    pub diagnostics: Vec<ParseDiagnostic>,
    pub symbols: Vec<Symbol>,
    pub definitions: Vec<Definition>,
    pub references: Vec<Reference>,
}

impl Analysis {
    fn empty_with_diagnostics(diagnostics: Vec<ParseDiagnostic>) -> Self {
        Self {
            diagnostics,
            symbols: Vec::new(),
            definitions: Vec::new(),
            references: Vec::new(),
        }
    }
}

pub fn analyze(source: &str) -> Analysis {
    let parser = ProgramParser::new();
    match parser.parse(source) {
        Ok(ast) => {
            let mut collector = SymbolCollector::default();
            collector.walk(&ast, 0);
            Analysis {
                diagnostics: Vec::new(),
                symbols: collector.symbols,
                definitions: collector.definitions,
                references: collector.references,
            }
        }
        Err(err) => {
            let diagnostic = parse_error_to_diagnostic(err, source);
            Analysis::empty_with_diagnostics(vec![diagnostic])
        }
    }
}

fn parse_error_to_diagnostic(
    err: ParseError<usize, Token<'_>, RecoveredParseError>,
    source: &str,
) -> ParseDiagnostic {
    match err {
        ParseError::UnrecognizedEOF { expected, location } => ParseDiagnostic {
            message: format!("Unexpected end of input{}", expected_hint(&expected)),
            span: normalize_span(source, Span(location, location)),
            severity: DiagnosticSeverity::Error,
        },
        ParseError::UnrecognizedToken {
            token: (start, token, end),
            expected,
        } => ParseDiagnostic {
            message: format!("Unexpected token `{token}`{}", expected_hint(&expected)),
            span: normalize_span(source, Span(start, end)),
            severity: DiagnosticSeverity::Error,
        },
        ParseError::InvalidToken { location } => ParseDiagnostic {
            message: "Invalid token".to_string(),
            span: normalize_span(source, Span(location, location.saturating_add(1))),
            severity: DiagnosticSeverity::Error,
        },
        ParseError::ExtraToken {
            token: (start, token, end),
        } => ParseDiagnostic {
            message: format!("Extra token `{token}`"),
            span: normalize_span(source, Span(start, end)),
            severity: DiagnosticSeverity::Error,
        },
        ParseError::User { error } => user_error_to_diagnostic(error, source),
    }
}

fn user_error_to_diagnostic(error: RecoveredParseError, source: &str) -> ParseDiagnostic {
    match error {
        RecoveredParseError::NumberTooLarge(number, span) => ParseDiagnostic {
            message: format!("Number `{number}` is too large"),
            span: normalize_span(source, span),
            severity: DiagnosticSeverity::Error,
        },
        RecoveredParseError::InvalidEscapeSequence(sequence, span) => ParseDiagnostic {
            message: format!("Invalid escape sequence `{sequence}`"),
            span: normalize_span(source, span),
            severity: DiagnosticSeverity::Error,
        },
        RecoveredParseError::InvalidUnicodeEscapeSequence(span) => ParseDiagnostic {
            message: "Invalid unicode escape sequence".to_string(),
            span: normalize_span(source, span),
            severity: DiagnosticSeverity::Error,
        },
        RecoveredParseError::InvalidFStringExpression(expr, span) => ParseDiagnostic {
            message: format!("Invalid f-string expression `{expr}`"),
            span: normalize_span(source, span),
            severity: DiagnosticSeverity::Error,
        },
    }
}

fn normalize_span(source: &str, span: Span) -> Span {
    let len = source.len();
    let start = span.0.min(len);
    let mut end = span.1.min(len);

    if end < start {
        end = start;
    }

    if start == end && start < len {
        end += 1;
    }

    Span(start, end)
}

fn expected_hint(expected: &[String]) -> String {
    if expected.is_empty() {
        return String::new();
    }

    let formatted = expected
        .iter()
        .take(6)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    format!("; expected one of: {formatted}")
}

#[derive(Default)]
struct SymbolCollector {
    symbols: Vec<Symbol>,
    definitions: Vec<Definition>,
    references: Vec<Reference>,
}

impl SymbolCollector {
    fn push_definition(&mut self, name: &str, span: Span, kind: SymbolKind, scope_depth: usize) {
        self.symbols.push(Symbol {
            name: name.to_string(),
            span,
            kind,
        });
        self.definitions.push(Definition {
            name: name.to_string(),
            span,
            kind,
            scope_depth,
        });
    }

    fn push_reference(&mut self, name: &str, span: Span, scope_depth: usize) {
        self.references.push(Reference {
            name: name.to_string(),
            span,
            scope_depth,
        });
    }

    fn walk(&mut self, node: &Node, scope_depth: usize) {
        match node.kind() {
            NodeKind::Program(nodes)
            | NodeKind::Statements(nodes)
            | NodeKind::UnscopedStatements(nodes)
            | NodeKind::Block(nodes) => {
                for child in nodes {
                    self.walk(child, scope_depth);
                }
            }
            NodeKind::FunctionDefinition(name, _args, body) => {
                self.push_definition(name, *node.span(), SymbolKind::Function, scope_depth);
                self.walk(body, scope_depth + 1);
            }
            NodeKind::ExternFunctionDefinition(name, _args) => {
                self.push_definition(name, *node.span(), SymbolKind::Function, scope_depth);
            }
            NodeKind::StructDefinition(name, members) => {
                self.push_definition(name, *node.span(), SymbolKind::Struct, scope_depth);
                for member in members {
                    self.walk(member, scope_depth + 1);
                }
            }
            NodeKind::StructFunctionDefinition(name, _args, body) => {
                self.push_definition(name, *node.span(), SymbolKind::Method, scope_depth);
                self.walk(body, scope_depth + 1);
            }
            NodeKind::Assign(name, value) => {
                self.walk(value, scope_depth);
                self.push_definition(name, *node.span(), SymbolKind::Variable, scope_depth);
            }
            NodeKind::Reassign(name, value, _) => {
                self.push_reference(name.value(), name.span(), scope_depth);
                self.walk(value, scope_depth);
            }
            NodeKind::For(var, iter, body) => {
                self.walk(iter, scope_depth);
                let loop_scope = scope_depth + 1;
                self.push_definition(var, *node.span(), SymbolKind::Variable, loop_scope);
                self.walk(body, loop_scope);
            }
            NodeKind::Return(value)
            | NodeKind::Print(value)
            | NodeKind::Println(value)
            | NodeKind::Throw(value)
            | NodeKind::Free(value)
            | NodeKind::Defer(value)
            | NodeKind::ExpressionStatement(value)
            | NodeKind::UnaryOp(_, value) => {
                self.walk(value, scope_depth);
            }
            NodeKind::If(cond, then_branch, else_branch) => {
                self.walk(cond, scope_depth);
                self.walk(then_branch, scope_depth + 1);
                if let Some(else_node) = else_branch {
                    self.walk(else_node, scope_depth + 1);
                }
            }
            NodeKind::Ternary(value, cond, else_value) => {
                self.walk(value, scope_depth);
                self.walk(cond, scope_depth);
                self.walk(else_value, scope_depth);
            }
            NodeKind::While(cond, body) => {
                self.walk(cond, scope_depth);
                self.walk(body, scope_depth + 1);
            }
            NodeKind::Try(try_block, catch_var, catch_block) => {
                self.walk(try_block, scope_depth + 1);
                let catch_scope = scope_depth + 1;
                self.push_definition(
                    catch_var,
                    *catch_block.span(),
                    SymbolKind::Variable,
                    catch_scope,
                );
                self.walk(catch_block, catch_scope);
            }
            NodeKind::BinOp(left, _, right)
            | NodeKind::Index(left, right)
            | NodeKind::Range(Some(left), right) => {
                self.walk(left, scope_depth);
                self.walk(right, scope_depth);
            }
            NodeKind::Range(None, right) => {
                self.walk(right, scope_depth);
            }
            NodeKind::IndexAssign(target, index, value) => {
                self.walk(target, scope_depth);
                self.walk(index, scope_depth);
                self.walk(value, scope_depth);
            }
            NodeKind::FunctionCall(callee, args) => {
                self.walk(callee, scope_depth);
                for arg in args {
                    self.walk(arg, scope_depth);
                }
            }
            NodeKind::MemberAccess(target, _member) => {
                self.walk(target, scope_depth);
            }
            NodeKind::List(items) => {
                for item in items {
                    self.walk(item, scope_depth);
                }
            }
            NodeKind::Dict(entries) => {
                for (key, value) in entries {
                    self.walk(key, scope_depth);
                    self.walk(value, scope_depth);
                }
            }
            NodeKind::FString(parts) => {
                for part in parts {
                    if let FStringPart::Expr(expr) = part {
                        self.walk(expr, scope_depth);
                    }
                }
            }
            NodeKind::AnonFunctionDefinition(_args, body) => {
                self.walk(body, scope_depth + 1);
            }
            NodeKind::PackageImport(name, alias) | NodeKind::ModuleImport(name, alias) => {
                let symbol_name = alias.as_deref().unwrap_or(name);
                self.push_definition(symbol_name, *node.span(), SymbolKind::Module, scope_depth);
            }
            NodeKind::Ident(name) => {
                self.push_reference(name, *node.span(), scope_depth);
            }
            NodeKind::Int(_)
            | NodeKind::Float(_)
            | NodeKind::String(_)
            | NodeKind::Bool(_)
            | NodeKind::Break
            | NodeKind::Continue
            | NodeKind::Void => {}
        }
    }
}
