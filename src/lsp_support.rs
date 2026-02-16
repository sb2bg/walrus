use std::collections::{HashMap, HashSet};

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
    Parameter,
    Module,
}

#[derive(Debug, Clone)]
pub struct ScopeInfo {
    pub id: usize,
    pub parent: Option<usize>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Definition {
    pub id: usize,
    pub name: String,
    pub span: Span,
    pub full_span: Span,
    pub kind: SymbolKind,
    pub scope_id: usize,
    pub declaration_start: usize,
    pub hoisted: bool,
    pub detail: Option<String>,
    pub documentation: Option<String>,
    pub parameters: Vec<String>,
    pub container: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct Reference {
    pub name: String,
    pub span: Span,
    pub scope_id: usize,
    pub is_write: bool,
    pub resolved_definition: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct Analysis {
    pub diagnostics: Vec<ParseDiagnostic>,
    pub definitions: Vec<Definition>,
    pub references: Vec<Reference>,
    pub scopes: Vec<ScopeInfo>,
    definitions_by_scope: HashMap<usize, Vec<usize>>,
}

impl Analysis {
    fn empty_with_diagnostics(diagnostics: Vec<ParseDiagnostic>, source_len: usize) -> Self {
        Self {
            diagnostics,
            definitions: Vec::new(),
            references: Vec::new(),
            scopes: vec![ScopeInfo {
                id: 0,
                parent: None,
                span: Span(0, source_len),
            }],
            definitions_by_scope: HashMap::new(),
        }
    }

    pub fn definition_by_id(&self, id: usize) -> Option<&Definition> {
        self.definitions.get(id)
    }

    pub fn definition_at_offset(&self, offset: usize) -> Option<&Definition> {
        self.definitions
            .iter()
            .filter(|definition| span_contains(definition.span, offset))
            .min_by_key(|definition| definition.span.1.saturating_sub(definition.span.0))
    }

    pub fn reference_at_offset(&self, offset: usize) -> Option<&Reference> {
        self.references
            .iter()
            .filter(|reference| span_contains(reference.span, offset))
            .min_by_key(|reference| reference.span.1.saturating_sub(reference.span.0))
    }

    pub fn scope_at(&self, offset: usize) -> usize {
        self.scopes
            .iter()
            .filter(|scope| span_contains(scope.span, offset))
            .min_by_key(|scope| scope.span.1.saturating_sub(scope.span.0))
            .map_or(0, |scope| scope.id)
    }

    pub fn resolve_visible_definition(
        &self,
        name: &str,
        scope_id: usize,
        offset: usize,
    ) -> Option<&Definition> {
        resolve_definition_id(
            &self.definitions,
            &self.definitions_by_scope,
            &self.scopes,
            name,
            scope_id,
            offset,
        )
        .and_then(|id| self.definition_by_id(id))
    }

    pub fn references_for_definition(&self, definition_id: usize) -> Vec<&Reference> {
        self.references
            .iter()
            .filter(|reference| reference.resolved_definition == Some(definition_id))
            .collect()
    }

    pub fn visible_definitions(&self, scope_id: usize, offset: usize) -> Vec<&Definition> {
        let mut visible = Vec::new();
        let mut seen_names = HashSet::new();

        for scoped_id in scope_chain(&self.scopes, scope_id) {
            let Some(indices) = self.definitions_by_scope.get(&scoped_id) else {
                continue;
            };

            let mut definitions_in_scope = indices
                .iter()
                .map(|index| &self.definitions[*index])
                .filter(|definition| definition.hoisted || definition.declaration_start <= offset)
                .collect::<Vec<_>>();

            definitions_in_scope.sort_by_key(|definition| definition.declaration_start);
            definitions_in_scope.reverse();

            for definition in definitions_in_scope {
                if seen_names.insert(definition.name.clone()) {
                    visible.push(definition);
                }
            }
        }

        visible
    }

    fn build_indices(&mut self) {
        self.definitions_by_scope.clear();
        for (index, definition) in self.definitions.iter().enumerate() {
            self.definitions_by_scope
                .entry(definition.scope_id)
                .or_default()
                .push(index);
        }
    }

    fn resolve_references(&mut self) {
        for reference in &mut self.references {
            reference.resolved_definition = resolve_definition_id(
                &self.definitions,
                &self.definitions_by_scope,
                &self.scopes,
                &reference.name,
                reference.scope_id,
                reference.span.0,
            );
        }
    }
}

pub fn analyze(source: &str) -> Analysis {
    let parser = ProgramParser::new();
    match parser.parse(source) {
        Ok(ast) => {
            let mut analyzer = Analyzer::new(source);
            analyzer.walk(&ast);

            let mut analysis = Analysis {
                diagnostics: Vec::new(),
                definitions: analyzer.definitions,
                references: analyzer.references,
                scopes: analyzer.scopes,
                definitions_by_scope: HashMap::new(),
            };

            analysis.build_indices();
            analysis.resolve_references();
            analysis
        }
        Err(err) => {
            let diagnostic = parse_error_to_diagnostic(err, source);
            Analysis::empty_with_diagnostics(vec![diagnostic], source.len())
        }
    }
}

fn resolve_definition_id(
    definitions: &[Definition],
    definitions_by_scope: &HashMap<usize, Vec<usize>>,
    scopes: &[ScopeInfo],
    name: &str,
    scope_id: usize,
    offset: usize,
) -> Option<usize> {
    for scoped_id in scope_chain(scopes, scope_id) {
        let Some(indices) = definitions_by_scope.get(&scoped_id) else {
            continue;
        };

        let mut best: Option<&Definition> = None;

        for index in indices {
            let definition = &definitions[*index];
            if definition.name != name {
                continue;
            }

            if !definition.hoisted && definition.declaration_start > offset {
                continue;
            }

            if best.is_none_or(|current| definition.declaration_start >= current.declaration_start)
            {
                best = Some(definition);
            }
        }

        if let Some(definition) = best {
            return Some(definition.id);
        }
    }

    None
}

fn scope_chain(scopes: &[ScopeInfo], start_scope: usize) -> Vec<usize> {
    let mut chain = Vec::new();
    let mut current = Some(start_scope);

    while let Some(scope_id) = current {
        chain.push(scope_id);
        current = scopes.get(scope_id).and_then(|scope| scope.parent);
    }

    chain
}

#[derive(Debug)]
struct Analyzer<'a> {
    source: &'a str,
    line_starts: Vec<usize>,
    scopes: Vec<ScopeInfo>,
    definitions: Vec<Definition>,
    references: Vec<Reference>,
    current_scope: usize,
    container_stack: Vec<usize>,
}

impl<'a> Analyzer<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source,
            line_starts: line_starts(source),
            scopes: vec![ScopeInfo {
                id: 0,
                parent: None,
                span: Span(0, source.len()),
            }],
            definitions: Vec::new(),
            references: Vec::new(),
            current_scope: 0,
            container_stack: Vec::new(),
        }
    }

    fn push_scope(&mut self, span: Span) -> usize {
        let scope_id = self.scopes.len();
        self.scopes.push(ScopeInfo {
            id: scope_id,
            parent: Some(self.current_scope),
            span: normalize_span(self.source, span),
        });
        scope_id
    }

    fn with_scope<T>(&mut self, scope_id: usize, f: impl FnOnce(&mut Self) -> T) -> T {
        let previous_scope = self.current_scope;
        self.current_scope = scope_id;
        let result = f(self);
        self.current_scope = previous_scope;
        result
    }

    fn with_container<T>(&mut self, definition_id: usize, f: impl FnOnce(&mut Self) -> T) -> T {
        self.container_stack.push(definition_id);
        let result = f(self);
        self.container_stack.pop();
        result
    }

    fn current_container(&self) -> Option<usize> {
        self.container_stack.last().copied()
    }

    fn add_definition(
        &mut self,
        name: &str,
        node_span: Span,
        kind: SymbolKind,
        hoisted: bool,
        detail: Option<String>,
        documentation_offset: usize,
        parameters: Vec<String>,
    ) -> usize {
        let span = find_name_span(self.source, node_span, name);
        self.add_definition_with_span(
            name,
            span,
            node_span,
            kind,
            hoisted,
            detail,
            documentation_offset,
            parameters,
        )
    }

    fn add_definition_with_span(
        &mut self,
        name: &str,
        span: Span,
        full_span: Span,
        kind: SymbolKind,
        hoisted: bool,
        detail: Option<String>,
        documentation_offset: usize,
        parameters: Vec<String>,
    ) -> usize {
        let normalized_span = normalize_span(self.source, span);
        let normalized_full_span = normalize_span(self.source, full_span);
        let id = self.definitions.len();
        let documentation = self.docs_before_offset(documentation_offset);

        self.definitions.push(Definition {
            id,
            name: name.to_string(),
            span: normalized_span,
            full_span: normalized_full_span,
            kind,
            scope_id: self.current_scope,
            declaration_start: normalized_span.0,
            hoisted,
            detail,
            documentation,
            parameters,
            container: self.current_container(),
        });

        id
    }

    fn add_reference(&mut self, name: &str, span: Span, is_write: bool) {
        self.references.push(Reference {
            name: name.to_string(),
            span: normalize_span(self.source, span),
            scope_id: self.current_scope,
            is_write,
            resolved_definition: None,
        });
    }

    fn docs_before_offset(&self, offset: usize) -> Option<String> {
        let offset = offset.min(self.source.len());
        let current_line = self
            .line_starts
            .partition_point(|line_start| *line_start <= offset)
            .saturating_sub(1);

        let mut docs = Vec::new();
        let mut line_index = current_line as isize - 1;
        let mut saw_comment = false;

        while line_index >= 0 {
            let index = line_index as usize;
            let start = self.line_starts[index];
            let end = if index + 1 < self.line_starts.len() {
                self.line_starts[index + 1].saturating_sub(1)
            } else {
                self.source.len()
            };

            let line = self.source[start..end].trim();

            if let Some(doc) = line.strip_prefix("///") {
                saw_comment = true;
                docs.push(doc.trim().to_string());
                line_index -= 1;
                continue;
            }

            if let Some(doc) = line.strip_prefix("//") {
                saw_comment = true;
                docs.push(doc.trim().to_string());
                line_index -= 1;
                continue;
            }

            if line.is_empty() {
                if saw_comment {
                    line_index -= 1;
                    continue;
                }
                line_index -= 1;
                continue;
            }

            break;
        }

        if docs.is_empty() {
            None
        } else {
            docs.reverse();
            Some(docs.join("\n"))
        }
    }

    fn walk(&mut self, node: &Node) {
        match node.kind() {
            NodeKind::Program(nodes)
            | NodeKind::Statements(nodes)
            | NodeKind::UnscopedStatements(nodes) => {
                for child in nodes {
                    self.walk(child);
                }
            }
            NodeKind::Block(nodes) => {
                let scope_id = self.push_scope(*node.span());
                self.with_scope(scope_id, |analyzer| {
                    for child in nodes {
                        analyzer.walk(child);
                    }
                });
            }
            NodeKind::FunctionDefinition(name, args, body) => {
                let signature = function_signature(name, args);
                let definition_id = self.add_definition(
                    name,
                    *node.span(),
                    SymbolKind::Function,
                    true,
                    Some(signature),
                    node.span().0,
                    args.clone(),
                );

                let function_scope = self.push_scope(*body.span());
                self.with_scope(function_scope, |analyzer| {
                    analyzer.with_container(definition_id, |analyzer| {
                        let header_span = Span(node.span().0, body.span().0.min(node.span().1));
                        let parameter_spans =
                            find_parameter_spans(analyzer.source, header_span, args);

                        for (index, parameter) in args.iter().enumerate() {
                            let span = parameter_spans.get(index).copied().unwrap_or_else(|| {
                                find_name_span(analyzer.source, header_span, parameter)
                            });
                            analyzer.add_definition_with_span(
                                parameter,
                                span,
                                span,
                                SymbolKind::Parameter,
                                false,
                                Some("parameter".to_string()),
                                span.0,
                                Vec::new(),
                            );
                        }

                        analyzer.walk(body);
                    });
                });
            }
            NodeKind::StructDefinition(name, members) => {
                let definition_id = self.add_definition(
                    name,
                    *node.span(),
                    SymbolKind::Struct,
                    true,
                    Some(format!("struct {name}")),
                    node.span().0,
                    Vec::new(),
                );

                let struct_scope = self.push_scope(*node.span());
                self.with_scope(struct_scope, |analyzer| {
                    analyzer.with_container(definition_id, |analyzer| {
                        for member in members {
                            analyzer.walk(member);
                        }
                    });
                });
            }
            NodeKind::StructFunctionDefinition(name, args, body) => {
                let signature = method_signature(name, args);
                let definition_id = self.add_definition(
                    name,
                    *node.span(),
                    SymbolKind::Method,
                    true,
                    Some(signature),
                    node.span().0,
                    args.clone(),
                );

                let method_scope = self.push_scope(*body.span());
                self.with_scope(method_scope, |analyzer| {
                    analyzer.with_container(definition_id, |analyzer| {
                        let header_span = Span(node.span().0, body.span().0.min(node.span().1));
                        let parameter_spans =
                            find_parameter_spans(analyzer.source, header_span, args);

                        for (index, parameter) in args.iter().enumerate() {
                            let span = parameter_spans.get(index).copied().unwrap_or_else(|| {
                                find_name_span(analyzer.source, header_span, parameter)
                            });
                            analyzer.add_definition_with_span(
                                parameter,
                                span,
                                span,
                                SymbolKind::Parameter,
                                false,
                                Some("parameter".to_string()),
                                span.0,
                                Vec::new(),
                            );
                        }

                        analyzer.walk(body);
                    });
                });
            }
            NodeKind::ExternFunctionDefinition(name, args) => {
                self.add_definition(
                    name,
                    *node.span(),
                    SymbolKind::Function,
                    true,
                    Some(extern_signature(name, args)),
                    node.span().0,
                    args.clone(),
                );
            }
            NodeKind::Assign(name, value) => {
                self.walk(value);
                self.add_definition(
                    name,
                    *node.span(),
                    SymbolKind::Variable,
                    false,
                    Some(format!("let {name}")),
                    node.span().0,
                    Vec::new(),
                );
            }
            NodeKind::Reassign(name, value, _) => {
                self.add_reference(name.value(), name.span(), true);
                self.walk(value);
            }
            NodeKind::For(var, iter, body) => {
                self.walk(iter);
                let loop_scope = self.push_scope(*body.span());
                self.with_scope(loop_scope, |analyzer| {
                    analyzer.add_definition(
                        var,
                        *node.span(),
                        SymbolKind::Variable,
                        false,
                        Some(format!("let {var}")),
                        node.span().0,
                        Vec::new(),
                    );
                    analyzer.walk(body);
                });
            }
            NodeKind::Try(try_block, catch_var, catch_block) => {
                self.walk(try_block);
                let catch_scope = self.push_scope(*catch_block.span());
                self.with_scope(catch_scope, |analyzer| {
                    analyzer.add_definition(
                        catch_var,
                        *catch_block.span(),
                        SymbolKind::Variable,
                        false,
                        Some(format!("let {catch_var}")),
                        catch_block.span().0,
                        Vec::new(),
                    );
                    analyzer.walk(catch_block);
                });
            }
            NodeKind::PackageImport(package, alias) => {
                let binding = alias.clone().unwrap_or_else(|| package.clone());
                self.add_definition(
                    &binding,
                    *node.span(),
                    SymbolKind::Module,
                    true,
                    Some(format!("import @{package}")),
                    node.span().0,
                    Vec::new(),
                );
            }
            NodeKind::ModuleImport(module, alias) => {
                let binding = alias
                    .clone()
                    .unwrap_or_else(|| module_default_binding(module));
                self.add_definition(
                    &binding,
                    *node.span(),
                    SymbolKind::Module,
                    true,
                    Some(format!("import \"{module}\"")),
                    node.span().0,
                    Vec::new(),
                );
            }
            NodeKind::Return(value)
            | NodeKind::Print(value)
            | NodeKind::Println(value)
            | NodeKind::Throw(value)
            | NodeKind::Free(value)
            | NodeKind::Defer(value)
            | NodeKind::ExpressionStatement(value)
            | NodeKind::UnaryOp(_, value) => {
                self.walk(value);
            }
            NodeKind::If(condition, then_branch, else_branch) => {
                self.walk(condition);
                let then_scope = self.push_scope(*then_branch.span());
                self.with_scope(then_scope, |analyzer| {
                    analyzer.walk(then_branch);
                });
                if let Some(else_node) = else_branch {
                    let else_scope = self.push_scope(*else_node.span());
                    self.with_scope(else_scope, |analyzer| {
                        analyzer.walk(else_node);
                    });
                }
            }
            NodeKind::While(condition, body) => {
                self.walk(condition);
                let body_scope = self.push_scope(*body.span());
                self.with_scope(body_scope, |analyzer| {
                    analyzer.walk(body);
                });
            }
            NodeKind::Ternary(value, condition, else_value) => {
                self.walk(value);
                self.walk(condition);
                self.walk(else_value);
            }
            NodeKind::BinOp(left, _, right)
            | NodeKind::Index(left, right)
            | NodeKind::Range(Some(left), right) => {
                self.walk(left);
                self.walk(right);
            }
            NodeKind::Range(None, right) => {
                self.walk(right);
            }
            NodeKind::IndexAssign(target, index, value) => {
                self.walk(target);
                self.walk(index);
                self.walk(value);
            }
            NodeKind::FunctionCall(callee, args) => {
                self.walk(callee);
                for arg in args {
                    self.walk(arg);
                }
            }
            NodeKind::MemberAccess(target, _member) => {
                self.walk(target);
            }
            NodeKind::List(items) => {
                for item in items {
                    self.walk(item);
                }
            }
            NodeKind::Dict(entries) => {
                for (key, value) in entries {
                    self.walk(key);
                    self.walk(value);
                }
            }
            NodeKind::FString(parts) => {
                for part in parts {
                    if let FStringPart::Expr(expr) = part {
                        self.walk(expr);
                    }
                }
            }
            NodeKind::AnonFunctionDefinition(args, body) => {
                let function_scope = self.push_scope(*body.span());
                self.with_scope(function_scope, |analyzer| {
                    let header_span = Span(node.span().0, body.span().0.min(node.span().1));
                    let parameter_spans = find_parameter_spans(analyzer.source, header_span, args);

                    for (index, parameter) in args.iter().enumerate() {
                        let span = parameter_spans.get(index).copied().unwrap_or_else(|| {
                            find_name_span(analyzer.source, header_span, parameter)
                        });
                        analyzer.add_definition_with_span(
                            parameter,
                            span,
                            span,
                            SymbolKind::Parameter,
                            false,
                            Some("parameter".to_string()),
                            span.0,
                            Vec::new(),
                        );
                    }

                    analyzer.walk(body);
                });
            }
            NodeKind::Ident(name) => {
                self.add_reference(name, *node.span(), false);
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

fn function_signature(name: &str, args: &[String]) -> String {
    if args.is_empty() {
        format!("fn {name}")
    } else {
        format!("fn {name} : {}", args.join(", "))
    }
}

fn method_signature(name: &str, args: &[String]) -> String {
    if args.is_empty() {
        format!("fn {name}")
    } else {
        format!("fn {name} : {}", args.join(", "))
    }
}

fn extern_signature(name: &str, args: &[String]) -> String {
    if args.is_empty() {
        format!("extern fn {name}")
    } else {
        format!("extern fn {name} : {}", args.join(", "))
    }
}

fn module_default_binding(module_path: &str) -> String {
    module_path
        .rsplit('/')
        .next()
        .filter(|segment| !segment.is_empty())
        .unwrap_or(module_path)
        .to_string()
}

fn find_parameter_spans(source: &str, header_span: Span, params: &[String]) -> Vec<Span> {
    let mut spans = Vec::with_capacity(params.len());
    let mut search_start = header_span.0.min(source.len());
    let search_end = header_span.1.min(source.len());

    for parameter in params {
        let span = find_identifier_occurrence(source, search_start, search_end, parameter)
            .unwrap_or_else(|| fallback_span(source, search_start, parameter.len()));
        search_start = span.1.min(search_end);
        spans.push(span);
    }

    spans
}

fn find_name_span(source: &str, node_span: Span, name: &str) -> Span {
    let start = node_span.0.min(source.len());
    let end = node_span.1.min(source.len());

    find_identifier_occurrence(source, start, end, name)
        .unwrap_or_else(|| fallback_span(source, start, name.len()))
}

fn find_identifier_occurrence(source: &str, start: usize, end: usize, name: &str) -> Option<Span> {
    if name.is_empty() || start >= end || start >= source.len() {
        return None;
    }

    let end = end.min(source.len());
    let slice = &source[start..end];
    let bytes = source.as_bytes();

    for (relative_index, _) in slice.match_indices(name) {
        let absolute_start = start + relative_index;
        let absolute_end = absolute_start + name.len();
        if absolute_end > source.len() {
            continue;
        }

        let left_ok = absolute_start == 0 || !is_ident_byte(bytes[absolute_start - 1]);
        let right_ok = absolute_end == source.len() || !is_ident_byte(bytes[absolute_end]);

        if left_ok && right_ok {
            return Some(Span(absolute_start, absolute_end));
        }
    }

    None
}

fn fallback_span(source: &str, start: usize, len: usize) -> Span {
    let normalized_start = start.min(source.len());
    let normalized_end = normalized_start.saturating_add(len).min(source.len());
    Span(normalized_start, normalized_end.max(normalized_start))
}

fn line_starts(source: &str) -> Vec<usize> {
    let mut starts = vec![0usize];
    for (index, ch) in source.char_indices() {
        if ch == '\n' {
            starts.push(index + 1);
        }
    }
    starts
}

fn is_ident_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
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

fn span_contains(span: Span, offset: usize) -> bool {
    span.0 <= offset && offset <= span.1
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

#[cfg(test)]
mod tests {
    use super::{SymbolKind, analyze};

    #[test]
    fn resolves_shadowed_variables_to_nearest_scope() {
        let source = r#"
let x = 1;
if true {
    let x = 2;
    let y = x;
}
let z = x;
"#;

        let analysis = analyze(source);
        assert!(analysis.diagnostics.is_empty());

        let x_defs = analysis
            .definitions
            .iter()
            .filter(|definition| definition.name == "x" && definition.kind == SymbolKind::Variable)
            .collect::<Vec<_>>();
        assert_eq!(x_defs.len(), 2);

        let inner_use = source.find("let y = x;").unwrap() + "let y = ".len();
        let outer_use = source.find("let z = x;").unwrap() + "let z = ".len();

        let inner_reference = analysis.reference_at_offset(inner_use).unwrap();
        let outer_reference = analysis.reference_at_offset(outer_use).unwrap();

        let inner_definition = analysis
            .definition_by_id(inner_reference.resolved_definition.unwrap())
            .unwrap();
        let outer_definition = analysis
            .definition_by_id(outer_reference.resolved_definition.unwrap())
            .unwrap();

        assert!(inner_definition.declaration_start > outer_definition.declaration_start);
    }

    #[test]
    fn extracts_function_signature_and_docs() {
        let source = r#"
// Adds numbers
fn add : a, b {
    return a + b;
}
let total = add(1, 2);
"#;

        let analysis = analyze(source);
        assert!(analysis.diagnostics.is_empty());

        let add_definition = analysis
            .definitions
            .iter()
            .find(|definition| definition.name == "add" && definition.kind == SymbolKind::Function)
            .unwrap();

        assert_eq!(
            add_definition.parameters,
            vec!["a".to_string(), "b".to_string()]
        );
        assert_eq!(add_definition.detail.as_deref(), Some("fn add : a, b"));
        assert_eq!(
            add_definition.documentation.as_deref(),
            Some("Adds numbers")
        );
    }
}
