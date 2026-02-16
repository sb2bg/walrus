use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tokio::sync::RwLock;
use tower_lsp::jsonrpc::{Error as JsonRpcError, ErrorCode, Result};
use tower_lsp::lsp_types::{
    CompletionItem, CompletionItemKind, CompletionOptions, CompletionResponse, Diagnostic,
    DiagnosticSeverity, DidChangeTextDocumentParams, DidCloseTextDocumentParams,
    DidOpenTextDocumentParams, DidSaveTextDocumentParams, DocumentHighlight, DocumentHighlightKind,
    DocumentHighlightParams, DocumentSymbol, DocumentSymbolParams, DocumentSymbolResponse,
    Documentation, GotoDefinitionParams, GotoDefinitionResponse, Hover, HoverContents, HoverParams,
    HoverProviderCapability, InitializeParams, InitializeResult, InsertTextFormat, Location,
    MarkupContent, MarkupKind, MessageType, OneOf, ParameterInformation, ParameterLabel, Position,
    PrepareRenameResponse, Range, ReferenceParams, RenameOptions, RenameParams, ServerCapabilities,
    SignatureHelp, SignatureHelpOptions, SignatureHelpParams, SignatureInformation,
    SymbolKind as LspSymbolKind, TextDocumentSyncCapability, TextDocumentSyncKind, TextEdit, Url,
    WorkspaceEdit,
};
use tower_lsp::{Client, LanguageServer, LspService, Server};
use walrus::lsp_support::{
    Analysis, Definition, DiagnosticSeverity as WalrusDiagnosticSeverity, ParseDiagnostic,
    SymbolKind as WalrusSymbolKind, analyze,
};
use walrus::span::Span;

const KEYWORDS: &[&str] = &[
    "let", "fn", "struct", "return", "if", "else", "while", "for", "in", "break", "continue",
    "import", "as", "true", "false", "void", "try", "catch", "throw", "free", "defer", "extern",
    "and", "or", "not", "start", "end", "print", "println",
];

struct BuiltinInfo {
    name: &'static str,
    signature: &'static str,
    docs: &'static str,
    params: &'static [&'static str],
}

const BUILTINS: &[BuiltinInfo] = &[
    BuiltinInfo {
        name: "len",
        signature: "len(value)",
        docs: "Returns length of a string, list, or dictionary.",
        params: &["value"],
    },
    BuiltinInfo {
        name: "str",
        signature: "str(value)",
        docs: "Converts a value to string.",
        params: &["value"],
    },
    BuiltinInfo {
        name: "type",
        signature: "type(value)",
        docs: "Returns the runtime type name for a value.",
        params: &["value"],
    },
    BuiltinInfo {
        name: "input",
        signature: "input(prompt)",
        docs: "Reads user input from stdin.",
        params: &["prompt"],
    },
    BuiltinInfo {
        name: "__gc__",
        signature: "__gc__()",
        docs: "Manually triggers garbage collection.",
        params: &[],
    },
    BuiltinInfo {
        name: "__heap_stats__",
        signature: "__heap_stats__()",
        docs: "Returns heap statistics as a dictionary.",
        params: &[],
    },
    BuiltinInfo {
        name: "__gc_threshold__",
        signature: "__gc_threshold__(threshold)",
        docs: "Sets garbage-collection allocation threshold.",
        params: &["threshold"],
    },
];

#[derive(Debug, Clone)]
struct DocumentState {
    text: String,
    analysis: Analysis,
}

#[derive(Debug)]
struct Backend {
    client: Client,
    docs: Arc<RwLock<HashMap<Url, DocumentState>>>,
}

impl Backend {
    fn new(client: Client) -> Self {
        Self {
            client,
            docs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn set_document(&self, uri: Url, text: String) {
        let analysis = analyze(&text);
        let diagnostics = analysis
            .diagnostics
            .iter()
            .map(|diagnostic| diagnostic_to_lsp(diagnostic, &text))
            .collect::<Vec<_>>();

        {
            let mut docs = self.docs.write().await;
            docs.insert(uri.clone(), DocumentState { text, analysis });
        }

        self.client
            .publish_diagnostics(uri, diagnostics, None)
            .await;
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            server_info: None,
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                completion_provider: Some(CompletionOptions {
                    resolve_provider: Some(false),
                    trigger_characters: Some(vec![".".to_string()]),
                    ..CompletionOptions::default()
                }),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                definition_provider: Some(OneOf::Left(true)),
                references_provider: Some(OneOf::Left(true)),
                rename_provider: Some(OneOf::Right(RenameOptions {
                    prepare_provider: Some(true),
                    work_done_progress_options: Default::default(),
                })),
                document_symbol_provider: Some(OneOf::Left(true)),
                document_highlight_provider: Some(OneOf::Left(true)),
                signature_help_provider: Some(SignatureHelpOptions {
                    trigger_characters: Some(vec!["(".to_string(), ",".to_string()]),
                    retrigger_characters: Some(vec![",".to_string()]),
                    work_done_progress_options: Default::default(),
                }),
                ..ServerCapabilities::default()
            },
        })
    }

    async fn initialized(&self, _: tower_lsp::lsp_types::InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "Walrus LSP initialized")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        self.set_document(params.text_document.uri, params.text_document.text)
            .await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        if let Some(change) = params.content_changes.into_iter().last() {
            self.set_document(params.text_document.uri, change.text)
                .await;
        }
    }

    async fn did_save(&self, params: DidSaveTextDocumentParams) {
        if let Some(text) = params.text {
            self.set_document(params.text_document.uri, text).await;
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        {
            let mut docs = self.docs.write().await;
            docs.remove(&params.text_document.uri);
        }

        self.client
            .publish_diagnostics(params.text_document.uri, Vec::new(), None)
            .await;
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;
        let docs = self.docs.read().await;
        let Some(doc) = docs.get(uri) else {
            return Ok(None);
        };

        let Some(offset) = position_to_byte(&doc.text, position) else {
            return Ok(None);
        };

        if let Some((identifier, span)) = identifier_at_offset(&doc.text, offset) {
            if let Some(keyword_docs) = keyword_docs(&identifier) {
                let markdown = format!("```walrus\n{identifier}\n```\n\n{keyword_docs}");
                return Ok(Some(Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: markdown,
                    }),
                    range: Some(span_to_range(&doc.text, span)),
                }));
            }

            if let Some(builtin) = builtin_by_name(&identifier) {
                let markdown = format!("```walrus\n{}\n```\n\n{}", builtin.signature, builtin.docs);
                return Ok(Some(Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: markdown,
                    }),
                    range: Some(span_to_range(&doc.text, span)),
                }));
            }
        }

        let Some(definition) = definition_at_offset(&doc.analysis, offset) else {
            return Ok(None);
        };

        let markdown = hover_markdown_for_definition(definition);
        let range = doc
            .analysis
            .definition_at_offset(offset)
            .map_or(definition.span, |def| def.span);

        Ok(Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: markdown,
            }),
            range: Some(span_to_range(&doc.text, range)),
        }))
    }

    async fn completion(
        &self,
        params: tower_lsp::lsp_types::CompletionParams,
    ) -> Result<Option<CompletionResponse>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;
        let docs = self.docs.read().await;
        let Some(doc) = docs.get(uri) else {
            return Ok(None);
        };

        let Some(offset) = position_to_byte(&doc.text, position) else {
            return Ok(None);
        };

        let scope_id = doc.analysis.scope_at(offset);
        let mut items = Vec::new();
        let mut seen = HashSet::new();

        for definition in doc.analysis.visible_definitions(scope_id, offset) {
            if seen.insert(definition.name.clone()) {
                items.push(completion_for_definition(definition));
            }
        }

        for builtin in BUILTINS {
            if seen.insert(builtin.name.to_string()) {
                items.push(completion_for_builtin(builtin));
            }
        }

        for keyword in KEYWORDS {
            if seen.insert((*keyword).to_string()) {
                items.push(CompletionItem {
                    label: (*keyword).to_string(),
                    kind: Some(CompletionItemKind::KEYWORD),
                    detail: Some("keyword".to_string()),
                    documentation: keyword_docs(keyword).map(documentation_from_text),
                    sort_text: Some(format!("2_{keyword}")),
                    ..CompletionItem::default()
                });
            }
        }

        Ok(Some(CompletionResponse::Array(items)))
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;
        let docs = self.docs.read().await;
        let Some(doc) = docs.get(uri) else {
            return Ok(None);
        };

        let Some(offset) = position_to_byte(&doc.text, position) else {
            return Ok(None);
        };

        let Some(definition) = definition_at_offset(&doc.analysis, offset) else {
            return Ok(None);
        };

        Ok(Some(GotoDefinitionResponse::Scalar(Location::new(
            uri.clone(),
            span_to_range(&doc.text, definition.span),
        ))))
    }

    async fn references(&self, params: ReferenceParams) -> Result<Option<Vec<Location>>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;
        let include_declaration = params.context.include_declaration;
        let docs = self.docs.read().await;
        let Some(doc) = docs.get(uri) else {
            return Ok(None);
        };

        let Some(offset) = position_to_byte(&doc.text, position) else {
            return Ok(None);
        };

        let Some(definition) = definition_at_offset(&doc.analysis, offset) else {
            return Ok(None);
        };

        let mut locations = Vec::new();
        if include_declaration {
            locations.push(Location::new(
                uri.clone(),
                span_to_range(&doc.text, definition.span),
            ));
        }

        for reference in doc.analysis.references_for_definition(definition.id) {
            locations.push(Location::new(
                uri.clone(),
                span_to_range(&doc.text, reference.span),
            ));
        }

        locations.sort_by_key(|location| {
            (
                location.range.start.line,
                location.range.start.character,
                location.range.end.line,
                location.range.end.character,
            )
        });

        Ok(Some(locations))
    }

    async fn document_highlight(
        &self,
        params: DocumentHighlightParams,
    ) -> Result<Option<Vec<DocumentHighlight>>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;
        let docs = self.docs.read().await;
        let Some(doc) = docs.get(uri) else {
            return Ok(None);
        };

        let Some(offset) = position_to_byte(&doc.text, position) else {
            return Ok(None);
        };

        let Some(definition) = definition_at_offset(&doc.analysis, offset) else {
            return Ok(None);
        };

        let mut highlights = Vec::new();
        highlights.push(DocumentHighlight {
            range: span_to_range(&doc.text, definition.span),
            kind: Some(DocumentHighlightKind::WRITE),
        });

        for reference in doc.analysis.references_for_definition(definition.id) {
            highlights.push(DocumentHighlight {
                range: span_to_range(&doc.text, reference.span),
                kind: Some(if reference.is_write {
                    DocumentHighlightKind::WRITE
                } else {
                    DocumentHighlightKind::READ
                }),
            });
        }

        Ok(Some(highlights))
    }

    async fn prepare_rename(
        &self,
        params: tower_lsp::lsp_types::TextDocumentPositionParams,
    ) -> Result<Option<PrepareRenameResponse>> {
        let uri = &params.text_document.uri;
        let position = params.position;
        let docs = self.docs.read().await;
        let Some(doc) = docs.get(uri) else {
            return Ok(None);
        };

        let Some(offset) = position_to_byte(&doc.text, position) else {
            return Ok(None);
        };

        if let Some(definition) = doc.analysis.definition_at_offset(offset) {
            return Ok(Some(PrepareRenameResponse::RangeWithPlaceholder {
                range: span_to_range(&doc.text, definition.span),
                placeholder: definition.name.clone(),
            }));
        }

        if let Some(reference) = doc.analysis.reference_at_offset(offset) {
            return Ok(Some(PrepareRenameResponse::RangeWithPlaceholder {
                range: span_to_range(&doc.text, reference.span),
                placeholder: reference.name.clone(),
            }));
        }

        Ok(None)
    }

    async fn rename(&self, params: RenameParams) -> Result<Option<WorkspaceEdit>> {
        if !is_valid_identifier(&params.new_name) {
            return Err(JsonRpcError {
                code: ErrorCode::InvalidParams,
                message: "New name is not a valid Walrus identifier.".into(),
                data: None,
            });
        }

        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;
        let docs = self.docs.read().await;
        let Some(doc) = docs.get(uri) else {
            return Ok(None);
        };

        let Some(offset) = position_to_byte(&doc.text, position) else {
            return Ok(None);
        };

        let Some(definition) = definition_at_offset(&doc.analysis, offset) else {
            return Ok(None);
        };

        let mut edits = Vec::new();
        let mut seen_ranges = HashSet::new();

        if seen_ranges.insert((definition.span.0, definition.span.1)) {
            edits.push(TextEdit {
                range: span_to_range(&doc.text, definition.span),
                new_text: params.new_name.clone(),
            });
        }

        for reference in doc.analysis.references_for_definition(definition.id) {
            if seen_ranges.insert((reference.span.0, reference.span.1)) {
                edits.push(TextEdit {
                    range: span_to_range(&doc.text, reference.span),
                    new_text: params.new_name.clone(),
                });
            }
        }

        if edits.is_empty() {
            return Ok(None);
        }

        let mut changes = HashMap::new();
        changes.insert(uri.clone(), edits);

        Ok(Some(WorkspaceEdit {
            changes: Some(changes),
            document_changes: None,
            change_annotations: None,
        }))
    }

    async fn signature_help(&self, params: SignatureHelpParams) -> Result<Option<SignatureHelp>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;
        let docs = self.docs.read().await;
        let Some(doc) = docs.get(uri) else {
            return Ok(None);
        };

        let Some(offset) = position_to_byte(&doc.text, position) else {
            return Ok(None);
        };

        let Some(context) = call_context_at_offset(&doc.text, offset) else {
            return Ok(None);
        };

        let scope_id = doc.analysis.scope_at(context.open_paren_offset);
        if let Some(definition) = doc.analysis.resolve_visible_definition(
            &context.callee_name,
            scope_id,
            context.open_paren_offset,
        ) {
            if matches!(
                definition.kind,
                WalrusSymbolKind::Function | WalrusSymbolKind::Method
            ) {
                let signature = SignatureInformation {
                    label: definition_signature(definition),
                    documentation: definition
                        .documentation
                        .as_deref()
                        .map(documentation_from_text),
                    parameters: Some(
                        definition
                            .parameters
                            .iter()
                            .map(|parameter| ParameterInformation {
                                label: ParameterLabel::Simple(parameter.clone()),
                                documentation: None,
                            })
                            .collect(),
                    ),
                    active_parameter: None,
                };

                let active_parameter = if definition.parameters.is_empty() {
                    0
                } else {
                    context
                        .arg_index
                        .min(definition.parameters.len().saturating_sub(1))
                };

                return Ok(Some(SignatureHelp {
                    signatures: vec![signature],
                    active_signature: Some(0),
                    active_parameter: Some(active_parameter as u32),
                }));
            }
        }

        if let Some(builtin) = builtin_by_name(&context.callee_name) {
            let signature = SignatureInformation {
                label: builtin.signature.to_string(),
                documentation: Some(documentation_from_text(builtin.docs)),
                parameters: Some(
                    builtin
                        .params
                        .iter()
                        .map(|parameter| ParameterInformation {
                            label: ParameterLabel::Simple((*parameter).to_string()),
                            documentation: None,
                        })
                        .collect(),
                ),
                active_parameter: None,
            };

            let active_parameter = if builtin.params.is_empty() {
                0
            } else {
                context
                    .arg_index
                    .min(builtin.params.len().saturating_sub(1))
            };

            return Ok(Some(SignatureHelp {
                signatures: vec![signature],
                active_signature: Some(0),
                active_parameter: Some(active_parameter as u32),
            }));
        }

        Ok(None)
    }

    async fn document_symbol(
        &self,
        params: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        let uri = &params.text_document.uri;
        let docs = self.docs.read().await;
        let Some(doc) = docs.get(uri) else {
            return Ok(None);
        };

        let symbols = build_document_symbols(&doc.analysis, &doc.text);
        Ok(Some(DocumentSymbolResponse::Nested(symbols)))
    }
}

fn completion_for_definition(definition: &Definition) -> CompletionItem {
    let mut item = CompletionItem {
        label: definition.name.clone(),
        kind: Some(completion_item_kind(definition.kind)),
        detail: definition.detail.clone(),
        documentation: definition
            .documentation
            .as_deref()
            .map(documentation_from_text),
        sort_text: Some(format!("0_{}", definition.name)),
        ..CompletionItem::default()
    };

    if matches!(
        definition.kind,
        WalrusSymbolKind::Function | WalrusSymbolKind::Method
    ) {
        item.insert_text = Some(function_snippet(&definition.name, &definition.parameters));
        item.insert_text_format = Some(InsertTextFormat::SNIPPET);
    }

    item
}

fn completion_for_builtin(builtin: &BuiltinInfo) -> CompletionItem {
    CompletionItem {
        label: builtin.name.to_string(),
        kind: Some(CompletionItemKind::FUNCTION),
        detail: Some("builtin".to_string()),
        documentation: Some(documentation_from_text(builtin.docs)),
        insert_text: Some(function_snippet(builtin.name, builtin.params)),
        insert_text_format: Some(InsertTextFormat::SNIPPET),
        sort_text: Some(format!("1_{}", builtin.name)),
        ..CompletionItem::default()
    }
}

fn function_snippet(name: &str, params: &[impl AsRef<str>]) -> String {
    if params.is_empty() {
        format!("{name}($0)")
    } else {
        let placeholders = params
            .iter()
            .enumerate()
            .map(|(index, param)| format!("${{{}:{}}}", index + 1, param.as_ref()))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{name}({placeholders})")
    }
}

fn definition_at_offset<'a>(analysis: &'a Analysis, offset: usize) -> Option<&'a Definition> {
    if let Some(definition) = analysis.definition_at_offset(offset) {
        return Some(definition);
    }

    analysis
        .reference_at_offset(offset)
        .and_then(|reference| reference.resolved_definition)
        .and_then(|definition_id| analysis.definition_by_id(definition_id))
}

fn hover_markdown_for_definition(definition: &Definition) -> String {
    let mut sections = Vec::new();
    sections.push(format!(
        "```walrus\n{}\n```",
        definition_signature(definition)
    ));

    if let Some(documentation) = &definition.documentation {
        sections.push(documentation.clone());
    }

    sections.push(format!("Kind: `{}`", symbol_kind_label(definition.kind)));
    sections.join("\n\n")
}

fn definition_signature(definition: &Definition) -> String {
    if let Some(detail) = &definition.detail {
        return detail.clone();
    }

    match definition.kind {
        WalrusSymbolKind::Function | WalrusSymbolKind::Method => {
            if definition.parameters.is_empty() {
                format!("fn {}", definition.name)
            } else {
                format!(
                    "fn {} : {}",
                    definition.name,
                    definition.parameters.join(", ")
                )
            }
        }
        WalrusSymbolKind::Struct => format!("struct {}", definition.name),
        WalrusSymbolKind::Variable => format!("let {}", definition.name),
        WalrusSymbolKind::Parameter => format!("parameter {}", definition.name),
        WalrusSymbolKind::Module => definition.name.clone(),
    }
}

fn documentation_from_text(text: &str) -> Documentation {
    Documentation::MarkupContent(MarkupContent {
        kind: MarkupKind::Markdown,
        value: text.to_string(),
    })
}

fn builtin_by_name(name: &str) -> Option<&'static BuiltinInfo> {
    BUILTINS.iter().find(|builtin| builtin.name == name)
}

fn keyword_docs(keyword: &str) -> Option<&'static str> {
    match keyword {
        "let" => Some("Declares a variable in the current scope."),
        "fn" => Some("Declares a function."),
        "struct" => Some("Declares a struct type."),
        "return" => Some("Returns a value from a function."),
        "if" => Some("Conditional branch."),
        "else" => Some("Fallback branch for an `if` block."),
        "while" => Some("Loops while a condition is true."),
        "for" => Some("Iterates over a range or iterable."),
        "in" => Some("Used by `for` loops to denote the source iterable."),
        "break" => Some("Exits the nearest loop."),
        "continue" => Some("Skips to the next loop iteration."),
        "import" => Some("Imports a module or package."),
        "as" => Some("Renames an imported module binding."),
        "try" => Some("Starts an exception handling block."),
        "catch" => Some("Handles a thrown value from `try`."),
        "throw" => Some("Raises an exception value."),
        "defer" => Some("Defers expression execution until scope exit."),
        "free" => Some("Manually frees heap-managed memory."),
        "extern" => Some("Declares an external function signature."),
        "and" => Some("Logical AND operator."),
        "or" => Some("Logical OR operator."),
        "not" => Some("Logical NOT operator."),
        "true" => Some("Boolean true literal."),
        "false" => Some("Boolean false literal."),
        "void" => Some("Void literal value."),
        "start" => Some("Starts a test-style block."),
        "end" => Some("Ends a test-style block."),
        "print" => Some("Prints a value without a newline."),
        "println" => Some("Prints a value with a newline."),
        _ => None,
    }
}

fn is_valid_identifier(identifier: &str) -> bool {
    let mut chars = identifier.chars();
    let Some(first) = chars.next() else {
        return false;
    };

    if !first.is_ascii_alphabetic() && first != '_' {
        return false;
    }

    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn completion_item_kind(kind: WalrusSymbolKind) -> CompletionItemKind {
    match kind {
        WalrusSymbolKind::Function => CompletionItemKind::FUNCTION,
        WalrusSymbolKind::Method => CompletionItemKind::METHOD,
        WalrusSymbolKind::Struct => CompletionItemKind::CLASS,
        WalrusSymbolKind::Variable => CompletionItemKind::VARIABLE,
        WalrusSymbolKind::Parameter => CompletionItemKind::VARIABLE,
        WalrusSymbolKind::Module => CompletionItemKind::MODULE,
    }
}

fn symbol_kind(kind: WalrusSymbolKind) -> LspSymbolKind {
    match kind {
        WalrusSymbolKind::Function => LspSymbolKind::FUNCTION,
        WalrusSymbolKind::Method => LspSymbolKind::METHOD,
        WalrusSymbolKind::Struct => LspSymbolKind::STRUCT,
        WalrusSymbolKind::Variable => LspSymbolKind::VARIABLE,
        WalrusSymbolKind::Parameter => LspSymbolKind::VARIABLE,
        WalrusSymbolKind::Module => LspSymbolKind::MODULE,
    }
}

fn symbol_kind_label(kind: WalrusSymbolKind) -> &'static str {
    match kind {
        WalrusSymbolKind::Function => "function",
        WalrusSymbolKind::Method => "method",
        WalrusSymbolKind::Struct => "struct",
        WalrusSymbolKind::Variable => "variable",
        WalrusSymbolKind::Parameter => "parameter",
        WalrusSymbolKind::Module => "module",
    }
}

fn diagnostic_to_lsp(diagnostic: &ParseDiagnostic, text: &str) -> Diagnostic {
    let severity = match diagnostic.severity {
        WalrusDiagnosticSeverity::Error => DiagnosticSeverity::ERROR,
    };

    Diagnostic {
        range: span_to_range(text, diagnostic.span),
        severity: Some(severity),
        code: None,
        code_description: None,
        source: Some("walrus-lsp".to_string()),
        message: diagnostic.message.clone(),
        related_information: None,
        tags: None,
        data: None,
    }
}

fn span_to_range(text: &str, span: Span) -> Range {
    Range {
        start: byte_to_position(text, span.0),
        end: byte_to_position(text, span.1),
    }
}

fn byte_to_position(text: &str, byte_offset: usize) -> Position {
    let offset = byte_offset.min(text.len());
    let starts = line_starts(text);
    let line_index = starts
        .partition_point(|line_start| *line_start <= offset)
        .saturating_sub(1);
    let line_start = starts[line_index];
    let utf16_col = text[line_start..offset].encode_utf16().count() as u32;
    Position::new(line_index as u32, utf16_col)
}

fn position_to_byte(text: &str, position: Position) -> Option<usize> {
    let starts = line_starts(text);
    let line = usize::try_from(position.line).ok()?;

    if line >= starts.len() {
        if line == starts.len() && text.ends_with('\n') {
            return Some(text.len());
        }
        return None;
    }

    let line_start = starts[line];
    let line_end = if line + 1 < starts.len() {
        starts[line + 1]
    } else {
        text.len()
    };

    let line_text = &text[line_start..line_end];
    let mut utf16_offset = 0u32;
    for (byte_offset, ch) in line_text.char_indices() {
        if utf16_offset >= position.character {
            return Some(line_start + byte_offset);
        }
        let next = utf16_offset + ch.len_utf16() as u32;
        if next > position.character {
            return Some(line_start + byte_offset);
        }
        utf16_offset = next;
    }

    Some(line_end)
}

fn line_starts(text: &str) -> Vec<usize> {
    let mut starts = vec![0usize];
    for (index, ch) in text.char_indices() {
        if ch == '\n' {
            starts.push(index + 1);
        }
    }
    starts
}

fn identifier_at_offset(text: &str, offset: usize) -> Option<(String, Span)> {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return None;
    }

    let mut index = offset.min(bytes.len().saturating_sub(1));
    if !is_ident_char(bytes[index]) {
        if index == 0 || !is_ident_char(bytes[index - 1]) {
            return None;
        }
        index -= 1;
    }

    let mut start = index;
    while start > 0 && is_ident_char(bytes[start - 1]) {
        start -= 1;
    }

    let mut end = index + 1;
    while end < bytes.len() && is_ident_char(bytes[end]) {
        end += 1;
    }

    if !is_ident_start(bytes[start]) {
        return None;
    }

    Some((
        String::from_utf8_lossy(&bytes[start..end]).to_string(),
        Span(start, end),
    ))
}

fn is_ident_char(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn is_ident_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

#[derive(Debug)]
struct CallContext {
    callee_name: String,
    open_paren_offset: usize,
    arg_index: usize,
}

fn call_context_at_offset(text: &str, offset: usize) -> Option<CallContext> {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return None;
    }

    let mut index = offset.min(bytes.len());
    let mut depth = 0usize;
    let mut arg_index = 0usize;

    while index > 0 {
        index -= 1;
        match bytes[index] {
            b')' => depth += 1,
            b'(' => {
                if depth == 0 {
                    let mut cursor = index;
                    while cursor > 0 && bytes[cursor - 1].is_ascii_whitespace() {
                        cursor -= 1;
                    }

                    let end = cursor;
                    while cursor > 0 && is_ident_char(bytes[cursor - 1]) {
                        cursor -= 1;
                    }

                    if cursor == end || !is_ident_start(bytes[cursor]) {
                        return None;
                    }

                    let callee_name = String::from_utf8_lossy(&bytes[cursor..end]).to_string();
                    return Some(CallContext {
                        callee_name,
                        open_paren_offset: index,
                        arg_index,
                    });
                }
                depth = depth.saturating_sub(1);
            }
            b',' if depth == 0 => arg_index += 1,
            _ => {}
        }
    }

    None
}

fn build_document_symbols(analysis: &Analysis, text: &str) -> Vec<DocumentSymbol> {
    let included = analysis
        .definitions
        .iter()
        .filter(|definition| definition.kind != WalrusSymbolKind::Parameter)
        .collect::<Vec<_>>();
    let included_ids = included
        .iter()
        .map(|definition| definition.id)
        .collect::<HashSet<_>>();

    let mut by_parent: HashMap<Option<usize>, Vec<&Definition>> = HashMap::new();
    for definition in included {
        let parent = definition
            .container
            .filter(|container| included_ids.contains(container));
        by_parent.entry(parent).or_default().push(definition);
    }

    fn build_for_parent(
        parent: Option<usize>,
        by_parent: &HashMap<Option<usize>, Vec<&Definition>>,
        text: &str,
    ) -> Vec<DocumentSymbol> {
        let Some(definitions) = by_parent.get(&parent) else {
            return Vec::new();
        };

        let mut sorted = definitions.clone();
        sorted.sort_by_key(|definition| definition.span.0);

        sorted
            .into_iter()
            .map(|definition| {
                let children = build_for_parent(Some(definition.id), by_parent, text);
                #[allow(deprecated)]
                let symbol = DocumentSymbol {
                    name: definition.name.clone(),
                    detail: definition.detail.clone(),
                    kind: symbol_kind(definition.kind),
                    tags: None,
                    deprecated: None,
                    range: span_to_range(text, definition.full_span),
                    selection_range: span_to_range(text, definition.span),
                    children: if children.is_empty() {
                        None
                    } else {
                        Some(children)
                    },
                };
                symbol
            })
            .collect()
    }

    build_for_parent(None, &by_parent, text)
}

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let (service, socket) = LspService::new(Backend::new);
    Server::new(stdin, stdout, socket).serve(service).await;
}
