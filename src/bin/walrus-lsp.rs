use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::{
    CompletionItem, CompletionItemKind, CompletionOptions, CompletionResponse, Diagnostic,
    DiagnosticSeverity, DidChangeTextDocumentParams, DidCloseTextDocumentParams,
    DidOpenTextDocumentParams, DidSaveTextDocumentParams, DocumentSymbolParams,
    DocumentSymbolResponse, GotoDefinitionParams, GotoDefinitionResponse, Hover, HoverContents,
    HoverParams, HoverProviderCapability, InitializeParams, InitializeResult, Location,
    MarkedString, MessageType, OneOf, Position, Range, ServerCapabilities, SymbolInformation,
    SymbolKind as LspSymbolKind, TextDocumentSyncCapability, TextDocumentSyncKind, Url,
};
use tower_lsp::{Client, LanguageServer, LspService, Server};
use walrus::lsp_support::{
    Analysis, Definition, DiagnosticSeverity as WalrusDiagnosticSeverity, ParseDiagnostic,
    SymbolKind as WalrusSymbolKind, analyze,
};
use walrus::span::Span;

const KEYWORDS: &[&str] = &[
    "let", "fn", "struct", "return", "if", "else", "while", "for", "in", "break", "continue",
    "import", "as", "print", "println", "true", "false", "void", "try", "catch", "throw", "free",
    "defer", "extern", "and", "or", "not", "start", "end",
];

const BUILTINS: &[&str] = &[
    "len",
    "str",
    "type",
    "input",
    "__gc__",
    "__heap_stats__",
    "__gc_threshold__",
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
            .map(|d| diagnostic_to_lsp(d, &text))
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
                document_symbol_provider: Some(OneOf::Left(true)),
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

        let Some((identifier, span)) = identifier_at_position(&doc.text, position) else {
            return Ok(None);
        };

        if let Some(message) = docs_for_identifier(&identifier) {
            return Ok(Some(Hover {
                contents: HoverContents::Scalar(MarkedString::String(message.to_string())),
                range: Some(span_to_range(&doc.text, span)),
            }));
        }

        if let Some(definition) = doc
            .analysis
            .definitions
            .iter()
            .find(|def| def.name == identifier && span_contains(def.span, span.0))
        {
            let detail = format!(
                "{} `{}`",
                symbol_kind_label(definition.kind),
                definition.name
            );
            return Ok(Some(Hover {
                contents: HoverContents::Scalar(MarkedString::String(detail)),
                range: Some(span_to_range(&doc.text, span)),
            }));
        }

        Ok(None)
    }

    async fn completion(
        &self,
        _: tower_lsp::lsp_types::CompletionParams,
    ) -> Result<Option<CompletionResponse>> {
        let mut items = Vec::with_capacity(KEYWORDS.len() + BUILTINS.len());

        for keyword in KEYWORDS {
            items.push(CompletionItem {
                label: (*keyword).to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                detail: Some("keyword".to_string()),
                ..CompletionItem::default()
            });
        }

        for builtin in BUILTINS {
            items.push(CompletionItem {
                label: (*builtin).to_string(),
                kind: Some(CompletionItemKind::FUNCTION),
                detail: Some("builtin".to_string()),
                ..CompletionItem::default()
            });
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

        let Some((identifier, usage_span)) = identifier_at_position(&doc.text, position) else {
            return Ok(None);
        };

        let usage_scope = doc
            .analysis
            .references
            .iter()
            .find(|reference| {
                reference.name == identifier && span_contains(reference.span, usage_span.0)
            })
            .map(|reference| reference.scope_depth)
            .or_else(|| {
                doc.analysis
                    .definitions
                    .iter()
                    .find(|definition| {
                        definition.name == identifier
                            && span_contains(definition.span, usage_span.0)
                    })
                    .map(|definition| definition.scope_depth)
            });

        let Some(definition) = choose_definition(
            &doc.analysis.definitions,
            &identifier,
            usage_span.0,
            usage_scope,
        ) else {
            return Ok(None);
        };

        Ok(Some(GotoDefinitionResponse::Scalar(Location::new(
            uri.clone(),
            span_to_range(&doc.text, definition.span),
        ))))
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

        let symbols = doc
            .analysis
            .symbols
            .iter()
            .map(|symbol| {
                #[allow(deprecated)]
                let info = SymbolInformation {
                    name: symbol.name.clone(),
                    kind: lsp_symbol_kind(symbol.kind),
                    location: Location::new(uri.clone(), span_to_range(&doc.text, symbol.span)),
                    deprecated: None,
                    tags: None,
                    container_name: None,
                };
                info
            })
            .collect::<Vec<_>>();

        Ok(Some(DocumentSymbolResponse::Flat(symbols)))
    }
}

fn choose_definition<'a>(
    definitions: &'a [Definition],
    name: &str,
    usage_start: usize,
    usage_scope: Option<usize>,
) -> Option<&'a Definition> {
    let filtered_by_name = definitions
        .iter()
        .filter(|definition| definition.name == name);
    let mut candidates = filtered_by_name
        .filter(|definition| usage_scope.is_none_or(|scope| definition.scope_depth <= scope))
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        candidates = definitions
            .iter()
            .filter(|definition| definition.name == name)
            .collect::<Vec<_>>();
    }

    candidates.into_iter().max_by_key(|definition| {
        (
            if definition.span.0 <= usage_start {
                1
            } else {
                0
            },
            definition.scope_depth,
            definition.span.0,
        )
    })
}

fn docs_for_identifier(identifier: &str) -> Option<&'static str> {
    match identifier {
        "let" => Some("Declare a variable in the current scope."),
        "fn" => Some("Define a function."),
        "struct" => Some("Define a struct type with methods."),
        "return" => Some("Return a value from the current function."),
        "if" => Some("Conditional branch."),
        "else" => Some("Fallback branch for an `if` expression."),
        "while" => Some("Loop while a condition is true."),
        "for" => Some("Iterate over a range or iterable."),
        "import" => Some("Import a standard module or package."),
        "try" => Some("Start exception handling block."),
        "catch" => Some("Handle an exception from a `try` block."),
        "throw" => Some("Raise an exception value."),
        "len" => Some("Built-in: returns length of string, list, or dictionary."),
        "str" => Some("Built-in: converts a value to string."),
        "type" => Some("Built-in: returns type name for a value."),
        "input" => Some("Built-in: reads user input from stdin."),
        "__gc__" => Some("Built-in: manually trigger garbage collection."),
        "__heap_stats__" => Some("Built-in: return heap statistics as a dictionary."),
        "__gc_threshold__" => Some("Built-in: configure GC allocation threshold."),
        _ => None,
    }
}

fn symbol_kind_label(kind: WalrusSymbolKind) -> &'static str {
    match kind {
        WalrusSymbolKind::Function => "function",
        WalrusSymbolKind::Method => "method",
        WalrusSymbolKind::Struct => "struct",
        WalrusSymbolKind::Variable => "variable",
        WalrusSymbolKind::Module => "module",
    }
}

fn lsp_symbol_kind(kind: WalrusSymbolKind) -> LspSymbolKind {
    match kind {
        WalrusSymbolKind::Function => LspSymbolKind::FUNCTION,
        WalrusSymbolKind::Method => LspSymbolKind::METHOD,
        WalrusSymbolKind::Struct => LspSymbolKind::STRUCT,
        WalrusSymbolKind::Variable => LspSymbolKind::VARIABLE,
        WalrusSymbolKind::Module => LspSymbolKind::MODULE,
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

fn span_contains(span: Span, offset: usize) -> bool {
    span.0 <= offset && offset <= span.1
}

fn byte_to_position(text: &str, byte_offset: usize) -> Position {
    let offset = byte_offset.min(text.len());
    let line_starts = line_starts(text);
    let line_index = line_starts
        .partition_point(|start| *start <= offset)
        .saturating_sub(1);
    let line_start = line_starts[line_index];
    let slice = &text[line_start..offset];
    let utf16_col = slice.encode_utf16().count() as u32;
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

    let mut utf16_count = 0u32;
    for (byte, ch) in line_text.char_indices() {
        if utf16_count >= position.character {
            return Some(line_start + byte);
        }
        let next = utf16_count + ch.len_utf16() as u32;
        if next > position.character {
            return Some(line_start + byte);
        }
        utf16_count = next;
    }

    Some(line_end)
}

fn line_starts(text: &str) -> Vec<usize> {
    let mut starts = vec![0usize];
    for (idx, ch) in text.char_indices() {
        if ch == '\n' {
            starts.push(idx + 1);
        }
    }
    starts
}

fn identifier_at_position(text: &str, position: Position) -> Option<(String, Span)> {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return None;
    }

    let mut byte = position_to_byte(text, position)?;
    if byte >= bytes.len() {
        byte = bytes.len().saturating_sub(1);
    }

    if !is_ident_char(bytes[byte]) {
        if byte == 0 || !is_ident_char(bytes[byte - 1]) {
            return None;
        }
        byte -= 1;
    }

    let mut start = byte;
    while start > 0 && is_ident_char(bytes[start - 1]) {
        start -= 1;
    }

    let mut end = byte + 1;
    while end < bytes.len() && is_ident_char(bytes[end]) {
        end += 1;
    }

    if !is_ident_start(bytes[start]) {
        return None;
    }

    let name = String::from_utf8_lossy(&bytes[start..end]).to_string();
    Some((name, Span(start, end)))
}

fn is_ident_char(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn is_ident_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let (service, socket) = LspService::new(Backend::new);
    Server::new(stdin, stdout, socket).serve(service).await;
}
