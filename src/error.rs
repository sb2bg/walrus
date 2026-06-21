use std::io::{self, Write};
use std::ops::Range;
use std::str::FromStr;
use std::sync::Arc;

use ariadne::{Config, IndexType, Label, Report, ReportKind, sources};
use float_ord::FloatOrd;
use git_version::git_version;
use lalrpop_util::ParseError;
use lalrpop_util::lexer::Token;
use snailquote::{UnescapeError, unescape};
use thiserror::Error;

use crate::ast::NodeKind;
use crate::span::{Span, Spanned};
use crate::vm::opcode::Opcode;

const FSTRING_INTERP_QUOTE_SENTINEL: char = '\u{001F}';

#[derive(Debug)]
pub enum RecoveredParseError {
    NumberTooLarge(String, Span),
    InvalidEscapeSequence(String, Span),
    InvalidUnicodeEscapeSequence(Span),
    InvalidFStringExpression(String, Span),
}

fn is_ident_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

/// Rewrites raw `"` inside f-string interpolations to a sentinel byte so the
/// lexer can still tokenize the full `f"..."` literal.
pub fn preprocess_fstrings_for_lexer(source: &str) -> String {
    fn has_matching_quote_before_interp_end(
        chars: std::iter::Peekable<std::str::Chars<'_>>,
        mut interp_depth: usize,
    ) -> bool {
        for ch in chars {
            if ch == '"' {
                return true;
            }

            if ch == '{' {
                interp_depth += 1;
                continue;
            }

            if ch == '}' {
                interp_depth = interp_depth.saturating_sub(1);
                if interp_depth == 0 {
                    return false;
                }
            }
        }

        false
    }

    #[derive(Clone, Copy)]
    enum ScannerState {
        Normal,
        String {
            escape: bool,
        },
        LineComment,
        BlockComment {
            prev_star: bool,
        },
        FString {
            interp_depth: usize,
            literal_escape: bool,
            in_interp_string: bool,
        },
    }

    let mut out = String::with_capacity(source.len());
    let mut chars = source.chars().peekable();
    let mut state = ScannerState::Normal;
    let mut prev_char: Option<char> = None;

    while let Some(ch) = chars.next() {
        match state {
            ScannerState::Normal => {
                if ch == '/' {
                    if let Some('/') = chars.peek().copied() {
                        out.push(ch);
                        let next = chars.next().unwrap_or('/');
                        out.push(next);
                        prev_char = Some(next);
                        state = ScannerState::LineComment;
                        continue;
                    }
                    if let Some('*') = chars.peek().copied() {
                        out.push(ch);
                        let next = chars.next().unwrap_or('*');
                        out.push(next);
                        prev_char = Some(next);
                        state = ScannerState::BlockComment { prev_star: false };
                        continue;
                    }
                }

                if ch == '"' {
                    out.push(ch);
                    prev_char = Some(ch);
                    state = ScannerState::String { escape: false };
                    continue;
                }

                let starts_fstring = ch == 'f'
                    && matches!(chars.peek(), Some('"'))
                    && prev_char.is_none_or(|prev| !is_ident_char(prev));

                if starts_fstring {
                    out.push(ch);
                    let quote = chars.next().unwrap_or('"');
                    out.push(quote);
                    prev_char = Some(quote);
                    state = ScannerState::FString {
                        interp_depth: 0,
                        literal_escape: false,
                        in_interp_string: false,
                    };
                    continue;
                }

                out.push(ch);
                prev_char = Some(ch);
            }
            ScannerState::String { mut escape } => {
                out.push(ch);

                if escape {
                    escape = false;
                } else if ch == '\\' {
                    escape = true;
                } else if ch == '"' {
                    state = ScannerState::Normal;
                    prev_char = Some(ch);
                    continue;
                }

                prev_char = Some(ch);
                state = ScannerState::String { escape };
            }
            ScannerState::LineComment => {
                out.push(ch);
                prev_char = Some(ch);

                if ch == '\n' || ch == '\r' {
                    state = ScannerState::Normal;
                }
            }
            ScannerState::BlockComment { mut prev_star } => {
                out.push(ch);
                prev_char = Some(ch);

                if prev_star && ch == '/' {
                    state = ScannerState::Normal;
                    continue;
                }
                prev_star = ch == '*';
                state = ScannerState::BlockComment { prev_star };
            }
            ScannerState::FString {
                mut interp_depth,
                mut literal_escape,
                mut in_interp_string,
            } => {
                if interp_depth == 0 {
                    out.push(ch);
                    prev_char = Some(ch);

                    if literal_escape {
                        literal_escape = false;
                    } else if ch == '\\' {
                        literal_escape = true;
                    } else if ch == '{' {
                        interp_depth = 1;
                        in_interp_string = false;
                    } else if ch == '"' {
                        state = ScannerState::Normal;
                        continue;
                    }
                } else {
                    if in_interp_string {
                        let emitted = if ch == '"' {
                            FSTRING_INTERP_QUOTE_SENTINEL
                        } else {
                            ch
                        };
                        out.push(emitted);
                        prev_char = Some(ch);

                        if ch == '"' {
                            in_interp_string = false;
                        }
                    } else {
                        if ch == '"' {
                            if has_matching_quote_before_interp_end(chars.clone(), interp_depth) {
                                out.push(FSTRING_INTERP_QUOTE_SENTINEL);
                                prev_char = Some(ch);
                                in_interp_string = true;
                            } else {
                                out.push(ch);
                                prev_char = Some(ch);
                                state = ScannerState::Normal;
                                continue;
                            }
                        } else {
                            out.push(ch);
                            prev_char = Some(ch);

                            if ch == '{' {
                                interp_depth += 1;
                            } else if ch == '}' {
                                interp_depth -= 1;
                            }
                        }
                    }
                }

                state = ScannerState::FString {
                    interp_depth,
                    literal_escape,
                    in_interp_string,
                };
            }
        }
    }

    out
}

pub fn parse_int<T>(
    spanned: Spanned<String>,
    base: u32,
) -> Result<NodeKind, ParseError<usize, T, RecoveredParseError>> {
    let num = i64::from_str_radix(
        if base == 10 {
            spanned.value()
        } else {
            &spanned.value()[2..]
        },
        base,
    )
    .map_err(|_| {
        let span = spanned.span();

        ParseError::User {
            error: RecoveredParseError::NumberTooLarge(spanned.into_value(), span),
        }
    })?;

    Ok(NodeKind::Int(num))
}

pub fn parse_float<T>(
    spanned: Spanned<String>,
) -> Result<NodeKind, ParseError<usize, T, RecoveredParseError>> {
    let num = f64::from_str(spanned.value()).map_err(|_| {
        let span = spanned.span();

        ParseError::User {
            error: RecoveredParseError::NumberTooLarge(spanned.into_value(), span),
        }
    })?;

    Ok(NodeKind::Float(FloatOrd(num)))
}

pub fn escape_string<T>(
    spanned: Spanned<String>,
) -> Result<NodeKind, ParseError<usize, T, RecoveredParseError>> {
    match unescape(spanned.value()) {
        Ok(s) => Ok(NodeKind::String(s)),
        Err(err) => {
            let span = spanned.span();

            match err {
                UnescapeError::InvalidEscape { escape, .. } => Err(ParseError::User {
                    error: RecoveredParseError::InvalidEscapeSequence(escape, span),
                }),
                UnescapeError::InvalidUnicode { .. } => Err(ParseError::User {
                    error: RecoveredParseError::InvalidUnicodeEscapeSequence(span),
                }),
            }
        }
    }
}

pub fn parse_fstring<T>(
    spanned: Spanned<String>,
) -> Result<NodeKind, ParseError<usize, T, RecoveredParseError>> {
    use crate::ast::FStringPart;
    use lalrpop_util::lalrpop_mod;

    lalrpop_mod!(#[allow(clippy::all, unused_imports)] pub grammar);
    let parser = grammar::ExpressionParser::new();

    let raw = spanned.value();
    let normalized_raw;
    let raw = if raw.contains(FSTRING_INTERP_QUOTE_SENTINEL) {
        normalized_raw = raw.replace(FSTRING_INTERP_QUOTE_SENTINEL, "\"");
        normalized_raw.as_str()
    } else {
        raw
    };
    let base_offset = spanned.span().0; // Start position of the f-string

    // Remove f" and trailing " -> f" is 2 chars, so content starts at base_offset + 2
    let content = &raw[2..raw.len() - 1];
    let content_offset = base_offset + 2;

    let mut parts = Vec::new();
    let mut current_literal = String::new();
    let mut char_indices = content.char_indices().peekable();

    fn decode_fstring_escape<T>(
        ch: char,
        span: Span,
    ) -> Result<char, ParseError<usize, T, RecoveredParseError>> {
        match ch {
            'n' => Ok('\n'),
            't' => Ok('\t'),
            'r' => Ok('\r'),
            '\\' => Ok('\\'),
            '"' => Ok('"'),
            '{' => Ok('{'),
            '}' => Ok('}'),
            'u' => Err(ParseError::User {
                error: RecoveredParseError::InvalidUnicodeEscapeSequence(span),
            }),
            _ => Err(ParseError::User {
                error: RecoveredParseError::InvalidEscapeSequence(format!("\\{ch}"), span),
            }),
        }
    }

    while let Some((byte_pos, ch)) = char_indices.next() {
        if ch == '\\' {
            if let Some((_, next_ch)) = char_indices.next() {
                let escape_span = Span(
                    content_offset + byte_pos,
                    content_offset + byte_pos + 1 + next_ch.len_utf8(),
                );
                let decoded = decode_fstring_escape(next_ch, escape_span)?;
                current_literal.push(decoded);
            } else {
                let escape_span = Span(content_offset + byte_pos, content_offset + byte_pos + 1);
                return Err(ParseError::User {
                    error: RecoveredParseError::InvalidEscapeSequence(
                        "\\".to_string(),
                        escape_span,
                    ),
                });
            }
        } else if ch == '{' {
            // Start of an interpolation
            if !current_literal.is_empty() {
                parts.push(FStringPart::Literal(current_literal.clone()));
                current_literal.clear();
            }

            // Find the matching closing brace
            let expr_start = byte_pos + 1; // Position after {
            let mut expr_str = String::new();
            let mut brace_depth = 1;
            let mut interpolation_closed = false;
            let mut in_string_literal = false;
            let mut string_escape = false;

            while let Some((expr_pos, expr_ch)) = char_indices.next() {
                let emitted = if expr_ch == '\\' {
                    if let Some((_, next_ch)) = char_indices.next() {
                        let escape_span = Span(
                            content_offset + expr_pos,
                            content_offset + expr_pos + 1 + next_ch.len_utf8(),
                        );
                        decode_fstring_escape(next_ch, escape_span)?
                    } else {
                        let escape_span =
                            Span(content_offset + expr_pos, content_offset + expr_pos + 1);
                        return Err(ParseError::User {
                            error: RecoveredParseError::InvalidEscapeSequence(
                                "\\".to_string(),
                                escape_span,
                            ),
                        });
                    }
                } else {
                    expr_ch
                };

                if in_string_literal {
                    if string_escape {
                        string_escape = false;
                    } else if emitted == '\\' {
                        string_escape = true;
                    } else if emitted == '"' {
                        in_string_literal = false;
                    }

                    expr_str.push(emitted);
                    continue;
                }

                if emitted == '"' {
                    in_string_literal = true;
                    expr_str.push(emitted);
                    continue;
                }

                if emitted == '{' {
                    brace_depth += 1;
                    expr_str.push(emitted);
                    continue;
                }

                if emitted == '}' {
                    brace_depth -= 1;
                    if brace_depth == 0 {
                        interpolation_closed = true;
                        break;
                    }
                    expr_str.push(emitted);
                    continue;
                }

                expr_str.push(emitted);
            }

            if !interpolation_closed {
                let expr_span = Span(content_offset + expr_start, content_offset + content.len());
                return Err(ParseError::User {
                    error: RecoveredParseError::InvalidFStringExpression(expr_str, expr_span),
                });
            }

            // Parse the expression with proper span
            if expr_str.trim().is_empty() {
                let expr_span = Span(content_offset + expr_start, content_offset + expr_start);
                return Err(ParseError::User {
                    error: RecoveredParseError::InvalidFStringExpression(expr_str, expr_span),
                });
            }

            match parser.parse(&expr_str) {
                Ok(node) => {
                    // Adjust the span to reflect the actual position in the source file
                    let expr_span = node.span();
                    let adjusted_span = Span(
                        content_offset + expr_start + expr_span.0,
                        content_offset + expr_start + expr_span.1,
                    );
                    // Create a new node with the adjusted span
                    let adjusted_node = crate::ast::Node::new(node.into_kind(), adjusted_span);
                    parts.push(FStringPart::Expr(Box::new(adjusted_node)));
                }
                Err(_) => {
                    // If parsing fails, return an error
                    let expr_span = Span(
                        content_offset + expr_start,
                        content_offset + expr_start + expr_str.len(),
                    );
                    return Err(ParseError::User {
                        error: RecoveredParseError::InvalidFStringExpression(expr_str, expr_span),
                    });
                }
            }
        } else {
            current_literal.push(ch);
        }
    }

    if !current_literal.is_empty() {
        parts.push(FStringPart::Literal(current_literal));
    }

    Ok(NodeKind::FString(parts))
}

#[derive(Debug, Clone)]
pub struct ErrorContext {
    span: Span,
    src: Arc<str>,
    filename: Arc<str>,
}

impl ErrorContext {
    pub fn new(span: Span, src: impl Into<Arc<str>>, filename: impl Into<Arc<str>>) -> Self {
        Self {
            span,
            src: src.into(),
            filename: filename.into(),
        }
    }

    pub fn from_shared(span: Span, src: Arc<str>, filename: Arc<str>) -> Self {
        Self {
            span,
            src,
            filename,
        }
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn source(&self) -> &str {
        &self.src
    }

    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn with_span(&self, span: Span) -> Self {
        Self {
            span,
            src: Arc::clone(&self.src),
            filename: Arc::clone(&self.filename),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WalrusDiagnostic {
    message: String,
    source: Option<ErrorContext>,
    labels: Vec<DiagnosticLabel>,
    notes: Vec<String>,
}

#[derive(Debug, Clone)]
struct DiagnosticLabel {
    span: Span,
    message: Option<String>,
}

impl WalrusDiagnostic {
    fn plain(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source: None,
            labels: Vec::new(),
            notes: Vec::new(),
        }
    }

    fn source_error(message: impl Into<String>, context: ErrorContext) -> Self {
        let span = context.span();
        Self {
            message: message.into(),
            source: Some(context),
            labels: vec![DiagnosticLabel {
                span,
                message: Some("here".to_string()),
            }],
            notes: Vec::new(),
        }
    }

    pub fn write<W: Write>(&self, mut writer: W) -> io::Result<()> {
        let Some(source) = &self.source else {
            writeln!(writer, "Error: {}", self.message)?;
            for note in &self.notes {
                writeln!(writer, "Note: {note}")?;
            }
            return Ok(());
        };

        let primary_span = self
            .labels
            .first()
            .map(|label| label.span)
            .unwrap_or_default();
        let primary_range = diagnostic_range(source.source(), primary_span);
        let filename = source.filename().to_string();
        let mut builder =
            Report::build(ReportKind::Error, (filename.clone(), primary_range.clone()))
                .with_config(
                    Config::default()
                        .with_color(false)
                        .with_index_type(IndexType::Byte),
                )
                .with_message(&self.message);

        for label in &self.labels {
            let mut ariadne_label = Label::new((
                filename.clone(),
                diagnostic_range(source.source(), label.span),
            ));
            if let Some(message) = &label.message {
                ariadne_label = ariadne_label.with_message(message);
            }
            builder = builder.with_label(ariadne_label);
        }

        for note in &self.notes {
            builder = builder.with_note(note);
        }

        builder.finish().write(
            sources(vec![(filename, source.source().to_string())]),
            &mut writer,
        )
    }
}

fn diagnostic_range(src: &str, span: Span) -> Range<usize> {
    let len = src.len();
    let start = span.0.min(len);
    let end = span.1.min(len).max(start);
    start..end
}

#[derive(Error, Debug)]
pub enum WalrusError {
    #[error("Unknown error '{message}'. Please report this bug with the following information: Walrus Version = '{}', Git Revision = '{}'", env!("CARGO_PKG_VERSION"), git_version!(fallback = "flamegraph"))]
    UnknownError { message: String },

    #[error("{message}")]
    GenericError { message: String },

    // temporary error for development
    #[error("{message}")]
    TodoError { message: String },

    #[error("IO error occurred while reading/writing")]
    IOError {
        #[from]
        source: std::io::Error,
    },

    #[error(
        "Unable to locate file '{filename}'. Make sure the file exists and that you have permission to read it."
    )]
    FileNotFound { filename: String },

    #[error("Unexpected EOF in source")]
    UnexpectedEndOfInput { context: ErrorContext },

    #[error("Unexpected token '{token}'")]
    UnexpectedToken {
        token: String,
        context: ErrorContext,
    },

    #[error("Invalid token '{token}'")]
    InvalidToken {
        token: String,
        context: ErrorContext,
    },

    #[error("Extra token '{token}'")]
    ExtraToken {
        token: String,
        context: ErrorContext,
    },

    #[error("Number '{number}' is too large")]
    NumberTooLarge {
        number: String,
        context: ErrorContext,
    },

    #[error("Circular import detected: module '{module}' is already being imported")]
    CircularImport { module: String },

    #[error("Invalid argument combination: '{first_arg}' cannot be used with '{second_arg}'")]
    InvalidArgumentCombination {
        first_arg: String,
        second_arg: String,
    },

    #[error("Invalid operation '{op}' on operands '{left}' and '{right}'")]
    InvalidOperation {
        op: Opcode,
        left: String,
        right: String,
        context: ErrorContext,
    },

    #[error("Invalid unary operation '{op}' on operand {operand}")]
    InvalidUnaryOperation {
        op: Opcode,
        operand: String,
        context: ErrorContext,
    },

    #[error("Undefined variable '{name}'")]
    UndefinedVariable { name: String, context: ErrorContext },

    #[error("Return statement outside of function")]
    ReturnOutsideFunction { context: ErrorContext },

    #[error("Break statement outside of loop")]
    BreakOutsideLoop { context: ErrorContext },

    #[error("Continue statement outside of loop")]
    ContinueOutsideLoop { context: ErrorContext },

    #[error("Expected type '{expected}', but found type '{found}'")]
    TypeMismatch {
        expected: String,
        found: String,
        context: ErrorContext,
    },

    #[error("Exception '{message}' thrown")]
    Exception {
        message: String,
        context: ErrorContext,
    },

    #[error("{message}")]
    RuntimeError {
        message: String,
        context: ErrorContext,
    },

    #[error("Invalid escape sequence '{sequence}'")]
    InvalidEscapeSequence {
        sequence: String,
        context: ErrorContext,
    },

    #[error("Invalid unicode escape sequence")]
    InvalidUnicodeEscapeSequence { context: ErrorContext },

    #[error("Failed to parse f-string expression '{expr}': {error}")]
    FStringParseError {
        expr: String,
        error: String,
        context: ErrorContext,
    },

    #[error("Cannot free memory not in the heap")]
    FailedFree { context: ErrorContext },

    #[error("Cannot call non-function '{value}'")]
    NotCallable {
        value: String,
        context: ErrorContext,
    },

    #[error("Function '{name}' expected {expected} arg(s) but got {got} arg(s)")]
    InvalidArgCount {
        name: String,
        expected: usize,
        got: usize,
        context: ErrorContext,
    },

    #[error("Index {index} out of bounds for object with len {len}")]
    IndexOutOfBounds {
        index: i64,
        len: usize,
        context: ErrorContext,
    },

    #[error("Cannot index non-indexable type '{value}'")]
    NotIndexable {
        value: String,
        context: ErrorContext,
    },

    #[error("Cannot index type '{non_indexable}' with type '{index_type}'")]
    InvalidIndexType {
        non_indexable: String,
        index_type: String,
        context: ErrorContext,
    },

    #[error(
        "Attempted to access released memory. This is either a bug in the interpreter or you freed memory allocated by the interpreter."
    )]
    AccessReleasedMemory { context: ErrorContext },

    #[error("Failed to gather PWD. This may be due to a permissions error.")]
    FailedGatherPWD,

    #[error("Type '{type_name}' is not iterable")]
    NotIterable {
        type_name: String,
        context: ErrorContext,
    },

    #[error("Division by zero")]
    DivisionByZero { context: ErrorContext },

    #[error("Cannot get length of non-indexable type '{type_name}'")]
    NoLength {
        type_name: String,
        context: ErrorContext,
    },

    #[error("Stack underflow while executing opcode '{op:?}'")]
    StackUnderflow { op: Opcode, context: ErrorContext },

    #[error("Invalid instruction '{op:?}'")]
    InvalidInstruction { op: Opcode, context: ErrorContext },

    #[error("Cannot redefine local variable with name '{name}'")]
    RedefinedLocal { name: String, context: ErrorContext },

    #[error("Key '{key}' not found")]
    KeyNotFound { key: String, context: ErrorContext },

    #[error("Range start '{start}' must be less than or equal to range end '{end}'")]
    InvalidRange {
        start: i64,
        end: i64,
        context: ErrorContext,
    },

    #[error("Type '{type_name}' has no method '{method}'")]
    MethodNotFound {
        type_name: String,
        method: String,
        context: ErrorContext,
    },

    #[error("Type '{type_name}' has no member '{member}'")]
    MemberNotFound {
        type_name: String,
        member: String,
        context: ErrorContext,
    },

    #[error(
        "Member access requires object type 'struct', 'struct instance', or 'dict' and member name type 'string', found object '{object_type}' and member '{member_type}'"
    )]
    InvalidMemberAccessTarget {
        object_type: String,
        member_type: String,
        context: ErrorContext,
    },

    #[error("Struct methods must compile to VM bytecode functions")]
    StructMethodMustBeVmFunction { context: ErrorContext },

    #[error("Cannot call method '{method}' on value of type '{type_name}'")]
    InvalidMethodReceiver {
        method: String,
        type_name: String,
        context: ErrorContext,
    },

    #[error("Module '{module}' not found")]
    ModuleNotFound {
        module: String,
        context: ErrorContext,
    },

    #[error("Thrown value: {message}")]
    ThrownValue {
        message: String,
        context: ErrorContext,
    },

    #[error("Invalid file mode '{mode}'. Supported modes: r, w, a, rw")]
    InvalidFileMode { mode: String },

    #[error("Failed to open file '{path}': {reason}")]
    FileOpenFailed { path: String, reason: String },

    #[error("Invalid file handle: {handle}")]
    InvalidFileHandle { handle: i64 },

    #[error("Failed to read from handle {handle}: {reason}")]
    FileReadFailed { handle: i64, reason: String },

    #[error("Failed to read line from handle {handle}: {reason}")]
    FileReadLineFailed { handle: i64, reason: String },

    #[error("Failed to write to handle {handle}: {reason}")]
    FileWriteFailed { handle: i64, reason: String },

    #[error("Failed to read '{path}': {reason}")]
    ReadFileFailed { path: String, reason: String },

    #[error("Failed to write '{path}': {reason}")]
    WriteFileFailed { path: String, reason: String },

    #[error("Cannot pop from an empty list")]
    EmptyListPop { context: ErrorContext },

    #[error("core.gc_threshold requires a positive integer argument")]
    InvalidGcThresholdArg { context: ErrorContext },

    #[error("Package imports (@package) are not yet implemented")]
    PackageImportNotImplemented { context: ErrorContext },

    #[error("{error}")]
    RuntimeErrorWithStackTrace {
        error: Box<WalrusError>,
        stack_trace: String,
    },
}

impl WalrusError {
    pub fn diagnostic(&self) -> Option<WalrusDiagnostic> {
        match self {
            WalrusError::RuntimeErrorWithStackTrace { error, stack_trace } => {
                let mut diagnostic = error
                    .diagnostic()
                    .unwrap_or_else(|| WalrusDiagnostic::plain(error.to_string()));
                let trimmed = stack_trace.trim();
                if !trimmed.is_empty() {
                    diagnostic.notes.push(trimmed.to_string());
                }
                Some(diagnostic)
            }
            _ => self
                .source_context()
                .cloned()
                .map(|context| WalrusDiagnostic::source_error(self.to_string(), context)),
        }
    }

    pub fn write_cli<W: Write>(&self, mut writer: W) -> io::Result<()> {
        if let Some(diagnostic) = self.diagnostic() {
            diagnostic.write(&mut writer)
        } else {
            writeln!(writer, "[ERROR] Fatal exception during execution -> {self}")
        }
    }

    pub fn with_call_site(self, context: ErrorContext) -> Self {
        if self.source_context().is_some() {
            return self;
        }

        WalrusError::RuntimeError {
            message: self.to_string(),
            context,
        }
    }

    fn source_context(&self) -> Option<&ErrorContext> {
        macro_rules! source_context_match {
            ($error:expr, $($variant:ident),+ $(,)?) => {
                match $error {
                    $(
                        WalrusError::$variant { context, .. } => Some(context),
                    )+
                    WalrusError::RuntimeErrorWithStackTrace { error, .. } => error.source_context(),
                    _ => None,
                }
            };
        }

        source_context_match!(
            self,
            UnexpectedEndOfInput,
            UnexpectedToken,
            InvalidToken,
            ExtraToken,
            NumberTooLarge,
            InvalidOperation,
            InvalidUnaryOperation,
            UndefinedVariable,
            ReturnOutsideFunction,
            BreakOutsideLoop,
            ContinueOutsideLoop,
            TypeMismatch,
            Exception,
            RuntimeError,
            InvalidEscapeSequence,
            InvalidUnicodeEscapeSequence,
            FStringParseError,
            FailedFree,
            NotCallable,
            InvalidArgCount,
            IndexOutOfBounds,
            NotIndexable,
            InvalidIndexType,
            AccessReleasedMemory,
            NotIterable,
            DivisionByZero,
            NoLength,
            StackUnderflow,
            InvalidInstruction,
            RedefinedLocal,
            KeyNotFound,
            InvalidRange,
            MethodNotFound,
            MemberNotFound,
            InvalidMemberAccessTarget,
            StructMethodMustBeVmFunction,
            InvalidMethodReceiver,
            ModuleNotFound,
            ThrownValue,
            EmptyListPop,
            InvalidGcThresholdArg,
            PackageImportNotImplemented,
        )
    }
}

pub fn parser_err_mapper(
    err: ParseError<usize, Token<'_>, RecoveredParseError>,
    source: Arc<str>,
    filename: Arc<str>,
) -> WalrusError {
    let context =
        |span| ErrorContext::from_shared(span, Arc::clone(&source), Arc::clone(&filename));

    match err {
        ParseError::UnrecognizedEOF {
            expected: _,
            location,
        } => WalrusError::UnexpectedEndOfInput {
            context: context(Span(location, location)),
        },
        ParseError::UnrecognizedToken {
            token: (start, token, end),
            expected: _,
        } => WalrusError::UnexpectedToken {
            token: token.to_string(),
            context: context(Span(start, end)),
        },
        ParseError::InvalidToken { location } => WalrusError::InvalidToken {
            token: source
                .get(location..location.saturating_add(1))
                .unwrap_or("<eof>")
                .to_string(),
            context: context(Span(location, location + 1)),
        },
        ParseError::ExtraToken {
            token: (start, token, end),
        } => WalrusError::ExtraToken {
            token: token.to_string(),
            context: context(Span(start, end)),
        },
        ParseError::User { error } => match error {
            RecoveredParseError::NumberTooLarge(number, span) => WalrusError::NumberTooLarge {
                number,
                context: context(span),
            },
            RecoveredParseError::InvalidEscapeSequence(sequence, span) => {
                WalrusError::InvalidEscapeSequence {
                    sequence,
                    context: context(span),
                }
            }
            RecoveredParseError::InvalidUnicodeEscapeSequence(span) => {
                WalrusError::InvalidUnicodeEscapeSequence {
                    context: context(span),
                }
            }
            RecoveredParseError::InvalidFStringExpression(expr, span) => {
                WalrusError::FStringParseError {
                    expr,
                    error: "invalid expression syntax".to_string(),
                    context: context(span),
                }
            }
        },
    }
}
