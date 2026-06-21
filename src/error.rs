use std::io::{self, Write};
use std::ops::Range;
use std::rc::Rc;
use std::str::FromStr;

use ariadne::{ColorGenerator, Config, IndexType, Label, Report, ReportKind, sources};
use float_ord::FloatOrd;
use git_version::git_version;
use lalrpop_util::ParseError;
use lalrpop_util::lexer::Token;
use snailquote::{UnescapeError, unescape};
use thiserror::Error;

use crate::ast::NodeKind;
use crate::source_ref::SourceMap;
use crate::span::{Span, Spanned};
use crate::vm::opcode::Opcode;

const FSTRING_INTERP_QUOTE_SENTINEL: char = '\u{001F}';

#[derive(Clone, Copy)]
enum FStringInterpStringMode {
    Raw,
    Escaped,
}

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
    fn has_matching_interp_string_quote(
        chars: std::iter::Peekable<std::str::Chars<'_>>,
        mode: FStringInterpStringMode,
    ) -> bool {
        let mut escaped = false;

        for ch in chars {
            match mode {
                FStringInterpStringMode::Raw => {
                    if escaped {
                        escaped = false;
                    } else if ch == '\\' {
                        escaped = true;
                    } else if ch == '"' {
                        return true;
                    }
                }
                FStringInterpStringMode::Escaped => {
                    if ch == '"' {
                        return true;
                    }
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
            interp_string_mode: Option<FStringInterpStringMode>,
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
                        interp_string_mode: None,
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
                mut interp_string_mode,
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
                        interp_string_mode = None;
                    } else if ch == '"' {
                        state = ScannerState::Normal;
                        continue;
                    }
                } else {
                    if let Some(mode) = interp_string_mode {
                        let emitted = if ch == '"' {
                            FSTRING_INTERP_QUOTE_SENTINEL
                        } else {
                            ch
                        };
                        out.push(emitted);
                        prev_char = Some(ch);

                        match mode {
                            FStringInterpStringMode::Raw => {
                                if ch == '\\' {
                                    literal_escape = !literal_escape;
                                } else {
                                    if ch == '"' && !literal_escape {
                                        interp_string_mode = None;
                                    }
                                    literal_escape = false;
                                }
                            }
                            FStringInterpStringMode::Escaped => {
                                if ch == '"' {
                                    interp_string_mode = None;
                                }
                            }
                        }
                    } else {
                        if ch == '"' {
                            let mode = if prev_char == Some('\\') {
                                FStringInterpStringMode::Escaped
                            } else {
                                FStringInterpStringMode::Raw
                            };
                            if has_matching_interp_string_quote(chars.clone(), mode) {
                                out.push(FSTRING_INTERP_QUOTE_SENTINEL);
                                prev_char = Some(ch);
                                interp_string_mode = Some(mode);
                                literal_escape = false;
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
                    interp_string_mode,
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
    let base_span = spanned.span();
    let base_offset = base_span.0; // Start position of the f-string

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

    fn decode_next_fstring_escape<T>(
        char_indices: &mut std::iter::Peekable<std::str::CharIndices<'_>>,
        base_span: Span,
        content_offset: usize,
        slash_pos: usize,
    ) -> Result<(char, char), ParseError<usize, T, RecoveredParseError>> {
        if let Some((_, next_ch)) = char_indices.next() {
            let escape_span = base_span.in_same_file(
                content_offset + slash_pos,
                content_offset + slash_pos + 1 + next_ch.len_utf8(),
            );
            Ok((decode_fstring_escape(next_ch, escape_span)?, next_ch))
        } else {
            let escape_span =
                base_span.in_same_file(content_offset + slash_pos, content_offset + slash_pos + 1);
            Err(ParseError::User {
                error: RecoveredParseError::InvalidEscapeSequence("\\".to_string(), escape_span),
            })
        }
    }

    while let Some((byte_pos, ch)) = char_indices.next() {
        if ch == '\\' {
            let (decoded, _) =
                decode_next_fstring_escape(&mut char_indices, base_span, content_offset, byte_pos)?;
            current_literal.push(decoded);
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
            let mut string_mode = None;
            let mut string_escape = false;

            while let Some((expr_pos, expr_ch)) = char_indices.next() {
                if let Some(mode) = string_mode {
                    match mode {
                        FStringInterpStringMode::Raw => {
                            expr_str.push(expr_ch);

                            if string_escape {
                                string_escape = false;
                            } else if expr_ch == '\\' {
                                string_escape = true;
                            } else if expr_ch == '"' {
                                string_mode = None;
                            }
                        }
                        FStringInterpStringMode::Escaped => {
                            let emitted = if expr_ch == '\\' {
                                let (decoded, _) = decode_next_fstring_escape(
                                    &mut char_indices,
                                    base_span,
                                    content_offset,
                                    expr_pos,
                                )?;
                                decoded
                            } else {
                                expr_ch
                            };

                            if string_escape {
                                string_escape = false;
                            } else if emitted == '\\' {
                                string_escape = true;
                            } else if emitted == '"' {
                                string_mode = None;
                            }

                            expr_str.push(emitted);
                        }
                    }

                    continue;
                }

                let (emitted, escaped_quote) = if expr_ch == '\\' {
                    let (decoded, next_ch) = decode_next_fstring_escape(
                        &mut char_indices,
                        base_span,
                        content_offset,
                        expr_pos,
                    )?;
                    (decoded, next_ch == '"')
                } else {
                    (expr_ch, false)
                };

                if emitted == '"' {
                    string_mode = Some(if escaped_quote {
                        FStringInterpStringMode::Escaped
                    } else {
                        FStringInterpStringMode::Raw
                    });
                    string_escape = false;
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
                let expr_span = base_span
                    .in_same_file(content_offset + expr_start, content_offset + content.len());
                return Err(ParseError::User {
                    error: RecoveredParseError::InvalidFStringExpression(expr_str, expr_span),
                });
            }

            // Parse the expression with proper span
            if expr_str.trim().is_empty() {
                let expr_span = base_span
                    .in_same_file(content_offset + expr_start, content_offset + expr_start);
                return Err(ParseError::User {
                    error: RecoveredParseError::InvalidFStringExpression(expr_str, expr_span),
                });
            }

            match parser.parse(base_span.file_id(), &expr_str) {
                Ok(node) => {
                    // Adjust the span to reflect the actual position in the source file
                    let expr_span = node.span();
                    let adjusted_span = base_span.in_same_file(
                        content_offset + expr_start + expr_span.0,
                        content_offset + expr_start + expr_span.1,
                    );
                    // Create a new node with the adjusted span
                    let adjusted_node = crate::ast::Node::new(node.into_kind(), adjusted_span);
                    parts.push(FStringPart::Expr(Box::new(adjusted_node)));
                }
                Err(_) => {
                    // If parsing fails, return an error
                    let expr_span = base_span.in_same_file(
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
    source_map: SourceMap,
}

impl ErrorContext {
    pub fn new(span: Span, src: impl Into<Rc<str>>, filename: impl Into<Rc<str>>) -> Self {
        let source_map = SourceMap::new();
        let (file_id, _) = source_map.add_source(src, filename);
        let span = if span.file_id().is_unknown() {
            span.with_file_id(file_id)
        } else {
            span
        };
        Self { span, source_map }
    }

    pub fn from_source_map(span: Span, source_map: SourceMap) -> Self {
        Self { span, source_map }
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn source_map(&self) -> SourceMap {
        self.source_map.clone()
    }

    fn source_file(&self, span: Span) -> Option<crate::source_ref::SourceFileRef> {
        self.source_map.get(span.file_id())
    }

    pub fn with_span(&self, span: Span) -> Self {
        Self {
            span,
            source_map: self.source_map(),
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

    fn source_error_with_labels(
        message: impl Into<String>,
        context: ErrorContext,
        labels: Vec<DiagnosticLabel>,
    ) -> Self {
        Self {
            message: message.into(),
            source: Some(context),
            labels,
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
        let Some(primary_file) = source.source_file(primary_span) else {
            writeln!(writer, "Error: {}", self.message)?;
            for note in &self.notes {
                writeln!(writer, "Note: {note}")?;
            }
            return Ok(());
        };

        let primary_source = primary_file.source;
        let primary_filename = primary_file.filename.to_string();
        let primary_range = diagnostic_range(&primary_source, primary_span);
        let mut builder = Report::build(
            ReportKind::Error,
            (primary_filename.clone(), primary_range.clone()),
        )
        .with_config(
            Config::default()
                .with_color(true)
                .with_index_type(IndexType::Byte),
        )
        .with_message(&self.message);

        let mut colors = ColorGenerator::new();
        for label in &self.labels {
            let Some(label_file) = source.source_file(label.span) else {
                continue;
            };
            let mut ariadne_label = Label::new((
                label_file.filename.to_string(),
                diagnostic_range(&label_file.source, label.span),
            ))
            .with_color(colors.next());
            if let Some(message) = &label.message {
                ariadne_label = ariadne_label.with_message(message);
            }
            builder = builder.with_label(ariadne_label);
        }

        for note in &self.notes {
            builder = builder.with_note(note);
        }

        let mut source_entries = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for label in &self.labels {
            if let Some(file) = source.source_file(label.span) {
                let filename = file.filename.to_string();
                if seen.insert(filename.clone()) {
                    source_entries.push((filename, file.source.to_string()));
                }
            }
        }
        if seen.insert(primary_filename.clone()) {
            source_entries.push((primary_filename, primary_source.to_string()));
        }

        builder.finish().write(sources(source_entries), &mut writer)
    }
}

fn diagnostic_range(src: &str, span: Span) -> Range<usize> {
    let len = src.len();
    let start = span.0.min(len);
    let end = span.1.min(len).max(start);
    start..end
}

fn label_span_for_context(span: Span, context: &ErrorContext) -> Option<Span> {
    if span == Span::default() {
        None
    } else if span.file_id().is_unknown() {
        Some(span.with_file_id(context.span().file_id()))
    } else {
        Some(span)
    }
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
        left_span: Option<Span>,
        right_span: Option<Span>,
        context: ErrorContext,
    },

    #[error("Invalid unary operation '{op}' on operand {operand}")]
    InvalidUnaryOperation {
        op: Opcode,
        operand: String,
        operand_span: Option<Span>,
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
        index_span: Option<Span>,
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
        target_span: Option<Span>,
        index_span: Option<Span>,
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
        start_span: Option<Span>,
        end_span: Option<Span>,
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
            WalrusError::InvalidOperation {
                op,
                left,
                right,
                left_span,
                right_span,
                context,
            } => {
                let mut labels = Vec::new();
                let (left_label, right_label) = match op {
                    Opcode::Index => (
                        format!("indexed value is {left}"),
                        format!("index is {right}"),
                    ),
                    Opcode::Range => (
                        format!("range start is {left}"),
                        format!("range end is {right}"),
                    ),
                    _ => (
                        format!("left operand is {left}"),
                        format!("right operand is {right}"),
                    ),
                };
                if let Some(span) = left_span.and_then(|span| label_span_for_context(span, context))
                {
                    labels.push(DiagnosticLabel {
                        span,
                        message: Some(left_label),
                    });
                }
                if let Some(span) =
                    right_span.and_then(|span| label_span_for_context(span, context))
                {
                    labels.push(DiagnosticLabel {
                        span,
                        message: Some(right_label),
                    });
                }
                if labels.is_empty() {
                    labels.push(DiagnosticLabel {
                        span: context.span(),
                        message: Some(format!("{left} cannot be combined with {right}")),
                    });
                }
                Some(WalrusDiagnostic::source_error_with_labels(
                    self.to_string(),
                    context.clone(),
                    labels,
                ))
            }
            WalrusError::InvalidUnaryOperation {
                operand,
                operand_span,
                context,
                ..
            } => {
                let span = operand_span
                    .and_then(|span| label_span_for_context(span, context))
                    .unwrap_or_else(|| context.span());
                Some(WalrusDiagnostic::source_error_with_labels(
                    self.to_string(),
                    context.clone(),
                    vec![DiagnosticLabel {
                        span,
                        message: Some(format!("operand is {operand}")),
                    }],
                ))
            }
            _ => self.source_context().cloned().map(|context| {
                WalrusDiagnostic::source_error_with_labels(
                    self.to_string(),
                    context.clone(),
                    self.diagnostic_labels(&context),
                )
            }),
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

    fn diagnostic_labels(&self, context: &ErrorContext) -> Vec<DiagnosticLabel> {
        let message = match self {
            WalrusError::UnexpectedEndOfInput { .. } => {
                "input ends before this construct is complete"
            }
            WalrusError::UnexpectedToken { .. } => "unexpected token",
            WalrusError::InvalidToken { .. } => "invalid token",
            WalrusError::ExtraToken { .. } => "extra token",
            WalrusError::NumberTooLarge { .. } => "number is too large",
            WalrusError::UndefinedVariable { .. } => "name is not defined in this scope",
            WalrusError::ReturnOutsideFunction { .. } => "return is only valid inside a function",
            WalrusError::BreakOutsideLoop { .. } => "break is only valid inside a loop",
            WalrusError::ContinueOutsideLoop { .. } => "continue is only valid inside a loop",
            WalrusError::TypeMismatch {
                expected, found, ..
            } => {
                return vec![DiagnosticLabel {
                    span: context.span(),
                    message: Some(format!("expected {expected}, found {found}")),
                }];
            }
            WalrusError::Exception { .. } => "exception thrown here",
            WalrusError::RuntimeError { .. } => "runtime error occurs here",
            WalrusError::InvalidEscapeSequence { .. } => "invalid escape sequence",
            WalrusError::InvalidUnicodeEscapeSequence { .. } => "invalid unicode escape",
            WalrusError::FStringParseError { .. } => "invalid expression inside f-string",
            WalrusError::FailedFree { .. } => "value is not heap allocated",
            WalrusError::NotCallable { value, .. } => {
                return vec![DiagnosticLabel {
                    span: context.span(),
                    message: Some(format!("{value} values cannot be called")),
                }];
            }
            WalrusError::InvalidArgCount { expected, got, .. } => {
                return vec![DiagnosticLabel {
                    span: context.span(),
                    message: Some(format!("expected {expected} argument(s), got {got}")),
                }];
            }
            WalrusError::IndexOutOfBounds {
                index,
                len,
                index_span,
                ..
            } => {
                return vec![DiagnosticLabel {
                    span: index_span
                        .and_then(|span| label_span_for_context(span, context))
                        .unwrap_or_else(|| context.span()),
                    message: Some(format!("index {index} is outside length {len}")),
                }];
            }
            WalrusError::NotIndexable { value, .. } => {
                return vec![DiagnosticLabel {
                    span: context.span(),
                    message: Some(format!("{value} values cannot be indexed")),
                }];
            }
            WalrusError::InvalidIndexType {
                non_indexable,
                index_type,
                target_span,
                index_span,
                ..
            } => {
                let mut labels = Vec::new();
                if let Some(span) =
                    target_span.and_then(|span| label_span_for_context(span, context))
                {
                    labels.push(DiagnosticLabel {
                        span,
                        message: Some(format!("indexed value is {non_indexable}")),
                    });
                }
                if let Some(span) =
                    index_span.and_then(|span| label_span_for_context(span, context))
                {
                    labels.push(DiagnosticLabel {
                        span,
                        message: Some(format!("index is {index_type}")),
                    });
                }
                if labels.is_empty() {
                    labels.push(DiagnosticLabel {
                        span: context.span(),
                        message: Some(format!("cannot index {non_indexable} with {index_type}")),
                    });
                }
                return labels;
            }
            WalrusError::AccessReleasedMemory { .. } => "released memory accessed here",
            WalrusError::NotIterable { type_name, .. } => {
                return vec![DiagnosticLabel {
                    span: context.span(),
                    message: Some(format!("{type_name} values are not iterable")),
                }];
            }
            WalrusError::DivisionByZero { .. } => "division by zero",
            WalrusError::NoLength { type_name, .. } => {
                return vec![DiagnosticLabel {
                    span: context.span(),
                    message: Some(format!("{type_name} values do not have length")),
                }];
            }
            WalrusError::StackUnderflow { .. } => "the VM stack did not contain enough values",
            WalrusError::InvalidInstruction { .. } => "invalid bytecode instruction",
            WalrusError::RedefinedLocal { .. } => "local was already defined",
            WalrusError::KeyNotFound { .. } => "key lookup failed here",
            WalrusError::InvalidRange {
                start,
                end,
                start_span,
                end_span,
                ..
            } => {
                let mut labels = Vec::new();
                if let Some(span) =
                    start_span.and_then(|span| label_span_for_context(span, context))
                {
                    labels.push(DiagnosticLabel {
                        span,
                        message: Some(format!("range starts at {start}")),
                    });
                }
                if let Some(span) = end_span.and_then(|span| label_span_for_context(span, context))
                {
                    labels.push(DiagnosticLabel {
                        span,
                        message: Some(format!("range ends at {end}")),
                    });
                }
                if labels.is_empty() {
                    labels.push(DiagnosticLabel {
                        span: context.span(),
                        message: Some("range bounds are invalid".to_string()),
                    });
                }
                return labels;
            }
            WalrusError::MethodNotFound {
                method, type_name, ..
            } => {
                return vec![DiagnosticLabel {
                    span: context.span(),
                    message: Some(format!("{type_name} has no method named {method}")),
                }];
            }
            WalrusError::MemberNotFound {
                member, type_name, ..
            } => {
                return vec![DiagnosticLabel {
                    span: context.span(),
                    message: Some(format!("{type_name} has no member named {member}")),
                }];
            }
            WalrusError::InvalidMemberAccessTarget { .. } => "member access target is invalid",
            WalrusError::StructMethodMustBeVmFunction { .. } => "struct method is not VM bytecode",
            WalrusError::InvalidMethodReceiver {
                method, type_name, ..
            } => {
                return vec![DiagnosticLabel {
                    span: context.span(),
                    message: Some(format!("cannot call {method} on {type_name}")),
                }];
            }
            WalrusError::ModuleNotFound { .. } => "module lookup failed here",
            WalrusError::ThrownValue { .. } => "value thrown here",
            WalrusError::EmptyListPop { .. } => "list is empty here",
            WalrusError::InvalidGcThresholdArg { .. } => "argument must be a positive integer",
            WalrusError::PackageImportNotImplemented { .. } => "package import starts here",
            _ => "error occurs here",
        };

        vec![DiagnosticLabel {
            span: context.span(),
            message: Some(message.to_string()),
        }]
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
    source_ref: &crate::source_ref::SourceRef<'_>,
) -> WalrusError {
    let context = |span| source_ref.error_context(span);

    match err {
        ParseError::UnrecognizedEOF {
            expected: _,
            location,
        } => WalrusError::UnexpectedEndOfInput {
            context: context(Span::new(source_ref.file_id(), location, location)),
        },
        ParseError::UnrecognizedToken {
            token: (start, token, end),
            expected: _,
        } => WalrusError::UnexpectedToken {
            token: token.to_string(),
            context: context(Span::new(source_ref.file_id(), start, end)),
        },
        ParseError::InvalidToken { location } => WalrusError::InvalidToken {
            token: source_ref
                .source()
                .get(location..location.saturating_add(1))
                .unwrap_or("<eof>")
                .to_string(),
            context: context(Span::new(source_ref.file_id(), location, location + 1)),
        },
        ParseError::ExtraToken {
            token: (start, token, end),
        } => WalrusError::ExtraToken {
            token: token.to_string(),
            context: context(Span::new(source_ref.file_id(), start, end)),
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
