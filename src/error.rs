use std::cmp::min;
use std::str::FromStr;

use float_ord::FloatOrd;
use git_version::git_version;
use lalrpop_util::ParseError;
use lalrpop_util::lexer::Token;
use line_span::{find_line_end, find_line_start};
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
        mut chars: std::iter::Peekable<std::str::Chars<'_>>,
        mut interp_depth: usize,
    ) -> bool {
        while let Some(ch) = chars.next() {
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
                    && prev_char.map_or(true, |prev| !is_ident_char(prev));

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

    lalrpop_mod!(pub grammar);
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

// todo: accept &str instead of String, and source_refs when possible
// fixme: lower error size
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

    #[error("Unexpected EOF in source '{filename}'.")]
    UnexpectedEndOfInput { filename: String },

    #[error("Unexpected token '{token}' at {line}")]
    UnexpectedToken { token: String, line: String },

    #[error("Invalid token '{token}' at {line}")]
    InvalidToken { token: String, line: String },

    #[error("Extra token '{token}' at {line}")]
    ExtraToken { token: String, line: String },

    #[error("Number '{number}' is too large at {line}")]
    NumberTooLarge { number: String, line: String },

    #[error("Circular import detected: module '{module}' is already being imported")]
    CircularImport { module: String },

    #[error("Invalid argument combination: '{first_arg}' cannot be used with '{second_arg}'")]
    InvalidArgumentCombination {
        first_arg: String,
        second_arg: String,
    },

    #[error(
        "Invalid operation '{op}' on operands '{left}' and '{right}' at {}",
        get_line(src, filename, *span)
    )]
    InvalidOperation {
        op: Opcode,
        left: String,
        right: String,
        span: Span,
        src: String,
        filename: String,
    },

    #[error(
        "Invalid unary operation '{op}' on operand {operand} at {}",
        get_line(src, filename, *span)
    )]
    InvalidUnaryOperation {
        op: Opcode,
        operand: String,
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Undefined variable '{name}' at {}",
        get_line(src, filename, *span)
    )]
    UndefinedVariable {
        name: String,
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Return statement outside of function at {}",
        get_line(src, filename, *span)
    )]
    ReturnOutsideFunction {
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Break statement outside of loop at {}",
        get_line(src, filename, *span)
    )]
    BreakOutsideLoop {
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Continue statement outside of loop at {}",
        get_line(src, filename, *span)
    )]
    ContinueOutsideLoop {
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Expected type '{expected}', but found type '{found}' at {}",
        get_line(src, filename, *span)
    )]
    TypeMismatch {
        expected: String,
        found: String,
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Exception '{message}' thrown at {}",
        get_line(src, filename, *span)
    )]
    Exception {
        message: String,
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Invalid escape sequence '{sequence}' at {line}")]
    InvalidEscapeSequence { sequence: String, line: String },

    #[error("Invalid unicode escape sequence at {line}")]
    InvalidUnicodeEscapeSequence { line: String },

    #[error("Failed to parse f-string expression '{expr}': {error} at {}",
        get_line(src, filename, *span)
    )]
    FStringParseError {
        expr: String,
        error: String,
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Cannot free memory not in the heap at {}",
        get_line(src, filename, *span)
    )]
    FailedFree {
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Cannot call non-function '{value}' at {}",
        get_line(src, filename, *span)
    )]
    NotCallable {
        value: String,
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Function '{name}' expected {expected} arg(s) but got {got} arg(s) at {}",
        get_line(src, filename, *span)
    )]
    InvalidArgCount {
        name: String,
        expected: usize,
        got: usize,
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Index {index} out of bounds for object with len {len} at {}",
        get_line(src, filename, *span)
    )]
    IndexOutOfBounds {
        index: i64,
        len: usize,
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Cannot index non-indexable type '{value}' at {}",
        get_line(src, filename, *span)
    )]
    NotIndexable {
        value: String,
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Cannot index type '{non_indexable}' with type '{index_type}' at {}",
        get_line(src, filename, *span)
    )]
    InvalidIndexType {
        non_indexable: String,
        index_type: String,
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Attempted to access released memory at {}. This is either a bug in the interpreter or you freed memory allocated by the interpreter.",
        get_line(src, filename, *span)
    )]
    AccessReleasedMemory {
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Failed to gather PWD. This may be due to a permissions error.")]
    FailedGatherPWD,

    #[error("Type '{type_name}' is not iterable at {}",
        get_line(src, filename, *span)
    )]
    NotIterable {
        type_name: String,
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Division by zero at {}",
        get_line(src, filename, *span)
    )]
    DivisionByZero {
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Cannot get length of non-indexable type '{type_name}' at {}",
        get_line(src, filename, *span)
    )]
    NoLength {
        type_name: String,
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Stack underflow while executing opcode '{op:?}' at {}",
        get_line(src, filename, *span)
    )]
    StackUnderflow {
        op: Opcode,
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Invalid instruction '{op:?}' at {}",
        get_line(src, filename, *span)
    )]
    InvalidInstruction {
        op: Opcode,
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Cannot redefine local variable with name '{name}' at {}",
        get_line(src, filename, *span)
    )]
    RedefinedLocal {
        name: String,
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Key '{key}' not found at {}",
        get_line(src, filename, *span)
    )]
    KeyNotFound {
        key: String,
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Range start '{start}' must be less than or equal to range end '{end}' at {}",
        get_line(src, filename, *span)
    )]
    InvalidRange {
        start: i64,
        end: i64,
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Type '{type_name}' has no method '{method}' at {}",
        get_line(src, filename, *span)
    )]
    MethodNotFound {
        type_name: String,
        method: String,
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Type '{type_name}' has no member '{member}' at {}",
        get_line(src, filename, *span)
    )]
    MemberNotFound {
        type_name: String,
        member: String,
        span: Span,
        src: String,
        filename: String,
    },

    #[error(
        "Member access requires object type 'struct', 'struct instance', or 'dict' and member name type 'string', found object '{object_type}' and member '{member_type}' at {}",
        get_line(src, filename, *span)
    )]
    InvalidMemberAccessTarget {
        object_type: String,
        member_type: String,
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Struct methods must compile to VM bytecode functions at {}",
        get_line(src, filename, *span)
    )]
    StructMethodMustBeVmFunction {
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Cannot call method '{method}' on value of type '{type_name}' at {}",
        get_line(src, filename, *span)
    )]
    InvalidMethodReceiver {
        method: String,
        type_name: String,
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Module '{module}' not found at {}",
        get_line(src, filename, *span)
    )]
    ModuleNotFound {
        module: String,
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Thrown value: {message} at {}",
        get_line(src, filename, *span)
    )]
    ThrownValue {
        message: String,
        span: Span,
        src: String,
        filename: String,
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

    #[error("Cannot pop from an empty list at {}",
        get_line(src, filename, *span)
    )]
    EmptyListPop {
        span: Span,
        src: String,
        filename: String,
    },

    #[error("core.gc_threshold requires a positive integer argument at {}",
        get_line(src, filename, *span)
    )]
    InvalidGcThresholdArg {
        span: Span,
        src: String,
        filename: String,
    },

    #[error("Package imports (@package) are not yet implemented at {}",
        get_line(src, filename, *span)
    )]
    PackageImportNotImplemented {
        span: Span,
        src: String,
        filename: String,
    },

    #[error("{error}{stack_trace}")]
    RuntimeErrorWithStackTrace { error: String, stack_trace: String },
}

pub fn parser_err_mapper(
    err: ParseError<usize, Token<'_>, RecoveredParseError>,
    source: &str,
    filename: &str,
) -> WalrusError {
    match err {
        ParseError::UnrecognizedEOF {
            expected: _,
            location: _,
        } => WalrusError::UnexpectedEndOfInput {
            filename: filename.into(),
        },
        ParseError::UnrecognizedToken {
            token: (start, token, end),
            expected: _,
        } => WalrusError::UnexpectedToken {
            token: token.to_string(),
            line: get_line(source, filename, Span(start, end)),
        },
        ParseError::InvalidToken { location } => WalrusError::InvalidToken {
            token: source
                .get(location..location.saturating_add(1))
                .unwrap_or("<eof>")
                .to_string(),
            line: get_line(source, filename, Span(location, location + 1)),
        },
        ParseError::ExtraToken {
            token: (start, token, end),
        } => WalrusError::ExtraToken {
            token: token.to_string(),
            line: get_line(source, filename, Span(start, end)),
        },
        ParseError::User { error } => match error {
            RecoveredParseError::NumberTooLarge(number, span) => WalrusError::NumberTooLarge {
                number,
                line: get_line(source, filename, span),
            },
            RecoveredParseError::InvalidEscapeSequence(sequence, span) => {
                WalrusError::InvalidEscapeSequence {
                    sequence,
                    line: get_line(source, filename, span),
                }
            }
            RecoveredParseError::InvalidUnicodeEscapeSequence(span) => {
                WalrusError::InvalidUnicodeEscapeSequence {
                    line: get_line(source, filename, span),
                }
            }
            RecoveredParseError::InvalidFStringExpression(expr, span) => {
                WalrusError::FStringParseError {
                    expr,
                    error: "invalid expression syntax".to_string(),
                    span,
                    src: source.to_string(),
                    filename: filename.to_string(),
                }
            }
        },
    }
}

// i dont know if this is the best place to put this comment but:
// fixme: if the error span starts on a different line than where the error is, the wrong line will be printed.
// should probably fix this by printing all of the lines contained in the spans
// todo: use codespan_reporting? https://github.com/brendanzab/codespan
fn get_line<'a>(src: &'a str, filename: &'a str, span: Span) -> String {
    if src.is_empty() {
        return format!("\n\n\t<empty source>\n\t^\n[{filename}:1:1]");
    }

    let src_len = src.len();
    let start_offset = span.0.min(src_len.saturating_sub(1));
    let mut end_offset = span.1.min(src_len);
    if end_offset <= start_offset {
        end_offset = (start_offset + 1).min(src_len);
    }

    let line_start = find_line_start(src, start_offset);
    let line_end = find_line_end(src, start_offset).min(src_len);
    let line = &src[line_start..line_end];
    let mut trimmed = line.trim_start();
    let trim_left = line.len() - trimmed.len();
    trimmed = trimmed.trim_end();

    let line_num = src[..start_offset].lines().count() + 1;
    let col_num = start_offset - line_start + 1;

    let caret_start = start_offset.saturating_sub(line_start + trim_left);
    let max_caret = trimmed.len().saturating_sub(caret_start);
    let desired_len = end_offset.saturating_sub(start_offset).max(1);
    let caret_len = min(desired_len, max_caret.max(1));

    format!(
        "\n\n\t{trimmed}\n\t{}{}\n[{filename}:{line_num}:{col_num}]",
        " ".repeat(caret_start),
        "^".repeat(caret_len),
    )
}
