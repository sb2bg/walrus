use std::cmp::min;
use std::str::FromStr;

use float_ord::FloatOrd;
use git_version::git_version;
use lalrpop_util::lexer::Token;
use lalrpop_util::ParseError;
use line_span::{find_line_end, find_line_start};
use snailquote::{unescape, UnescapeError};
use thiserror::Error;

use crate::ast::NodeKind;
use crate::span::{Span, Spanned};
use crate::vm::opcode::Opcode;

#[derive(Debug)]
pub enum RecoveredParseError {
    NumberTooLarge(String, Span),
    InvalidEscapeSequence(String, Span),
    InvalidUnicodeEscapeSequence(Span),
    InvalidFStringExpression(String, Span),
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
    let base_offset = spanned.span().0; // Start position of the f-string

    // Remove f" and trailing " -> f" is 2 chars, so content starts at base_offset + 2
    let content = &raw[2..raw.len() - 1];
    let content_offset = base_offset + 2;

    let mut parts = Vec::new();
    let mut current_literal = String::new();
    let mut char_indices = content.char_indices().peekable();

    while let Some((byte_pos, ch)) = char_indices.next() {
        if ch == '\\' {
            // Handle escape sequences
            if let Some(&(_, next_ch)) = char_indices.peek() {
                char_indices.next();
                match next_ch {
                    'n' => current_literal.push('\n'),
                    't' => current_literal.push('\t'),
                    'r' => current_literal.push('\r'),
                    '\\' => current_literal.push('\\'),
                    '"' => current_literal.push('"'),
                    '{' => current_literal.push('{'),
                    '}' => current_literal.push('}'),
                    _ => {
                        current_literal.push('\\');
                        current_literal.push(next_ch);
                    }
                }
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

            while let Some((_, expr_ch)) = char_indices.next() {
                if expr_ch == '{' {
                    brace_depth += 1;
                    expr_str.push(expr_ch);
                } else if expr_ch == '}' {
                    brace_depth -= 1;
                    if brace_depth == 0 {
                        break;
                    }
                    expr_str.push(expr_ch);
                } else {
                    expr_str.push(expr_ch);
                }
            }

            // Parse the expression with proper span
            if !expr_str.is_empty() {
                match parser.parse(&expr_str) {
                    Ok(node) => {
                        // Adjust the span to reflect the actual position in the source file
                        let expr_span = node.span();
                        let adjusted_span = crate::span::Span(
                            content_offset + expr_start + expr_span.0,
                            content_offset + expr_start + expr_span.1,
                        );
                        // Create a new node with the adjusted span
                        let adjusted_node = crate::ast::Node::new(node.into_kind(), adjusted_span);
                        parts.push(FStringPart::Expr(Box::new(adjusted_node)));
                    }
                    Err(_) => {
                        // If parsing fails, return an error
                        let expr_span = crate::span::Span(
                            content_offset + expr_start,
                            content_offset + expr_start + expr_str.len(),
                        );
                        return Err(ParseError::User {
                            error: RecoveredParseError::InvalidFStringExpression(
                                expr_str, expr_span,
                            ),
                        });
                    }
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

    #[error("Cannot execute node-based function bodies in the VM runtime at {}",
        get_line(src, filename, *span)
    )]
    NodeFunctionNotSupportedInVm {
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

    #[error("Module '{module}' not found. Available modules: std/io, std/sys, std/math at {}",
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

    #[error("__gc_threshold__ requires a positive integer argument at {}",
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
            token: source[location..location + 1].to_string(),
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
    let start = find_line_start(src, span.0);
    let end = find_line_end(src, span.0);
    let line = &src[start..end];
    let mut trimmed = line.trim_start();
    let diff = line.len() - trimmed.len();
    trimmed = trimmed.trim_end();
    let line_num = src[..span.0].lines().count(); // fixme: I have a sneaking suspicion this is wrong but only in some cases
    let affected_range = span.0 - start - diff..span.1 - start - diff;

    format!(
        "\n\n\t{trimmed}\n\t{}{}\n[{filename}:{line_num}:{}]",
        &" ".repeat(affected_range.start),
        &"^".repeat(min(affected_range.len(), line.len() - affected_range.start)),
        span.0 - start + 1,
    )
}
