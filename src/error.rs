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

pub enum RecoveredParseError {
    NumberTooLarge(String, Span),
    InvalidEscapeSequence(String, Span),
    InvalidUnicodeEscapeSequence(Span),
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

    #[error("Unable to locate file '{filename}'. Make sure the file exists and that you have permission to read it.")]
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
