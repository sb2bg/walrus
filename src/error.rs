use crate::ast::Op;
use crate::source_ref::SourceRef;
use crate::span::{Span, Spanned};
use crate::value::Value;
use git_version::git_version;
use lalrpop_util::lexer::Token;
use lalrpop_util::ParseError;
use line_span::{find_line_end, find_line_start};
use std::cmp::min;
use std::str::FromStr;
use thiserror::Error;

pub enum RecoveredParseError {
    NumberTooLarge(String, Span),
}

pub enum InterpreterError {
    InvalidOperation { op: Op, a: Value, b: Value },
    InvalidUnaryOperation { op: Op, value: Value },
}

pub fn parse_int<T>(
    Spanned { value, span }: Spanned<&str>,
    base: u32,
) -> Result<Spanned<i64>, ParseError<usize, T, RecoveredParseError>> {
    let num =
        i64::from_str_radix(if base == 10 { value } else { &value[2..] }, base).map_err(|_| {
            ParseError::User {
                error: RecoveredParseError::NumberTooLarge(value.to_string(), span),
            }
        })?;

    Ok(Spanned { value: num, span })
}

pub fn parse_float<T>(
    Spanned { value, span }: Spanned<&str>,
) -> Result<Spanned<f64>, ParseError<usize, T, RecoveredParseError>> {
    let num = f64::from_str(value).map_err(|_| ParseError::User {
        error: RecoveredParseError::NumberTooLarge(value.to_string(), span),
    })?;

    Ok(Spanned { value: num, span })
}

// todo: accept &str instead of String
#[derive(Error, Debug)]
pub enum GlassError {
    #[error("Unknown error '{message}'. Please report this bug with the following information: Glass Version = '{}', Git Revision = '{}'", env!("CARGO_PKG_VERSION"), git_version!(fallback = "<unknown>"))]
    UnknownError { message: String },

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
        op: Op,
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
        op: Op,
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
}

pub fn parser_err_mapper(
    err: ParseError<usize, Token<'_>, RecoveredParseError>,
    filename: &str,
    source: &str,
) -> GlassError {
    match err {
        ParseError::UnrecognizedEOF {
            expected: _,
            location: _,
        } => GlassError::UnexpectedEndOfInput {
            filename: filename.into(),
        },
        ParseError::UnrecognizedToken {
            token: (start, token, end),
            expected: _,
        } => GlassError::UnexpectedToken {
            token: token.to_string(),
            line: get_line(source, filename, Span(start, end)),
        },
        ParseError::InvalidToken { location } => GlassError::InvalidToken {
            token: source[location..location + 1].to_string(),
            line: get_line(source, filename, Span(location, location + 1)),
        },
        ParseError::ExtraToken {
            token: (start, token, end),
        } => GlassError::ExtraToken {
            token: token.to_string(),
            line: get_line(source, filename, Span(start, end)),
        },
        ParseError::User { error } => match error {
            RecoveredParseError::NumberTooLarge(number, span) => GlassError::NumberTooLarge {
                number,
                line: get_line(source, filename, span.into()),
            },
        },
    }
}

pub fn interpreter_err_mapper(
    err: InterpreterError,
    source_ref: &SourceRef,
    span: Span,
) -> GlassError {
    match err {
        InterpreterError::InvalidOperation { op, a, b } => GlassError::InvalidOperation {
            op,
            span,
            left: a.get_type().into(),
            right: b.get_type().into(),
            src: source_ref.source().into(),
            filename: source_ref.filename().into(),
        },
        InterpreterError::InvalidUnaryOperation { op, value } => {
            GlassError::InvalidUnaryOperation {
                op,
                span,
                operand: value.get_type().into(),
                src: source_ref.source().into(),
                filename: source_ref.filename().into(),
            }
        }
    }
}

// i dont know if this is the best place to put this comment but:
// fixme: if the error span starts on a different line than where the error is, the wrong line will be printed.
// should probably fix this by printing all of the lines contained in the span
fn get_line<'a>(src: &'a str, filename: &'a str, span: Span) -> String {
    let start = find_line_start(src, span.0);
    let end = find_line_end(src, span.0);
    let line = &src[start..end];
    let mut trimmed = line.trim_start();
    let diff = line.len() - trimmed.len();
    trimmed = trimmed.trim_end();
    let line_num = src[..span.0].lines().count();
    let affected_range = span.0 - start - diff..span.1 - start - diff;

    format!(
        "\n\n\t{trimmed}\n\t{}{}\n[{filename}(Ln:{line_num}, Col:{affected_range:?})]",
        &" ".repeat(affected_range.start),
        &"^".repeat(min(affected_range.len(), line.len() - affected_range.start)),
    )
}
