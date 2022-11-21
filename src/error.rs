use git_version::git_version;
use lalrpop_util::lexer::Token;
use lalrpop_util::ParseError;
use line_span::{find_line_end, find_line_start};
use std::cmp::min;
use std::ops::Range;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GlassError {
    #[error("Unknown error '{message}'. Please report this bug with the following information: Glass Version = '{}', Git Revision = '{}'", env!("CARGO_PKG_VERSION"), git_version!(fallback = "<unknown>"))]
    UnknownError { message: String },

    #[error("Unable to locate file '{filename}'. Make sure the file exists and that you have permission to read it.")]
    FileNotFound { filename: String },

    #[error("Unexpected EOF in source file '{filename}'.")]
    UnexpectedEndOfInput { filename: String },

    #[error("Unexpected token '{token}' at {line}")]
    UnexpectedToken { token: String, line: String },

    #[error("Invalid token '{token}' at {line}'")]
    InvalidToken { token: String, line: String },

    #[error("Extra token '{token}' at {line}'")]
    ExtraToken { token: String, line: String },

    #[error("")]
    LalrpopNumberTooLarge { span: Range<usize>, number: String },

    #[error("Number '{number}' is too large at {line}")]
    NumberTooLarge { number: String, line: String },
}

pub fn err_mapper(
    err: ParseError<usize, Token<'_>, GlassError>,
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
            line: get_line(source, filename, start..end),
        },
        ParseError::InvalidToken { location } => GlassError::InvalidToken {
            token: source[location..location + 1].to_string(),
            line: get_line(source, filename, location..location + 1),
        },
        ParseError::ExtraToken {
            token: (start, token, end),
        } => GlassError::ExtraToken {
            token: token.to_string(),
            line: get_line(source, filename, start..end),
        },
        ParseError::User { error } => match error {
            GlassError::LalrpopNumberTooLarge { number, span } => GlassError::NumberTooLarge {
                number,
                line: get_line(source, filename, span),
            },
            _ => GlassError::UnknownError {
                message: error.to_string(),
            },
        },
    }
}

fn get_line<'a>(src: &'a str, filename: &'a str, span: Range<usize>) -> String {
    let start = find_line_start(src, span.start);
    let end = find_line_end(src, span.start);
    let line = &src[start..end].trim();
    let line_num = src[..span.start].lines().count();
    let affected_range = span.start - start..span.end - start;

    format!(
        "\n\n\t{line}\n\t{}{}\n[{filename}(Ln:{line_num}, Col:{affected_range:?})]",
        &" ".repeat(affected_range.start),
        &"^".repeat(min(affected_range.len(), line.len() - affected_range.start)),
    )
}
