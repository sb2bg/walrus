use git_version::git_version;
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

    #[error("Unexpected token '{token}' at {span:?} in source file '{filename}'")]
    UnexpectedToken {
        token: String,
        span: Range<usize>,
        filename: String,
    },

    #[error("Invalid token at {index} in source file '{filename}'")]
    InvalidToken { index: usize, filename: String },

    #[error("Extra token '{token}' at {span:?} in source file '{filename}'")]
    ExtraToken {
        token: String,
        span: Range<usize>,
        filename: String,
    },

    #[error("")]
    LalrpopNumberTooLarge { span: Range<usize>, number: String },

    #[error("Number '{number}' at {span:?} in source file '{filename}' is too large")]
    NumberTooLarge {
        number: String,
        span: Range<usize>,
        filename: String,
    },
}
