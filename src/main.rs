mod ast;
mod error;

use crate::error::GlassError;
use clap::Parser as ClapParser;
use lalrpop_util::lexer::Token;
use lalrpop_util::{lalrpop_mod, ParseError};
use log::{debug, LevelFilter};
use simplelog::SimpleLogger;
use std::path::PathBuf;
use std::{fs, panic};

lalrpop_mod!(pub grammar);

#[derive(ClapParser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(help = "The script file to run", index = 1)]
    file: Option<PathBuf>,

    #[clap(short = 'd', long = "debug", help = "Enable debug mode")]
    debug: bool,

    #[clap(short = 'v', long = "verbose", help = "Enable verbose mode")]
    verbose: bool,
}

fn main() {
    panic::set_hook(Box::new(|err| {
        eprintln!(
            "{}",
            GlassError::UnknownError {
                message: err.to_string(),
            }
        );
    }));

    if let Err(err) = try_main() {
        eprintln!("Fatal exception during execution -> {}", err);
    }
}

fn try_main() -> Result<(), GlassError> {
    let args = Args::parse();
    setup_logger(args.debug)?;

    match args.file {
        Some(file) => run_script(file),
        None => Err(GlassError::UnknownError {
            message: "REPL not implemented yet".into(),
        }),
    }
}

fn run_script(file: PathBuf) -> Result<(), GlassError> {
    let filename = file.to_string_lossy().to_string();

    let src = match fs::read_to_string(&file) {
        Ok(src) => src,
        Err(_) => {
            return Err(GlassError::FileNotFound { filename });
        }
    };

    debug!("Read {} bytes from '{}'", &src.len(), &file.display());

    let ast = *grammar::ProgramParser::new()
        .parse(&src)
        .map_err(|err| err_mapper(err, filename))?;

    debug!("AST > {:?}", ast);

    Ok(())
}

fn err_mapper(err: ParseError<usize, Token<'_>, GlassError>, filename: String) -> GlassError {
    match err {
        ParseError::UnrecognizedEOF {
            expected: _,
            location: _,
        } => GlassError::UnexpectedEndOfInput { filename },
        ParseError::UnrecognizedToken {
            token: (start, token, end),
            expected: _,
        } => GlassError::UnexpectedToken {
            filename,
            token: token.to_string(),
            span: start..end,
        },
        ParseError::InvalidToken { location } => GlassError::InvalidToken {
            filename,
            index: location,
        },
        ParseError::ExtraToken {
            token: (start, token, end),
        } => GlassError::ExtraToken {
            filename,
            token: token.to_string(),
            span: start..end,
        },
        ParseError::User { error } => match error {
            GlassError::LalrpopNumberTooLarge { number, span } => GlassError::NumberTooLarge {
                number,
                span,
                filename,
            },
            _ => GlassError::UnknownError {
                message: error.to_string(),
            },
        },
    }
}

fn setup_logger(debug: bool) -> Result<(), GlassError> {
    let level = if debug {
        LevelFilter::Debug
    } else {
        LevelFilter::Info
    };

    match SimpleLogger::init(level, simplelog::Config::default()) {
        Ok(_) => Ok(()),
        Err(err) => Err(GlassError::UnknownError {
            message: err.to_string(),
        }),
    }
}
