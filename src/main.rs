mod ast;
mod error;
mod interpreter;
mod program;
mod scope;
mod source_ref;
mod span;
mod type_checker;
mod value;

use crate::error::GlassError;
use crate::program::Program;
use clap::Parser as ClapParser;
use lalrpop_util::lalrpop_mod;
use log::{error, LevelFilter};
use simplelog::SimpleLogger;
use std::io::Write;
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

type GlassResult = Result<(), GlassError>;

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
        error!("Fatal exception during execution -> {}", err);
    }
}

fn try_main() -> GlassResult {
    let args = Args::parse();
    setup_logger(args.debug)?;
    create_shell(args.file)
}

fn create_shell(file: Option<PathBuf>) -> GlassResult {
    match file {
        Some(file) => {
            let filename = file.to_string_lossy();

            let src = fs::read_to_string(&file).map_err(|_| GlassError::FileNotFound {
                filename: filename.to_string(),
            })?;

            Program::new(&filename, &src).run()?;
        }
        None => loop {
            let mut input = String::new();

            print!("REPL > ");

            std::io::stdout()
                .flush()
                .map_err(|_| GlassError::UnknownError {
                    message: "REPL failed to flush stdout".into(),
                })?;

            std::io::stdin()
                .read_line(&mut input)
                .map_err(|_| GlassError::UnknownError {
                    message: "REPL failed to read from stdin".into(),
                })?;

            Program::new("REPL", &input).run()?; // fixme: don't create a new program every time
        },
    }

    Ok(())
}

fn setup_logger(debug: bool) -> GlassResult {
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

#[cfg(test)]
mod tests {
    // TODO: Add tests
}
