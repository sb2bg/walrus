mod arenas;
mod ast;
mod error;
mod interpreter;
mod program;
Begin work mod range;
mod scope;
mod source_ref;
mod span;
mod type_checker;
mod value;

use crate::error::WalrusError;
use crate::interpreter::InterpreterResult;
use crate::program::Program;
use clap::Parser as ClapParser;
use lalrpop_util::lalrpop_mod;
use log::LevelFilter;
use simplelog::SimpleLogger;
use std::panic;
use std::path::PathBuf;

#[cfg(not(feature = "dhat-heap"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// profile heap usage with dhat
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

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

type WalrusResult = Result<(), WalrusError>;

fn main() {
    panic::set_hook(Box::new(|err| {
        eprintln!(
            "{}",
            WalrusError::UnknownError {
                message: err.to_string(),
            }
        );
    }));

    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    if let Err(err) = try_main() {
        eprintln!();
        eprintln!("[ERROR] Fatal exception during execution -> {}", err);
        eprintln!();
    }
}

fn try_main() -> WalrusResult {
    let args = Args::parse();
    setup_logger(args.debug)?;
    create_shell(args.file)?;
    Ok(())
}

pub fn create_shell(file: Option<PathBuf>) -> InterpreterResult {
    match file {
        Some(file) => Ok(Program::new(file, None)?.execute_file()?),
        None => Ok(Program::new(PathBuf::from("fixme"), None)?.execute_repl()?), // fixme: make path optional
    }
}

fn setup_logger(debug: bool) -> WalrusResult {
    let level = if debug {
        LevelFilter::Debug
    } else {
        LevelFilter::Info
    };

    match SimpleLogger::init(level, simplelog::Config::default()) {
        Ok(_) => Ok(()),
        Err(err) => Err(WalrusError::UnknownError {
            message: err.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    // TODO: Add tests
}
