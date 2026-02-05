use std::path::PathBuf;
use std::{backtrace, panic};

use clap::Parser as ClapParser;
use lalrpop_util::lalrpop_mod;
use log::{LevelFilter, trace};
use simplelog::SimpleLogger;

use crate::error::WalrusError;
use crate::interpreter::InterpreterResult;
use crate::program::{JitOpts, Opts, Program};

mod arenas;
mod ast;
mod error;
mod function;
mod gc;
mod interpreter;
mod iter;
pub mod jit;
mod program;
mod range;
mod scope;
mod source_ref;
mod span;
mod stdlib;
mod structs;
mod value;
pub mod vm;

#[cfg(not(feature = "dhat-heap"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// profile heap usage with dhat
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

lalrpop_mod!(#[allow(clippy::all)] pub grammar);

#[derive(ClapParser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(help = "The script file to run", index = 1)]
    file: Option<PathBuf>,

    #[clap(short = 'i', long = "interpreted", help = "Run with the interpreter")]
    interpreted: bool,

    #[clap(short = 'c', long = "compile", help = "Compile the script to bytecode")]
    compile: bool,

    #[clap(long = "disassemble", help = "Print the bytecode disassembly")]
    disassemble: bool,

    #[clap(short = 'd', long = "debug", help = "Enable debug mode")]
    debug: bool,

    #[clap(long = "debugger", help = "Enable interactive debugger")]
    debugger: bool,

    #[clap(short = 'v', long = "verbose", help = "Enable verbose mode")]
    verbose: bool,

    #[clap(short = 't', long = "trace", help = "Enable trace mode")]
    trace: bool,

    #[clap(
        long = "jit-stats",
        help = "Show JIT profiling statistics after execution"
    )]
    jit_stats: bool,

    #[clap(
        long = "no-jit-profile",
        help = "Disable JIT profiling (for benchmarking baseline)"
    )]
    no_jit_profile: bool,

    #[clap(
        long = "jit",
        help = "Enable JIT compilation of hot code (requires 'jit' feature)"
    )]
    enable_jit: bool,
}

type WalrusResult<T> = Result<T, WalrusError>;

fn main() {
    panic::set_hook(Box::new(|err| {
        eprintln!(
            "{}",
            WalrusError::UnknownError {
                message: err.to_string(),
            }
        );

        trace!("Backtrace -> {}", backtrace::Backtrace::capture());
    }));

    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    if let Err(err) = try_main() {
        eprintln!();
        eprintln!("[ERROR] Fatal exception during execution -> {}", err);
        eprintln!();
    }
}

fn try_main() -> WalrusResult<()> {
    let args = Args::parse();

    setup_logger(if args.trace {
        LevelFilter::Trace
    } else if args.debug {
        LevelFilter::Debug
    } else {
        LevelFilter::Warn
    })?;

    // TODO: make this more idiomatic
    let opts = match (args.interpreted, args.disassemble) {
        (true, false) => Opts::Interpret,
        (false, true) => Opts::Disassemble,
        (false, false) => Opts::Compile,
        (true, true) => {
            return Err(WalrusError::GenericError {
                message: "Invalid combination of program arguments.".to_string(),
            });
        }
    };

    let jit_opts = JitOpts {
        show_stats: args.jit_stats,
        disable_profiling: args.no_jit_profile,
        enable_jit: args.enable_jit,
        enable_debugger: args.debugger,
    };

    create_shell(args.file, opts, jit_opts)?;
    Ok(())
}

pub fn create_shell(file: Option<PathBuf>, opts: Opts, jit_opts: JitOpts) -> InterpreterResult {
    Program::new_with_jit_opts(file, None, opts, jit_opts)?.execute()
}

fn setup_logger(level: LevelFilter) -> WalrusResult<()> {
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
