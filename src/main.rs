mod ast;
mod error;
mod interpreter;
mod scope;
mod span;

use crate::error::{err_mapper, GlassError};
use crate::interpreter::Interpreter;
use clap::Parser as ClapParser;
use get_size::GetSize;
use lalrpop_util::lalrpop_mod;
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
    let filename = file.to_string_lossy();
    let filename = filename.as_ref();

    let src = fs::read_to_string(&file).map_err(|_| GlassError::FileNotFound {
        filename: filename.into(),
    })?;

    debug!("Read {} bytes from '{}'", &src.len(), &file.display());

    let ast = grammar::ProgramParser::new()
        .parse(&src)
        .map_err(|err| err_mapper(err, filename, &src))?;

    debug!("AST size > {:.2} KB", ast.get_heap_size() as f64 / 1024.0);
    debug!("AST > {:?}", ast);

    let interpreter = Interpreter::new();
    let result = interpreter.interpret(*ast)?;

    debug!("Interpreted > {:?}", result);

    Ok(())
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

#[cfg(test)]
mod tests {
    // TODO: Add tests
}
