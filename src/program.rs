use crate::error::{parser_err_mapper, WalrusError};
use crate::grammar::ProgramParser;
use crate::interpreter::{Interpreter, InterpreterResult};
use crate::source_ref::{OwnedSourceRef, SourceRef};
use crate::value::ValueKind;
use crate::vm::compiler::BytecodeEmitter;
use crate::vm::VM;
use log::debug;
use std::collections::HashSet;
use std::fs;
use std::io::{stdout, BufRead, Write};
use std::path::PathBuf;

pub struct Program {
    source_ref: OwnedSourceRef,
    parser: ProgramParser,
    loaded_modules: HashSet<String>,
}

impl Program {
    pub fn new(file: PathBuf, parser: Option<ProgramParser>) -> Result<Self, WalrusError> {
        Ok(Self {
            source_ref: get_source(file)?,
            parser: parser.unwrap_or_else(|| ProgramParser::new()),
            loaded_modules: HashSet::new(),
        })
    }

    pub fn execute_file(&mut self) -> InterpreterResult {
        debug!(
            "Read {} bytes from '{}'",
            self.source_ref.src.len(),
            self.source_ref.filename
        );

        let ast = self.parser.parse(&self.source_ref.src).map_err(|err| {
            parser_err_mapper(err, &self.source_ref.src, &self.source_ref.filename)
        })?;

        let mut compiler = BytecodeEmitter::new();
        compiler.emit(*ast)?;
        let instruction_set = compiler.instruction_set();

        let mut vm = VM::new(SourceRef::from(&self.source_ref), instruction_set);
        let result = vm.run()?;

        Ok(result)
    }

    pub fn execute_repl(&mut self) -> InterpreterResult {
        let mut interpreter = Interpreter::new(SourceRef::from(&self.source_ref), self);
        let mut input = String::new();
        let mut line = 1;

        loop {
            prompt()?;

            let stdin = std::io::stdin();
            let mut handle = stdin.lock();

            handle
                .read_line(&mut input)
                .map_err(|source| WalrusError::IOError { source })?;

            if input.trim().is_empty() {
                continue;
            }

            let ast = self
                .parser
                .parse(&input)
                .map_err(|err| parser_err_mapper(err, &input, &self.source_ref.filename))?;

            debug!("AST > {:?}", ast);

            let result = interpreter.interpret(*ast)?;
            debug!("Result > {:?}", result);

            println!("{}: {}", line, result);

            line += 1;
            input.clear();
        }
    }

    pub fn load_module(&mut self, module_name: &str) -> InterpreterResult {
        if self.loaded_modules.contains(module_name) {
            return Ok(ValueKind::Void);
        }

        // fixme: don't pass None
        let mut program = Program::new(PathBuf::from(module_name), None)?;
        let result = program.execute_file()?;

        self.loaded_modules.insert(module_name.to_string());

        Ok(result)
    }
}

fn get_source<'a>(file: PathBuf) -> Result<OwnedSourceRef, WalrusError> {
    let filename = file.to_string_lossy();

    let src = fs::read_to_string(&file).map_err(|_| WalrusError::FileNotFound {
        filename: filename.to_string(),
    })?;

    Ok(OwnedSourceRef::new(src, filename.to_string()))
}

fn prompt() -> Result<(), WalrusError> {
    print!("> ");

    stdout()
        .flush()
        .map_err(|source| WalrusError::IOError { source })?;

    Ok(())
}
