use std::collections::HashSet;
use std::fs;
use std::io::{stdout, BufRead, Write};
use std::path::PathBuf;

use log::debug;

use crate::error::{parser_err_mapper, WalrusError};
use crate::grammar::ProgramParser;
use crate::interpreter::{Interpreter, InterpreterResult};
use crate::source_ref::{OwnedSourceRef, SourceRef};
use crate::value::ValueKind;
use crate::vm::compiler::BytecodeEmitter;
use crate::vm::opcode::Opcode;
use crate::vm::VM;

pub struct Program {
    source_ref: Option<OwnedSourceRef>,
    parser: ProgramParser,
    interpreted: bool,
    loaded_modules: HashSet<String>,
}

impl Program {
    pub fn new(
        file: Option<PathBuf>,
        parser: Option<ProgramParser>,
        interpreted: bool,
    ) -> Result<Self, WalrusError> {
        let source_ref = match file {
            Some(file) => Some(get_source(file)?),
            None => None,
        };

        Ok(Self {
            source_ref,
            parser: parser.unwrap_or_else(ProgramParser::new),
            loaded_modules: HashSet::new(),
            interpreted,
        })
    }

    pub fn execute(&mut self) -> InterpreterResult {
        let source_ref = match &self.source_ref {
            Some(source_ref) => SourceRef::from(source_ref),
            None => return self.execute_repl(),
        };

        debug!(
            "Read {} bytes from '{}'",
            source_ref.source().len(),
            source_ref.filename()
        );

        let ast = self
            .parser
            .parse(source_ref.source())
            .map_err(|err| parser_err_mapper(err, source_ref.source(), source_ref.filename()))?;

        let result = if self.interpreted {
            let mut interpreter = Interpreter::new(source_ref, self);
            interpreter.interpret(ast)?
        } else {
            debug!("Size of Opcode = {}", std::mem::size_of::<Opcode>());

            let mut emitter = BytecodeEmitter::new();
            emitter.emit(ast)?;
            let mut vm = VM::new(source_ref, emitter.instruction_set());
            vm.run()?
        };

        Ok(result)
    }

    const REPL_FILENAME: &'static str = "<repl>";

    // todo: newline support and other advanced repl features
    fn execute_repl(&mut self) -> InterpreterResult {
        let mut input = String::new();
        // todo: find a way to pass source and update it
        let mut interpreter = Interpreter::new(SourceRef::new("", Self::REPL_FILENAME), self);

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
                .map_err(|err| parser_err_mapper(err, &input, Self::REPL_FILENAME))?;

            debug!("AST > {:?}", ast);

            let result = interpreter.interpret(ast)?;
            debug!("Result > {:?}", result);

            input.clear();
        }
    }

    pub fn load_module(&mut self, module_name: &str) -> InterpreterResult {
        if self.loaded_modules.contains(module_name) {
            return Ok(ValueKind::Void);
        }

        // fixme: don't pass None
        let mut program = Program::new(Some(PathBuf::from(module_name)), None, self.interpreted)?;
        let result = program.execute()?;

        self.loaded_modules.insert(module_name.to_string());

        Ok(result)
    }
}

fn get_source(file: PathBuf) -> Result<OwnedSourceRef, WalrusError> {
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
