use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, stdout, Write};
use std::path::PathBuf;

use log::debug;

use crate::ast::Node;
use crate::error::{parser_err_mapper, WalrusError};
use crate::grammar::ProgramParser;
use crate::interpreter::{Interpreter, InterpreterResult};
use crate::source_ref::{OwnedSourceRef, SourceRef};
use crate::value::Value;
use crate::vm::compiler::BytecodeEmitter;
use crate::vm::VM;

#[derive(Debug, Clone, Copy)]
pub enum Opts {
    Compile,
    Interpret,
    Disassemble,
}

pub struct Program {
    source_ref: Option<OwnedSourceRef>,
    parser: ProgramParser,
    opts: Opts,
    loaded_modules: HashSet<String>,
}

impl Program {
    pub fn new(
        file: Option<PathBuf>,
        parser: Option<ProgramParser>,
        opts: Opts,
    ) -> Result<Self, WalrusError> {
        let source_ref = match file {
            Some(file) => Some(get_source(file)?),
            None => None,
        };

        Ok(Self {
            source_ref,
            parser: parser.unwrap_or_else(ProgramParser::new),
            loaded_modules: HashSet::new(),
            opts,
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

        let result = match self.opts {
            Opts::Compile => self.compile(ast, source_ref),
            Opts::Interpret => self.interpret(ast, source_ref),
            Opts::Disassemble => self.disassemble(ast, source_ref),
        }?;

        Ok(result)
    }

    fn compile(&self, ast: Node, source_ref: SourceRef) -> InterpreterResult {
        let mut emitter = BytecodeEmitter::new(source_ref);
        emitter.emit(ast)?;
        let mut vm = VM::new(source_ref, emitter.instruction_set());
        vm.run()?;

        Ok(Value::Void)
    }

    fn interpret(&self, ast: Node, source_ref: SourceRef) -> InterpreterResult {
        let mut interpreter = Interpreter::new(source_ref, self);
        interpreter.interpret(ast)
    }

    fn disassemble(&self, ast: Node, source_ref: SourceRef) -> InterpreterResult {
        let mut emitter = BytecodeEmitter::new(source_ref);
        emitter.emit(ast)?;

        println!("{}", emitter.instruction_set());

        Ok(Value::Void)
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

    pub fn load_module(&self, module_name: &str) -> InterpreterResult {
        if self.loaded_modules.contains(module_name) {
            return Ok(Value::Void);
        }

        let mut program = Program::new(Some(PathBuf::from(module_name)), None, self.opts)?;
        let result = program.execute()?;

        // self.loaded_modules.insert(module_name.to_string());

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
