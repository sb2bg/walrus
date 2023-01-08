use crate::error::parser_err_mapper;
use crate::grammar::{ProgramParser, ReplParser};
use crate::interpreter::Interpreter;
use crate::WalrusResult;
use log::debug;
use std::io::{stdout, BufRead, Write};

pub struct Program {
    src: String,
    filename: String,
}

impl Program {
    pub fn new(src: String, filename: String) -> Self {
        Self { src, filename }
    }

    pub fn run(&mut self) -> WalrusResult {
        debug!("Read {} bytes from '{}'", self.src.len(), self.filename);

        let ast = *ProgramParser::new()
            .parse(&self.src)
            .map_err(|err| parser_err_mapper(err, &self.src, &self.filename))?;

        debug!("AST > {:?}", ast);

        let mut interpreter = Interpreter::new(&self.src, &self.filename);
        let result = interpreter.interpret(ast)?;
        debug!("Result > {:?}", result);

        return Ok(());
    }
}

pub struct Repl<'a> {
    interpreter: Interpreter<'a>,
}

impl Repl<'_> {
    pub fn new() -> Self {
        let interpreter = Interpreter::new("", "REPL");

        Self { interpreter }
    }

    pub fn run(&mut self) -> WalrusResult {
        let parser = ReplParser::new();
        Self::prompt_and_flush();

        for line in std::io::stdin().lock().lines() {
            let line = line.expect("Failed to read line");
            debug!("Read {} bytes from REPL", line.len());

            let ast = *parser
                .parse(&line)
                .map_err(|err| parser_err_mapper(err, &line, "REPL"))?;

            debug!("AST > {:?}", ast);

            self.interpreter.set_source_ref(""); // fixme: set source for REPL errors

            let result = self.interpreter.interpret(ast)?;
            debug!("{:?}", result);

            if log::log_enabled!(log::Level::Debug) {
                self.interpreter.dump();
            }

            Self::prompt_and_flush();
        }

        Ok(())
    }

    fn prompt_and_flush() {
        print!("REPL > ");
        stdout().flush().unwrap(); // fixme: handle error even though it's unlikely to happen
    }
}
