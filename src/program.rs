use crate::ast::NodeKind;
use crate::error::parser_err_mapper;
use crate::grammar::{ProgramParser, ReplParser};
use crate::interpreter::Interpreter;
use crate::source_ref::SourceRef;
use crate::WalrusResult;
use log::debug;
use std::io::{stdout, BufRead, Write};

pub struct Program<'a> {
    source_ref: SourceRef<'a>,
}

impl<'a> Program<'a> {
    pub fn new(src: &'a str, filename: &'a str) -> Self {
        Self {
            source_ref: SourceRef::new(src, filename),
        }
    }

    pub fn run(&mut self) -> WalrusResult {
        let src = self.source_ref.source();
        let filename = self.source_ref.filename();

        debug!("Read {} bytes from '{}'", src.len(), filename);

        let ast = *ProgramParser::new()
            .parse(self.source_ref.source())
            .map_err(|err| {
                parser_err_mapper(err, self.source_ref.source(), self.source_ref.filename())
            })?;

        debug!("AST > {:?}", collect_ast(ast.kind()));

        let mut interpreter = Interpreter::new(self.source_ref);
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
        let interpreter = Interpreter::new(SourceRef::new("", "REPL"));

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

            debug!("AST > {:?}", collect_ast(ast.kind()));

            self.interpreter.set_source_ref(SourceRef::new("", "REPL")); // fixme: no source for REPL errors

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
        stdout().flush().unwrap();
    }
}

fn collect_ast(ast: &NodeKind) -> Vec<&NodeKind> {
    let mut nodes = vec![];

    if let NodeKind::Statements(stmts) = ast {
        for stmt in stmts {
            nodes.push(stmt.kind());
        }
    }

    nodes
}
