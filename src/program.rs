use crate::ast::{Node, NodeKind};
use crate::error::parser_err_mapper;
use crate::grammar::ProgramParser;
use crate::interpreter::Interpreter;
use crate::source_ref::SourceRef;
use crate::WalrusResult;
use log::debug;

pub struct Program<'a> {
    parser: ProgramParser,
    global_interpreter: Interpreter<'a>,
}

impl<'a> Program<'a> {
    pub fn new(filename: &'a str, source: &'a str) -> Self {
        Self {
            parser: ProgramParser::new(),
            global_interpreter: Interpreter::new(SourceRef::new(filename, source), false),
        }
    }

    pub fn run(&mut self) -> WalrusResult {
        let filename = self.global_interpreter.source_ref().filename();
        let src = self.global_interpreter.source_ref().source();

        debug!("Read {} bytes from '{}'", &src.len(), filename);

        let ast = *self
            .parser
            .parse(src)
            .map_err(|err| parser_err_mapper(err, filename, src))?;

        debug!("AST > {:?}", self.collect_ast(ast.kind()));

        let res = self.global_interpreter.interpret(ast)?;
        debug!("Interpreted > {:?}", res);

        return Ok(());
    }

    fn collect_ast(&'a self, ast: &'a NodeKind) -> Vec<&Box<Node>> {
        let mut nodes = vec![];

        if let NodeKind::Statements(stmts) = ast {
            for stmt in stmts {
                nodes.push(stmt);
            }
        }

        nodes
    }
}
