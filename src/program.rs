use crate::ast::Node;
use crate::error::err_mapper;
use crate::grammar::ProgramParser;
use crate::interpreter::Interpreter;
use crate::source_ref::SourceRef;
use crate::GlassResult;
use get_size::GetSize;
use log::debug;

pub struct Program<'a> {
    parser: ProgramParser,
    global_interpreter: Interpreter<'a>,
}

impl<'a> Program<'a> {
    pub fn new(filename: &'a str, source: &'a str) -> Self {
        Self {
            parser: ProgramParser::new(),
            global_interpreter: Interpreter::new(SourceRef::new(filename, source)),
        }
    }

    pub fn run(&mut self) -> GlassResult {
        let filename = self.global_interpreter.source_ref.filename;
        let src = self.global_interpreter.source_ref.source;

        debug!("Read {} bytes from '{}'", &src.len(), filename);

        let ast = *self
            .parser
            .parse(src)
            .map_err(|err| err_mapper(err, filename, src))?;

        debug!("AST size > {:.2} KB", ast.get_heap_size() as f64 / 1024.0);

        if let Node::Statement(stmts) = &ast {
            for stmt in stmts {
                debug!("Statement > {:?}", stmt);
            }
        }

        let res = self.global_interpreter.interpret(ast)?;
        debug!("Interpreted > {:?}", res);

        return Ok(());
    }
}
