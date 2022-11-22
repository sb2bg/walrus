use crate::error::{err_mapper, GlassError};
use crate::grammar::ProgramParser;
use crate::interpreter::{Interpreter, InterpreterResult};
use get_size::GetSize;
use log::debug;

pub struct Program {
    filename: String,
    parser: ProgramParser,
    global_interpreter: Interpreter,
}

impl Program {
    pub fn new(filename: String) -> Self {
        Self {
            filename,
            parser: ProgramParser::new(),
            global_interpreter: Interpreter::new(),
        }
    }

    pub fn run<'a, T: Fn() -> Result<String, GlassError>>(
        &self,
        src_feed: T,
        repl: bool,
    ) -> InterpreterResult {
        loop {
            let src = src_feed()?;
            debug!("Read {} bytes from '{}'", &src.len(), self.filename);

            let ast = self
                .parser
                .parse(&src)
                .map_err(|err| err_mapper(err, &self.filename, &src))?;

            debug!("AST size > {:.2} KB", ast.get_heap_size() as f64 / 1024.0);
            debug!("AST > {:?}", ast);

            let res = self.global_interpreter.interpret(*ast)?;
            debug!("Interpreted > {:?}", res);

            if !repl {
                break;
            }
        }

        Ok(())
    }
}
