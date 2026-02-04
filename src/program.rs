use std::cell::RefCell;
use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, Write, stdout};
use std::path::PathBuf;

use log::debug;

use crate::ast::Node;
use crate::error::{WalrusError, parser_err_mapper};
use crate::grammar::ProgramParser;
use crate::interpreter::{Interpreter, InterpreterResult};
use crate::source_ref::{OwnedSourceRef, SourceRef};
use crate::value::Value;
use crate::vm::VM;
use crate::vm::compiler::BytecodeEmitter;

// Thread-local set to track modules currently being loaded (for circular import detection)
thread_local! {
    static LOADING_MODULES: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
}

#[derive(Debug, Clone, Copy)]
pub enum Opts {
    Compile,
    Interpret,
    Disassemble,
}

/// Options for JIT profiling
#[derive(Debug, Clone, Copy, Default)]
pub struct JitOpts {
    /// Whether to show JIT profiling statistics after execution
    pub show_stats: bool,
    /// Whether to disable JIT profiling entirely
    pub disable_profiling: bool,
    /// Whether to enable JIT compilation (requires "jit" feature)
    pub enable_jit: bool,
}

pub struct Program {
    source_ref: Option<OwnedSourceRef>,
    parser: ProgramParser,
    opts: Opts,
    jit_opts: JitOpts,
    loaded_modules: HashSet<String>,
}

impl Program {
    pub fn new(
        file: Option<PathBuf>,
        parser: Option<ProgramParser>,
        opts: Opts,
    ) -> Result<Self, WalrusError> {
        Self::new_with_jit_opts(file, parser, opts, JitOpts::default())
    }

    pub fn new_with_jit_opts(
        file: Option<PathBuf>,
        parser: Option<ProgramParser>,
        opts: Opts,
        jit_opts: JitOpts,
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
            jit_opts,
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
        let span = *ast.span();
        let mut emitter = BytecodeEmitter::new(source_ref);
        emitter.emit(ast)?;

        // Add implicit return at the end of the program
        emitter.emit_void(span);
        emitter.emit_return(span);

        let mut vm = if self.jit_opts.disable_profiling {
            VM::new_without_profiling(source_ref, emitter.instruction_set())
        } else {
            VM::new(source_ref, emitter.instruction_set())
        };
        
        // Enable JIT compilation if requested
        #[cfg(feature = "jit")]
        if self.jit_opts.enable_jit {
            vm.set_jit_enabled(true);
        }
        
        let result = vm.run()?;

        // Show JIT profiling stats if requested
        if self.jit_opts.show_stats {
            let stats = vm.hotspot_stats();
            eprintln!("\n{}", stats);
            
            // Show type profile summary
            let profile = vm.type_profile();
            if !profile.is_empty() {
                eprintln!("Type Profile: {} locations observed", profile.len());
            }
            
            // Show JIT compilation stats
            #[cfg(feature = "jit")]
            if let Some(jit_stats) = vm.jit_stats() {
                eprintln!("{}", jit_stats);
            }
        }

        Ok(result)
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
        // Check if module is already being loaded (circular import)
        let is_loading = LOADING_MODULES.with(|modules| {
            modules.borrow().contains(module_name)
        });

        if is_loading {
            return Err(WalrusError::GenericError {
                message: format!("Circular import detected: '{}' is already being imported", module_name),
            });
        }

        // Also check if already loaded
        if self.loaded_modules.contains(module_name) {
            return Ok(Value::Void);
        }

        // Mark as loading before we start
        LOADING_MODULES.with(|modules| {
            modules.borrow_mut().insert(module_name.to_string());
        });

        let mut program = Program::new(Some(PathBuf::from(module_name)), None, self.opts)?;
        let result = program.execute();

        // Remove from loading set when done (whether successful or not)
        LOADING_MODULES.with(|modules| {
            modules.borrow_mut().remove(module_name);
        });

        result
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
