use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, Write, stdout};
use std::path::{Path, PathBuf};

use log::debug;

use crate::ast::Node;
use crate::error::{WalrusError, parser_err_mapper};
use crate::grammar::ProgramParser;
use crate::interpreter::{Interpreter, InterpreterResult};
use crate::package;
use crate::source_ref::{OwnedSourceRef, SourceRef};
use crate::value::Value;
use crate::vm::VM;
use crate::vm::compiler::BytecodeEmitter;

// Thread-local set to track modules currently being loaded (for circular import detection)
thread_local! {
    static LOADING_MODULES: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
    static MODULE_CACHE: RefCell<HashMap<String, Value>> = RefCell::new(HashMap::new());
}

#[derive(Debug, Clone, Copy)]
pub enum Opts {
    Compile,
    Interpret,
    Disassemble,
}

/// Options for JIT profiling and runtime configuration
#[derive(Debug, Clone, Copy, Default)]
pub struct JitOpts {
    /// Whether to show JIT profiling statistics after execution
    pub show_stats: bool,
    /// Whether to disable JIT profiling entirely
    pub disable_profiling: bool,
    /// Whether to enable JIT compilation (requires "jit" feature)
    pub enable_jit: bool,
    /// Whether to enable the interactive debugger
    pub enable_debugger: bool,
}

pub struct Program {
    source_ref: Option<OwnedSourceRef>,
    parser: ProgramParser,
    opts: Opts,
    jit_opts: JitOpts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModuleCacheMode {
    Compile,
    Interpret,
}

impl ModuleCacheMode {
    fn from_opts(opts: Opts) -> Self {
        match opts {
            Opts::Interpret => Self::Interpret,
            Opts::Compile | Opts::Disassemble => Self::Compile,
        }
    }

    fn as_cache_prefix(self) -> &'static str {
        match self {
            Self::Compile => "compile",
            Self::Interpret => "interpret",
        }
    }
}

enum ResolvedImport {
    Std(String),
    File(PathBuf),
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

    pub fn execute_as_module(&mut self) -> InterpreterResult {
        let source_ref = match &self.source_ref {
            Some(source_ref) => SourceRef::from(source_ref),
            None => {
                return Err(WalrusError::GenericError {
                    message: "Cannot load modules from REPL context".to_string(),
                });
            }
        };

        let ast = self
            .parser
            .parse(source_ref.source())
            .map_err(|err| parser_err_mapper(err, source_ref.source(), source_ref.filename()))?;

        match self.opts {
            Opts::Interpret => self.interpret_module(ast, source_ref),
            Opts::Compile | Opts::Disassemble => self.compile_module(ast, source_ref),
        }
    }

    fn compile(&self, ast: Node, source_ref: SourceRef) -> InterpreterResult {
        let span = *ast.span();
        let mut emitter = BytecodeEmitter::new(source_ref);
        emitter.emit(ast)?;

        // Add implicit return at the end of the program
        emitter.emit_void(span);
        emitter.emit_return(span);

        // Build debug info if debugger is enabled
        if self.jit_opts.enable_debugger {
            emitter.build_debug_info();
        }

        let mut vm = VM::new(source_ref, emitter.instruction_set());

        // Profiling is only useful when JIT is enabled (with jit feature) or stats are requested.
        #[cfg(feature = "jit")]
        let jit_requested = self.jit_opts.enable_jit;
        #[cfg(not(feature = "jit"))]
        let jit_requested = false;

        let profiling_enabled =
            !self.jit_opts.disable_profiling && (jit_requested || self.jit_opts.show_stats);
        vm.set_profiling(profiling_enabled);

        // Enable debugger if requested
        if self.jit_opts.enable_debugger {
            vm.enable_debugger();
        }

        // JIT is opt-in from the CLI.
        #[cfg(feature = "jit")]
        vm.set_jit_enabled(self.jit_opts.enable_jit);

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

    fn compile_module(&self, ast: Node, source_ref: SourceRef) -> InterpreterResult {
        let span = *ast.span();
        let mut emitter = BytecodeEmitter::new(source_ref);
        emitter.emit(ast)?;
        emitter.emit_void(span);
        emitter.emit_return(span);

        let mut vm = VM::new(source_ref, emitter.instruction_set());

        #[cfg(feature = "jit")]
        vm.set_jit_enabled(self.jit_opts.enable_jit);

        let _ = vm.run()?;
        vm.export_globals_as_module()
    }

    fn interpret_module(&self, ast: Node, source_ref: SourceRef) -> InterpreterResult {
        let mut interpreter = Interpreter::new(source_ref, self);
        interpreter.interpret_module(ast)
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

    pub fn load_module(&self, module_name: &str, importer_filename: &str) -> InterpreterResult {
        let mode = ModuleCacheMode::from_opts(self.opts);
        load_module_with_mode(module_name, importer_filename, mode, self.jit_opts)
    }
}

pub fn load_module_for_vm(module_name: &str, importer_filename: &str) -> InterpreterResult {
    load_module_with_mode(
        module_name,
        importer_filename,
        ModuleCacheMode::Compile,
        JitOpts::default(),
    )
}

pub fn cached_module_roots() -> Vec<Value> {
    MODULE_CACHE.with(|cache| cache.borrow().values().copied().collect())
}

fn load_module_with_mode(
    module_name: &str,
    importer_filename: &str,
    mode: ModuleCacheMode,
    jit_opts: JitOpts,
) -> InterpreterResult {
    let resolved = resolve_import(module_name, importer_filename)?;
    let cache_key = format!(
        "{}::{}",
        mode.as_cache_prefix(),
        module_cache_name(&resolved)
    );

    if let Some(value) = MODULE_CACHE.with(|cache| cache.borrow().get(&cache_key).copied()) {
        return Ok(value);
    }

    let is_loading = LOADING_MODULES.with(|modules| modules.borrow().contains(&cache_key));
    if is_loading {
        return Err(WalrusError::CircularImport {
            module: module_name.to_string(),
        });
    }

    LOADING_MODULES.with(|modules| {
        modules.borrow_mut().insert(cache_key.clone());
    });

    let result = match resolved {
        ResolvedImport::Std(module) => Err(WalrusError::GenericError {
            message: format!("Module '{module}' is native stdlib and only available in VM mode"),
        }),
        ResolvedImport::File(path) => {
            let opts = match mode {
                ModuleCacheMode::Compile => Opts::Compile,
                ModuleCacheMode::Interpret => Opts::Interpret,
            };

            let mut program = Program::new_with_jit_opts(Some(path), None, opts, jit_opts)?;
            program.execute_as_module()
        }
    };

    LOADING_MODULES.with(|modules| {
        modules.borrow_mut().remove(&cache_key);
    });

    if let Ok(value) = result {
        MODULE_CACHE.with(|cache| {
            cache.borrow_mut().insert(cache_key, value);
        });
        return Ok(value);
    }

    result
}

fn module_cache_name(resolved: &ResolvedImport) -> String {
    match resolved {
        ResolvedImport::Std(name) => format!("std:{name}"),
        ResolvedImport::File(path) => path.to_string_lossy().to_string(),
    }
}

fn resolve_import(
    module_name: &str,
    importer_filename: &str,
) -> Result<ResolvedImport, WalrusError> {
    if module_name.starts_with("std/") {
        return Ok(ResolvedImport::Std(module_name.to_string()));
    }

    let base_dir = import_base_dir(importer_filename);

    let mut relative_path = if let Some(package_name) = module_name.strip_prefix('@') {
        if package_name.is_empty() {
            return Err(WalrusError::GenericError {
                message: "Invalid package import: empty package name".to_string(),
            });
        }

        if let Some(main_file) = package::resolve_package_main(&base_dir, package_name)? {
            return Ok(ResolvedImport::File(main_file));
        }

        // Backward-compatible fallback when no Walrus.toml is found.
        let mut package_path = PathBuf::from(package_name);
        package_path.push("main.walrus");
        package_path
    } else {
        PathBuf::from(module_name)
    };

    if relative_path.extension().is_none() {
        relative_path.set_extension("walrus");
    }

    let candidate = if relative_path.is_absolute() {
        relative_path
    } else {
        base_dir.join(relative_path)
    };

    let canonical = candidate
        .canonicalize()
        .map_err(|_| WalrusError::FileNotFound {
            filename: candidate.to_string_lossy().to_string(),
        })?;

    Ok(ResolvedImport::File(canonical))
}

fn import_base_dir(importer_filename: &str) -> PathBuf {
    let importer = Path::new(importer_filename);

    if importer_filename.starts_with('<') {
        return std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    }

    if importer.is_absolute() {
        return importer
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
    }

    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    cwd.join(importer)
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
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
