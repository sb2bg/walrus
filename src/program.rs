use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, Write, stdout};
use std::path::{Path, PathBuf};

use lalrpop_util::ParseError;
use log::debug;

use crate::WalrusResult;
use crate::arenas::{DictKey, HeapValue, with_arena, with_arena_mut};
use crate::ast::{Node, NodeKind};
use crate::error::{WalrusError, parser_err_mapper, preprocess_fstrings_for_lexer};
use crate::grammar::ProgramParser;
use crate::package;
use crate::source_ref::{OwnedSourceRef, SourceRef};
use crate::span::Span;
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

enum ResolvedImport {
    Std(String),
    File(PathBuf),
}

struct PreludeInjection {
    ast: Node,
    inserted_core: bool,
}

struct ReplState {
    module_key: DictKey,
    global_names: Vec<String>,
}

enum ReplParseOutcome {
    Complete(Node),
    Incomplete,
}

pub type ProgramResult = WalrusResult<Value>;

const REPL_FILENAME: &str = "<repl>";
const REPL_PROMPT: &str = "> ";
const REPL_CONTINUATION_PROMPT: &str = "... ";

impl Program {
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

    pub fn execute(&mut self) -> ProgramResult {
        let source_ref = match &self.source_ref {
            Some(source_ref) => SourceRef::from(source_ref),
            None => return self.execute_repl(),
        };

        debug!(
            "Read {} bytes from '{}'",
            source_ref.source().len(),
            source_ref.filename()
        );

        let parse_source = preprocess_fstrings_for_lexer(source_ref.source());
        let ast = self
            .parser
            .parse(&parse_source)
            .map_err(|err| parser_err_mapper(err, source_ref.source(), source_ref.filename()))?;
        let PreludeInjection { ast, .. } = inject_core_prelude(ast);

        let result = match self.opts {
            Opts::Compile => self.compile(ast, source_ref),
            Opts::Disassemble => self.disassemble(ast, source_ref),
        }?;

        Ok(result)
    }

    pub fn execute_as_module(&mut self) -> ProgramResult {
        let source_ref = match &self.source_ref {
            Some(source_ref) => SourceRef::from(source_ref),
            None => {
                return Err(WalrusError::GenericError {
                    message: "Cannot load a module without a backing file".to_string(),
                });
            }
        };

        let parse_source = preprocess_fstrings_for_lexer(source_ref.source());
        let ast = self
            .parser
            .parse(&parse_source)
            .map_err(|err| parser_err_mapper(err, source_ref.source(), source_ref.filename()))?;
        let PreludeInjection { ast, inserted_core } = inject_core_prelude(ast);

        self.compile_module(ast, source_ref, inserted_core)
    }

    fn compile(&self, ast: Node, source_ref: SourceRef) -> ProgramResult {
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

    fn compile_repl(
        &self,
        ast: Node,
        source_ref: SourceRef,
        repl_state: &mut ReplState,
    ) -> ProgramResult {
        let (ast, echo_result) = prepare_repl_ast(ast);
        let span = *ast.span();
        let mut emitter = BytecodeEmitter::new_with_globals(source_ref, &repl_state.global_names);
        emitter.emit(ast)?;

        if echo_result {
            emitter.emit_return(span);
        } else {
            emitter.emit_void(span);
            emitter.emit_return(span);
        }

        if self.jit_opts.enable_debugger {
            emitter.build_debug_info();
        }

        let instruction_set = emitter.instruction_set();
        let mut vm =
            VM::new_with_module_binding(source_ref, instruction_set, repl_state.module_key)?;

        #[cfg(feature = "jit")]
        let jit_requested = self.jit_opts.enable_jit;
        #[cfg(not(feature = "jit"))]
        let jit_requested = false;

        let profiling_enabled =
            !self.jit_opts.disable_profiling && (jit_requested || self.jit_opts.show_stats);
        vm.set_profiling(profiling_enabled);

        if self.jit_opts.enable_debugger {
            vm.enable_debugger();
        }

        #[cfg(feature = "jit")]
        vm.set_jit_enabled(self.jit_opts.enable_jit);

        let result = vm.run();
        repl_state.refresh_global_names()?;
        let result = result?;

        if self.jit_opts.show_stats {
            let stats = vm.hotspot_stats();
            eprintln!("\n{}", stats);

            let profile = vm.type_profile();
            if !profile.is_empty() {
                eprintln!("Type Profile: {} locations observed", profile.len());
            }

            #[cfg(feature = "jit")]
            if let Some(jit_stats) = vm.jit_stats() {
                eprintln!("{}", jit_stats);
            }
        }

        Ok(result)
    }

    fn disassemble(&self, ast: Node, source_ref: SourceRef) -> ProgramResult {
        let mut emitter = BytecodeEmitter::new(source_ref);
        emitter.emit(ast)?;

        println!("{}", emitter.instruction_set());

        Ok(Value::Void)
    }

    fn disassemble_repl(
        &self,
        ast: Node,
        source_ref: SourceRef,
        repl_state: &ReplState,
    ) -> ProgramResult {
        let (ast, echo_result) = prepare_repl_ast(ast);
        let span = *ast.span();
        let mut emitter = BytecodeEmitter::new_with_globals(source_ref, &repl_state.global_names);
        emitter.emit(ast)?;

        if echo_result {
            emitter.emit_return(span);
        } else {
            emitter.emit_void(span);
            emitter.emit_return(span);
        }

        println!("{}", emitter.instruction_set());
        Ok(Value::Void)
    }

    fn compile_module(
        &self,
        ast: Node,
        source_ref: SourceRef,
        strip_implicit_core_export: bool,
    ) -> ProgramResult {
        let span = *ast.span();
        let mut emitter = BytecodeEmitter::new(source_ref);
        emitter.emit(ast)?;
        emitter.emit_void(span);
        emitter.emit_return(span);

        let mut vm = VM::new(source_ref, emitter.instruction_set());

        #[cfg(feature = "jit")]
        vm.set_jit_enabled(self.jit_opts.enable_jit);

        let _ = vm.run()?;
        let module = vm.export_globals_as_module()?;
        strip_implicit_core_export_from_module(module, strip_implicit_core_export)
    }

    fn execute_repl(&mut self) -> ProgramResult {
        let stdin = std::io::stdin();
        let mut handle = stdin.lock();
        let mut input = String::new();
        let mut buffer = String::new();
        let mut repl_state = ReplState::new()?;

        loop {
            prompt(!buffer.is_empty())?;

            input.clear();
            let read = handle
                .read_line(&mut input)
                .map_err(|source| WalrusError::IOError { source })?;

            if read == 0 {
                if !buffer.trim().is_empty() {
                    eprintln!(
                        "{}",
                        WalrusError::UnexpectedEndOfInput {
                            filename: REPL_FILENAME.to_string(),
                        }
                    );
                }
                return Ok(Value::Void);
            }

            if buffer.is_empty() {
                match input.trim() {
                    "" => continue,
                    ".exit" | ".quit" | ":exit" | ":quit" => return Ok(Value::Void),
                    _ => {}
                }
            }

            buffer.push_str(&input);

            let parse_outcome = match self.parse_repl_input(&buffer) {
                Ok(outcome) => outcome,
                Err(err) => {
                    eprintln!("{err}");
                    buffer.clear();
                    continue;
                }
            };

            let ReplParseOutcome::Complete(ast) = parse_outcome else {
                continue;
            };

            let PreludeInjection { ast, .. } = inject_core_prelude(ast);
            let source_ref = SourceRef::new(&buffer, REPL_FILENAME);
            let result = match self.opts {
                Opts::Compile => self.compile_repl(ast, source_ref, &mut repl_state),
                Opts::Disassemble => self.disassemble_repl(ast, source_ref, &repl_state),
            };

            match result {
                Ok(value) if matches!(self.opts, Opts::Compile) && value != Value::Void => {
                    println!("{}", value.stringify()?);
                }
                Ok(_) => {}
                Err(err) => eprintln!("{err}"),
            }

            buffer.clear();
        }
    }

    fn parse_repl_input(&self, input: &str) -> Result<ReplParseOutcome, WalrusError> {
        let parse_source = preprocess_fstrings_for_lexer(input);
        match self.parser.parse(&parse_source) {
            Ok(ast) => Ok(ReplParseOutcome::Complete(ast)),
            Err(err) if parser_err_is_incomplete(&err) => Ok(ReplParseOutcome::Incomplete),
            Err(err) => Err(parser_err_mapper(err, input, REPL_FILENAME)),
        }
    }
}

pub fn load_module_for_vm(module_name: &str, importer_filename: &str) -> ProgramResult {
    load_module(module_name, importer_filename, JitOpts::default())
}

pub fn cached_module_roots() -> Vec<Value> {
    MODULE_CACHE.with(|cache| cache.borrow().values().copied().collect())
}

fn load_module(module_name: &str, importer_filename: &str, jit_opts: JitOpts) -> ProgramResult {
    let resolved = resolve_import(module_name, importer_filename)?;
    let cache_key = module_cache_name(&resolved);

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
            let mut program =
                Program::new_with_jit_opts(Some(path), None, Opts::Compile, jit_opts)?;
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

fn strip_implicit_core_export_from_module(
    module: Value,
    strip_implicit_core_export: bool,
) -> ProgramResult {
    if !strip_implicit_core_export {
        return Ok(module);
    }

    let Value::Module(module_key) = module else {
        return Ok(module);
    };

    with_arena_mut(|arena| -> Result<(), WalrusError> {
        let core_key = arena.push(HeapValue::String("core"));
        let module_dict = arena.get_mut_module(module_key)?;
        module_dict.remove(&core_key);
        Ok(())
    })?;

    Ok(module)
}

fn inject_core_prelude(ast: Node) -> PreludeInjection {
    fn has_core_import(nodes: &[Node]) -> bool {
        nodes.iter().any(|node| {
            matches!(
                node.kind(),
                NodeKind::ModuleImport(module, alias)
                    if module == "std/core"
                        && alias
                            .as_deref()
                            .map_or(true, |name| name == "core")
            )
        })
    }

    fn references_core(node: &Node) -> bool {
        match node.kind() {
            NodeKind::Program(nodes)
            | NodeKind::Statements(nodes)
            | NodeKind::UnscopedStatements(nodes)
            | NodeKind::List(nodes)
            | NodeKind::Block(nodes)
            | NodeKind::StructDefinition(_, nodes) => nodes.iter().any(references_core),
            NodeKind::FString(parts) => parts.iter().any(|part| match part {
                crate::ast::FStringPart::Literal(_) => false,
                crate::ast::FStringPart::Expr(expr) => references_core(expr),
            }),
            NodeKind::Dict(entries) => entries
                .iter()
                .any(|(key, value)| references_core(key) || references_core(value)),
            NodeKind::BinOp(left, _, right)
            | NodeKind::Index(left, right)
            | NodeKind::Range(Some(left), right)
            | NodeKind::Try(left, _, right) => references_core(left) || references_core(right),
            NodeKind::UnaryOp(_, expr)
            | NodeKind::Assign(_, expr)
            | NodeKind::ExpressionStatement(expr)
            | NodeKind::Return(expr)
            | NodeKind::Print(expr)
            | NodeKind::Println(expr)
            | NodeKind::Throw(expr)
            | NodeKind::Free(expr)
            | NodeKind::Defer(expr)
            | NodeKind::Await(expr)
            | NodeKind::MemberAccess(expr, _) => references_core(expr),
            NodeKind::Reassign(name, expr, _) => name.value() == "core" || references_core(expr),
            NodeKind::IndexAssign(target, index, value) => {
                references_core(target) || references_core(index) || references_core(value)
            }
            NodeKind::FunctionCall(func, args) => {
                references_core(func) || args.iter().any(references_core)
            }
            NodeKind::AnonFunctionDefinition(_, body)
            | NodeKind::AsyncAnonFunctionDefinition(_, body)
            | NodeKind::FunctionDefinition(_, _, body)
            | NodeKind::AsyncFunctionDefinition(_, _, body)
            | NodeKind::StructFunctionDefinition(_, _, body) => references_core(body),
            NodeKind::While(condition, body) => references_core(condition) || references_core(body),
            NodeKind::If(condition, then_branch, else_branch) => {
                references_core(condition)
                    || references_core(then_branch)
                    || else_branch.as_deref().is_some_and(references_core)
            }
            NodeKind::Ternary(condition, when_true, when_false) => {
                references_core(condition)
                    || references_core(when_true)
                    || references_core(when_false)
            }
            NodeKind::For(_, iter, body) => references_core(iter) || references_core(body),
            NodeKind::Range(None, end) => references_core(end),
            NodeKind::Ident(name) => name == "core",
            NodeKind::PackageImport(_, Some(name)) | NodeKind::ModuleImport(_, Some(name)) => {
                name == "core"
            }
            NodeKind::Int(_)
            | NodeKind::Float(_)
            | NodeKind::String(_)
            | NodeKind::Bool(_)
            | NodeKind::PackageImport(_, None)
            | NodeKind::ModuleImport(_, None)
            | NodeKind::ExternFunctionDefinition(_, _)
            | NodeKind::Break
            | NodeKind::Continue
            | NodeKind::Void => false,
        }
    }

    fn with_prelude(mut nodes: Vec<Node>, span: Span) -> PreludeInjection {
        if has_core_import(&nodes) || !nodes.iter().any(references_core) {
            return PreludeInjection {
                ast: Node::new(NodeKind::Program(nodes), span),
                inserted_core: false,
            };
        }

        let prelude = Node::new(
            NodeKind::ModuleImport("std/core".to_string(), Some("core".to_string())),
            span,
        );
        nodes.insert(0, prelude);
        PreludeInjection {
            ast: Node::new(NodeKind::Program(nodes), span),
            inserted_core: true,
        }
    }

    let span = *ast.span();

    match ast.into_kind() {
        NodeKind::Program(nodes) => with_prelude(nodes, span),
        NodeKind::Statements(nodes) => with_prelude(nodes, span),
        NodeKind::UnscopedStatements(nodes) => with_prelude(nodes, span),
        other => PreludeInjection {
            ast: Node::new(other, span),
            inserted_core: false,
        },
    }
}

fn get_source(file: PathBuf) -> Result<OwnedSourceRef, WalrusError> {
    let filename = file.to_string_lossy();

    let src = fs::read_to_string(&file).map_err(|_| WalrusError::FileNotFound {
        filename: filename.to_string(),
    })?;

    Ok(OwnedSourceRef::new(src, filename.to_string()))
}

impl ReplState {
    fn new() -> WalrusResult<Self> {
        let module_key = create_repl_module()?;
        Ok(Self {
            module_key,
            global_names: Vec::new(),
        })
    }

    fn refresh_global_names(&mut self) -> WalrusResult<()> {
        let mut discovered = with_arena(|arena| -> Result<Vec<String>, WalrusError> {
            let module = arena.get_module(self.module_key)?;
            let mut names = Vec::with_capacity(module.len());

            for (key, _) in module.iter() {
                let Value::String(name_key) = *key else {
                    continue;
                };
                names.push(arena.get_string(name_key)?.to_string());
            }

            Ok(names)
        })?;

        discovered.sort_unstable();

        for name in discovered {
            if !self.global_names.contains(&name) {
                self.global_names.push(name);
            }
        }

        Ok(())
    }
}

fn create_repl_module() -> WalrusResult<DictKey> {
    let module = with_arena_mut(|arena| arena.push(HeapValue::Module(Default::default())));
    let Value::Module(module_key) = module else {
        unreachable!("module allocation should return a module key");
    };
    Ok(module_key)
}

fn prepare_repl_ast(ast: Node) -> (Node, bool) {
    let span = *ast.span();

    match ast.into_kind() {
        NodeKind::Program(nodes)
        | NodeKind::Statements(nodes)
        | NodeKind::UnscopedStatements(nodes) => prepare_repl_body(nodes, span),
        other => (Node::new(other, span), false),
    }
}

fn prepare_repl_body(mut nodes: Vec<Node>, span: Span) -> (Node, bool) {
    if let Some(last) = nodes.last() {
        if let NodeKind::ExpressionStatement(expr) = last.kind() {
            let expr = (**expr).clone();
            *nodes.last_mut().expect("last node should exist") = expr;
            return (Node::new(NodeKind::Program(nodes), span), true);
        }
    }

    (Node::new(NodeKind::Program(nodes), span), false)
}

fn prompt(continuation: bool) -> Result<(), WalrusError> {
    let prompt = if continuation {
        REPL_CONTINUATION_PROMPT
    } else {
        REPL_PROMPT
    };

    print!("{prompt}");
    stdout()
        .flush()
        .map_err(|source| WalrusError::IOError { source })?;

    Ok(())
}

fn parser_err_is_incomplete<T, E>(err: &ParseError<usize, T, E>) -> bool {
    matches!(err, ParseError::UnrecognizedEOF { .. })
}
