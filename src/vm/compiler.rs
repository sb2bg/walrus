use crate::WalrusResult;
use crate::arenas::HeapValue;
use crate::ast::FStringPart;
use crate::ast::{Node, NodeKind};
use crate::error::WalrusError;
use crate::function::{VmFunction, WalrusFunction};
use crate::source_ref::SourceRef;
use crate::span::Span;
use crate::value::Value;
use crate::vm::instruction_set::InstructionSet;
use crate::vm::opcode::{Instruction, Opcode};
use crate::vm::optimize;
use rustc_hash::FxHashSet;

/// BytecodeEmitter compiles AST nodes into VM bytecode.
///
/// # Architecture Notes
///
/// ## Scope and Locals
/// The compiler tracks variable scopes using a depth counter and a symbol table.
/// Local variables are indexed by their position in the `locals` vector, which
/// is used by the VM at runtime. When a scope ends, `PopLocal` is emitted to
/// clean up the locals vector.
///
/// ## Function Compilation
/// Functions are compiled into separate `InstructionSet`s with their own symbol
/// tables. The VM creates a child VM for each function call, which provides
/// isolation but is less efficient than a proper call frame stack.
///
/// ## Closures (LIMITATION)
/// Currently, nested functions cannot capture variables from enclosing function
/// scopes. Only global variables are accessible within nested functions.
/// Implementing closures would require:
/// 1. Tracking "upvalues" - variables captured from enclosing scopes
/// 2. Adding LoadUpvalue/StoreUpvalue opcodes  
/// 3. Storing captured variables with the function object
///
/// ## Future Improvements
/// - Implement proper call frames with a frame pointer instead of child VMs
/// - Add closure/upvalue support for nested function variable capture
/// - Consider using Rc<InstructionSet> to avoid cloning on function calls
pub struct BytecodeEmitter<'a> {
    instructions: InstructionSet,
    source_ref: SourceRef<'a>,
    depth: usize,                   // 0 = global scope, >0 = local scope
    loop_stack: Vec<LoopContext>,   // Track nested loops for break/continue
    current_struct: Option<String>, // Name of struct currently being compiled (for method access)
    current_struct_methods: Option<FxHashSet<String>>, // Known method names for current struct
    current_function_name: Option<String>,
    current_function_arity: Option<usize>,
    top_level_global_names: FxHashSet<String>,
    known_int_globals: FxHashSet<String>,
    known_int_functions: FxHashSet<String>,
    known_pure_int_functions: FxHashSet<String>,
    known_pure_cloneable_functions: FxHashSet<String>,
    known_int_scopes: Vec<FxHashSet<String>>,
}

struct LoopContext {
    start: usize,             // Address of loop start (for continue)
    breaks: Vec<usize>,       // Addresses of break jumps to patch
    has_stack_iterator: bool, // True if loop has an iterator on the operand stack (iterator-based for loops only)
    locals_at_start: usize,   // Number of locals at loop body start (for continue cleanup)
}

impl<'a> BytecodeEmitter<'a> {
    #[inline]
    fn local_load_opcode(index: u32) -> Opcode {
        match index {
            0 => Opcode::LoadLocal0,
            1 => Opcode::LoadLocal1,
            2 => Opcode::LoadLocal2,
            3 => Opcode::LoadLocal3,
            4 => Opcode::LoadLocal4,
            5 => Opcode::LoadLocal5,
            6 => Opcode::LoadLocal6,
            7 => Opcode::LoadLocal7,
            8 => Opcode::LoadLocal8,
            9 => Opcode::LoadLocal9,
            10 => Opcode::LoadLocal10,
            11 => Opcode::LoadLocal11,
            _ => Opcode::Load(index),
        }
    }

    #[inline]
    fn local_store_opcode(index: u32) -> Opcode {
        match index {
            0 => Opcode::StoreLocal0,
            1 => Opcode::StoreLocal1,
            2 => Opcode::StoreLocal2,
            3 => Opcode::StoreLocal3,
            4 => Opcode::StoreLocal4,
            5 => Opcode::StoreLocal5,
            6 => Opcode::StoreLocal6,
            7 => Opcode::StoreLocal7,
            8 => Opcode::StoreLocal8,
            9 => Opcode::StoreLocal9,
            10 => Opcode::StoreLocal10,
            11 => Opcode::StoreLocal11,
            _ => Opcode::StoreAt(index),
        }
    }

    #[inline]
    fn local_reassign_opcode(index: u32) -> Opcode {
        match index {
            0 => Opcode::ReassignLocal0,
            1 => Opcode::ReassignLocal1,
            2 => Opcode::ReassignLocal2,
            3 => Opcode::ReassignLocal3,
            4 => Opcode::ReassignLocal4,
            5 => Opcode::ReassignLocal5,
            6 => Opcode::ReassignLocal6,
            7 => Opcode::ReassignLocal7,
            8 => Opcode::ReassignLocal8,
            9 => Opcode::ReassignLocal9,
            10 => Opcode::ReassignLocal10,
            11 => Opcode::ReassignLocal11,
            _ => Opcode::Reassign(index),
        }
    }

    #[inline]
    fn global_load_opcode(index: u32) -> Opcode {
        match index {
            0 => Opcode::LoadGlobal0,
            1 => Opcode::LoadGlobal1,
            2 => Opcode::LoadGlobal2,
            3 => Opcode::LoadGlobal3,
            _ => Opcode::LoadGlobal(index),
        }
    }

    #[inline]
    fn add_assign_opcode(index: u32, is_global: bool, int_specialized: bool) -> Opcode {
        if is_global {
            if int_specialized {
                Opcode::AddAssignGlobalInt(index)
            } else {
                Opcode::AddAssignGlobal(index)
            }
        } else {
            if int_specialized {
                Opcode::AddAssignLocalInt(index)
            } else {
                Opcode::AddAssignLocal(index)
            }
        }
    }

    pub fn new(source_ref: SourceRef<'a>) -> Self {
        Self {
            instructions: InstructionSet::new(),
            source_ref,
            depth: 0, // Start at global scope
            loop_stack: Vec::new(),
            current_struct: None,
            current_struct_methods: None,
            current_function_name: None,
            current_function_arity: None,
            top_level_global_names: FxHashSet::default(),
            known_int_globals: FxHashSet::default(),
            known_int_functions: FxHashSet::default(),
            known_pure_int_functions: FxHashSet::default(),
            known_pure_cloneable_functions: FxHashSet::default(),
            known_int_scopes: vec![FxHashSet::default()],
        }
    }

    pub fn new_with_globals(source_ref: SourceRef<'a>, global_names: &[String]) -> Self {
        let mut emitter = Self::new(source_ref);
        for name in global_names {
            emitter.instructions.push_global(name.clone());
        }
        emitter
    }

    fn new_child(&self) -> Self {
        Self {
            instructions: InstructionSet::new_child_with_globals(self.instructions.globals.clone()),
            source_ref: self.source_ref,
            depth: 1,               // Functions start at local scope depth 1
            loop_stack: Vec::new(), // Functions get their own loop stack
            current_struct: self.current_struct.clone(),
            current_struct_methods: self.current_struct_methods.clone(),
            current_function_name: None,
            current_function_arity: None,
            top_level_global_names: self.top_level_global_names.clone(),
            known_int_globals: self.known_int_globals.clone(),
            known_int_functions: self.known_int_functions.clone(),
            known_pure_int_functions: self.known_pure_int_functions.clone(),
            known_pure_cloneable_functions: self.known_pure_cloneable_functions.clone(),
            known_int_scopes: vec![FxHashSet::default(), FxHashSet::default()],
        }
    }

    fn is_known_int_name(&self, name: &str) -> bool {
        if let Some(depth) = self.instructions.resolve_depth(name) {
            return self
                .known_int_scopes
                .get(depth)
                .is_some_and(|scope| scope.contains(name));
        }

        self.known_int_globals.contains(name)
    }

    fn is_known_int_function_call(&self, func: &Node, recursive_name: Option<&str>) -> bool {
        match func.kind() {
            NodeKind::Ident(name) => {
                self.instructions.resolve_local_index(name).is_none()
                    && (self.known_int_functions.contains(name)
                        || recursive_name.is_some_and(|current| current == name))
            }
            NodeKind::MemberAccess(object, method_name) => {
                matches!(object.kind(), NodeKind::Ident(module_name) if module_name == "core")
                    && method_name == "len"
            }
            _ => false,
        }
    }

    fn is_known_int_expr_with_recursion(&self, node: &Node, recursive_name: Option<&str>) -> bool {
        match node.kind() {
            NodeKind::Int(_) => true,
            NodeKind::Ident(name) => self.is_known_int_name(name),
            NodeKind::UnaryOp(Opcode::Negate, expr) => {
                self.is_known_int_expr_with_recursion(expr, recursive_name)
            }
            NodeKind::FunctionCall(func, _) => {
                self.is_known_int_function_call(func, recursive_name)
            }
            NodeKind::BinOp(left, op, right) => {
                matches!(
                    op,
                    Opcode::Add
                        | Opcode::Subtract
                        | Opcode::Multiply
                        | Opcode::Divide
                        | Opcode::Modulo
                ) && self.is_known_int_expr_with_recursion(left, recursive_name)
                    && self.is_known_int_expr_with_recursion(right, recursive_name)
            }
            _ => false,
        }
    }

    fn is_known_int_expr(&self, node: &Node) -> bool {
        self.is_known_int_expr_with_recursion(node, None)
    }

    fn is_known_int_expr_with_assumptions(
        &self,
        node: &Node,
        recursive_name: Option<&str>,
        assumed_int_names: &FxHashSet<String>,
    ) -> bool {
        match node.kind() {
            NodeKind::Int(_) => true,
            NodeKind::Ident(name) => {
                assumed_int_names.contains(name) || self.is_known_int_name(name)
            }
            NodeKind::UnaryOp(Opcode::Negate, expr) => {
                self.is_known_int_expr_with_assumptions(expr, recursive_name, assumed_int_names)
            }
            NodeKind::FunctionCall(func, _) => {
                self.is_known_int_function_call(func, recursive_name)
            }
            NodeKind::BinOp(left, op, right) => {
                matches!(
                    op,
                    Opcode::Add
                        | Opcode::Subtract
                        | Opcode::Multiply
                        | Opcode::Divide
                        | Opcode::Modulo
                ) && self.is_known_int_expr_with_assumptions(
                    left,
                    recursive_name,
                    assumed_int_names,
                ) && self.is_known_int_expr_with_assumptions(
                    right,
                    recursive_name,
                    assumed_int_names,
                )
            }
            _ => false,
        }
    }

    fn specialized_int_opcode(&self, op: Opcode, left: &Node, right: &Node) -> Option<Opcode> {
        if self.is_known_int_expr(left) {
            match (op, right.kind()) {
                (Opcode::Add, NodeKind::Int(1)) => return Some(Opcode::AddInt1),
                (Opcode::Subtract, NodeKind::Int(1)) => return Some(Opcode::SubtractInt1),
                (Opcode::Subtract, NodeKind::Int(2)) => return Some(Opcode::SubtractInt2),
                (Opcode::LessEqual, NodeKind::Int(1)) => return Some(Opcode::LessEqualInt1),
                _ => {}
            }
        }

        if !self.is_known_int_expr(left) || !self.is_known_int_expr(right) {
            return None;
        }

        match op {
            Opcode::Add => Some(Opcode::AddInt),
            Opcode::Subtract => Some(Opcode::SubtractInt),
            Opcode::Multiply => Some(Opcode::MultiplyInt),
            Opcode::Divide => Some(Opcode::DivideInt),
            Opcode::Modulo => Some(Opcode::ModuloInt),
            Opcode::Less => Some(Opcode::LessInt),
            Opcode::LessEqual => Some(Opcode::LessEqualInt),
            _ => None,
        }
    }

    fn mark_known_int(&mut self, name: &str, is_global: bool, known: bool) {
        if is_global {
            if known {
                self.known_int_globals.insert(name.to_string());
            } else {
                self.known_int_globals.remove(name);
            }
            return;
        }

        let Some(depth) = self.instructions.resolve_depth(name) else {
            return;
        };
        let Some(scope) = self.known_int_scopes.get_mut(depth) else {
            return;
        };
        if known {
            scope.insert(name.to_string());
        } else {
            scope.remove(name);
        }
    }

    fn name_used_in_int_context(node: &Node, name: &str) -> bool {
        match node.kind() {
            NodeKind::BinOp(left, op, right) => {
                (matches!(
                    op,
                    Opcode::Add
                        | Opcode::Subtract
                        | Opcode::Multiply
                        | Opcode::Divide
                        | Opcode::Modulo
                        | Opcode::Less
                        | Opcode::LessEqual
                        | Opcode::Greater
                        | Opcode::GreaterEqual
                ) && (Self::references_name(left, name) || Self::references_name(right, name)))
                    || Self::name_used_in_int_context(left, name)
                    || Self::name_used_in_int_context(right, name)
            }
            NodeKind::UnaryOp(Opcode::Negate, expr) => {
                Self::references_name(expr, name) || Self::name_used_in_int_context(expr, name)
            }
            NodeKind::Index(_, index) => {
                Self::references_name(index, name) || Self::name_used_in_int_context(index, name)
            }
            NodeKind::Range(Some(start), end) => {
                Self::name_used_in_int_context(start, name)
                    || Self::name_used_in_int_context(end, name)
            }
            NodeKind::Range(None, end)
            | NodeKind::Assign(_, end)
            | NodeKind::ExpressionStatement(end)
            | NodeKind::Return(end)
            | NodeKind::Print(end)
            | NodeKind::Println(end)
            | NodeKind::Throw(end)
            | NodeKind::Free(end)
            | NodeKind::Defer(end)
            | NodeKind::Await(end) => Self::name_used_in_int_context(end, name),
            NodeKind::Reassign(_, expr, _) | NodeKind::MemberAccess(expr, _) => {
                Self::name_used_in_int_context(expr, name)
            }
            NodeKind::Program(nodes)
            | NodeKind::Statements(nodes)
            | NodeKind::UnscopedStatements(nodes)
            | NodeKind::List(nodes)
            | NodeKind::Block(nodes)
            | NodeKind::StructDefinition(_, nodes) => nodes
                .iter()
                .any(|node| Self::name_used_in_int_context(node, name)),
            NodeKind::Dict(entries) => entries.iter().any(|(key, value)| {
                Self::name_used_in_int_context(key, name)
                    || Self::name_used_in_int_context(value, name)
            }),
            NodeKind::FunctionCall(func, args) => {
                Self::name_used_in_int_context(func, name)
                    || args
                        .iter()
                        .any(|arg| Self::name_used_in_int_context(arg, name))
            }
            NodeKind::If(condition, then_branch, else_branch) => {
                Self::name_used_in_int_context(condition, name)
                    || Self::name_used_in_int_context(then_branch, name)
                    || else_branch
                        .as_deref()
                        .is_some_and(|branch| Self::name_used_in_int_context(branch, name))
            }
            NodeKind::Ternary(condition, when_true, when_false) => {
                Self::name_used_in_int_context(condition, name)
                    || Self::name_used_in_int_context(when_true, name)
                    || Self::name_used_in_int_context(when_false, name)
            }
            NodeKind::While(condition, body) => {
                Self::name_used_in_int_context(condition, name)
                    || Self::name_used_in_int_context(body, name)
            }
            NodeKind::For(_, iter, body) => {
                Self::name_used_in_int_context(iter, name)
                    || Self::name_used_in_int_context(body, name)
            }
            _ => false,
        }
    }

    fn int_parameter_names(args: &[String], body: &Node) -> FxHashSet<String> {
        args.iter()
            .filter(|arg| Self::name_used_in_int_context(body, arg))
            .cloned()
            .collect()
    }

    fn extend_assumed_int_names(
        &self,
        node: &Node,
        recursive_name: &str,
        assumed_int_names: &mut FxHashSet<String>,
    ) {
        match node.kind() {
            NodeKind::Program(nodes)
            | NodeKind::Statements(nodes)
            | NodeKind::UnscopedStatements(nodes)
            | NodeKind::Block(nodes)
            | NodeKind::List(nodes)
            | NodeKind::StructDefinition(_, nodes) => {
                for node in nodes {
                    self.extend_assumed_int_names(node, recursive_name, assumed_int_names);
                }
            }
            NodeKind::Assign(name, expr) => {
                if self.is_known_int_expr_with_assumptions(
                    expr,
                    Some(recursive_name),
                    assumed_int_names,
                ) {
                    assumed_int_names.insert(name.clone());
                }
            }
            NodeKind::Reassign(name, expr, op) => {
                let current_is_int = assumed_int_names.contains(name.value());
                let expr_is_int = self.is_known_int_expr_with_assumptions(
                    expr,
                    Some(recursive_name),
                    assumed_int_names,
                );
                let next_is_int = match op {
                    Opcode::Equal => expr_is_int,
                    Opcode::Add
                    | Opcode::Subtract
                    | Opcode::Multiply
                    | Opcode::Divide
                    | Opcode::Modulo => current_is_int && expr_is_int,
                    _ => false,
                };

                if next_is_int {
                    assumed_int_names.insert(name.value().to_string());
                }
            }
            NodeKind::Dict(entries) => {
                for (key, value) in entries {
                    self.extend_assumed_int_names(key, recursive_name, assumed_int_names);
                    self.extend_assumed_int_names(value, recursive_name, assumed_int_names);
                }
            }
            NodeKind::BinOp(left, _, right)
            | NodeKind::Index(left, right)
            | NodeKind::Range(Some(left), right)
            | NodeKind::Try(left, _, right) => {
                self.extend_assumed_int_names(left, recursive_name, assumed_int_names);
                self.extend_assumed_int_names(right, recursive_name, assumed_int_names);
            }
            NodeKind::UnaryOp(_, expr)
            | NodeKind::ExpressionStatement(expr)
            | NodeKind::Return(expr)
            | NodeKind::Print(expr)
            | NodeKind::Println(expr)
            | NodeKind::Throw(expr)
            | NodeKind::Free(expr)
            | NodeKind::Defer(expr)
            | NodeKind::Await(expr)
            | NodeKind::MemberAccess(expr, _)
            | NodeKind::Range(None, expr) => {
                self.extend_assumed_int_names(expr, recursive_name, assumed_int_names);
            }
            NodeKind::FunctionCall(func, args) => {
                self.extend_assumed_int_names(func, recursive_name, assumed_int_names);
                for arg in args {
                    self.extend_assumed_int_names(arg, recursive_name, assumed_int_names);
                }
            }
            NodeKind::If(condition, then_branch, else_branch) => {
                self.extend_assumed_int_names(condition, recursive_name, assumed_int_names);
                self.extend_assumed_int_names(then_branch, recursive_name, assumed_int_names);
                if let Some(else_branch) = else_branch {
                    self.extend_assumed_int_names(else_branch, recursive_name, assumed_int_names);
                }
            }
            NodeKind::Ternary(condition, when_true, when_false) => {
                self.extend_assumed_int_names(condition, recursive_name, assumed_int_names);
                self.extend_assumed_int_names(when_true, recursive_name, assumed_int_names);
                self.extend_assumed_int_names(when_false, recursive_name, assumed_int_names);
            }
            NodeKind::While(condition, body) => {
                self.extend_assumed_int_names(condition, recursive_name, assumed_int_names);
                self.extend_assumed_int_names(body, recursive_name, assumed_int_names);
            }
            NodeKind::For(name, iter, body) => {
                if Self::name_used_in_int_context(iter, name) {
                    assumed_int_names.insert(name.clone());
                }
                self.extend_assumed_int_names(iter, recursive_name, assumed_int_names);
                self.extend_assumed_int_names(body, recursive_name, assumed_int_names);
            }
            _ => {}
        }
    }

    fn assumed_int_names_for_function(
        &self,
        recursive_name: &str,
        args: &[String],
        body: &Node,
    ) -> FxHashSet<String> {
        let mut assumed_int_names = Self::int_parameter_names(args, body);
        self.extend_assumed_int_names(body, recursive_name, &mut assumed_int_names);
        assumed_int_names
    }

    fn sequence_guarantees_int_return(
        &self,
        nodes: &[Node],
        recursive_name: &str,
        assumed_int_names: &FxHashSet<String>,
    ) -> bool {
        nodes
            .iter()
            .any(|node| self.node_guarantees_int_return(node, recursive_name, assumed_int_names))
    }

    fn node_guarantees_int_return(
        &self,
        node: &Node,
        recursive_name: &str,
        assumed_int_names: &FxHashSet<String>,
    ) -> bool {
        match node.kind() {
            NodeKind::Return(expr) => self.is_known_int_expr_with_assumptions(
                expr,
                Some(recursive_name),
                assumed_int_names,
            ),
            NodeKind::If(_, then_branch, Some(else_branch)) => {
                self.node_guarantees_int_return(then_branch, recursive_name, assumed_int_names)
                    && self.node_guarantees_int_return(
                        else_branch,
                        recursive_name,
                        assumed_int_names,
                    )
            }
            NodeKind::Program(nodes)
            | NodeKind::Statements(nodes)
            | NodeKind::UnscopedStatements(nodes)
            | NodeKind::Block(nodes) => {
                self.sequence_guarantees_int_return(nodes, recursive_name, assumed_int_names)
            }
            _ => false,
        }
    }

    fn function_guarantees_int_return(&self, name: &str, args: &[String], body: &Node) -> bool {
        let assumed_int_names = self.assumed_int_names_for_function(name, args, body);
        self.node_guarantees_int_return(body, name, &assumed_int_names)
    }

    fn top_level_binding_name(node: &Node) -> Option<String> {
        match node.kind() {
            NodeKind::Assign(name, _) => Some(name.clone()),
            NodeKind::FunctionDefinition(name, _, _)
            | NodeKind::AsyncFunctionDefinition(name, _, _)
            | NodeKind::StructDefinition(name, _) => Some(name.clone()),
            NodeKind::PackageImport(name, alias) | NodeKind::ModuleImport(name, alias) => {
                Some(alias.clone().unwrap_or_else(|| name.clone()))
            }
            _ => None,
        }
    }

    fn top_level_binding_must_be_global(node: &Node) -> bool {
        matches!(
            node.kind(),
            NodeKind::FunctionDefinition(_, _, _)
                | NodeKind::AsyncFunctionDefinition(_, _, _)
                | NodeKind::StructDefinition(_, _)
        )
    }

    fn top_level_body_references(node: &Node, name: &str) -> bool {
        match node.kind() {
            NodeKind::FunctionDefinition(_, _, body)
            | NodeKind::AsyncFunctionDefinition(_, _, body) => Self::references_name(body, name),
            NodeKind::StructDefinition(_, members) => {
                members.iter().any(|member| match member.kind() {
                    NodeKind::StructFunctionDefinition(_, _, body) => {
                        Self::references_name(body, name)
                    }
                    _ => false,
                })
            }
            _ => false,
        }
    }

    fn prepare_top_level_scope(&mut self, nodes: &[Node]) {
        if self.depth != 0 || !self.top_level_global_names.is_empty() {
            return;
        }

        let mut binding_names = FxHashSet::default();
        let mut global_names = FxHashSet::default();

        for node in nodes {
            if let Some(name) = Self::top_level_binding_name(node) {
                binding_names.insert(name.clone());
                if Self::top_level_binding_must_be_global(node) {
                    global_names.insert(name);
                }
            }
        }

        for name in &binding_names {
            if global_names.contains(name) {
                continue;
            }

            if nodes
                .iter()
                .any(|node| Self::top_level_body_references(node, name))
            {
                global_names.insert(name.clone());
            }
        }

        self.top_level_global_names = global_names;
    }

    fn node_is_pure_int_ast(
        &self,
        node: &Node,
        recursive_name: &str,
        locals: &mut FxHashSet<String>,
    ) -> bool {
        match node.kind() {
            NodeKind::Program(nodes)
            | NodeKind::Statements(nodes)
            | NodeKind::UnscopedStatements(nodes)
            | NodeKind::Block(nodes) => nodes
                .iter()
                .all(|node| self.node_is_pure_int_ast(node, recursive_name, locals)),
            NodeKind::Assign(name, expr) => {
                if !self.node_is_pure_int_ast(expr, recursive_name, locals) {
                    return false;
                }
                locals.insert(name.clone());
                true
            }
            NodeKind::Reassign(name, expr, _) => {
                locals.contains(name.value())
                    && self.node_is_pure_int_ast(expr, recursive_name, locals)
            }
            NodeKind::Return(expr)
            | NodeKind::ExpressionStatement(expr)
            | NodeKind::UnaryOp(_, expr)
            | NodeKind::Range(None, expr)
            | NodeKind::MemberAccess(expr, _) => {
                self.node_is_pure_int_ast(expr, recursive_name, locals)
            }
            NodeKind::Int(_)
            | NodeKind::Float(_)
            | NodeKind::String(_)
            | NodeKind::Bool(_)
            | NodeKind::Void => true,
            NodeKind::FString(parts) => parts.iter().all(|part| match part {
                FStringPart::Literal(_) => true,
                FStringPart::Expr(expr) => self.node_is_pure_int_ast(expr, recursive_name, locals),
            }),
            NodeKind::Ident(name) => locals.contains(name),
            NodeKind::List(nodes) => nodes
                .iter()
                .all(|node| self.node_is_pure_int_ast(node, recursive_name, locals)),
            NodeKind::Dict(entries) => entries.iter().all(|(key, value)| {
                self.node_is_pure_int_ast(key, recursive_name, locals)
                    && self.node_is_pure_int_ast(value, recursive_name, locals)
            }),
            NodeKind::BinOp(left, _, right)
            | NodeKind::Index(left, right)
            | NodeKind::Range(Some(left), right)
            | NodeKind::Try(left, _, right) => {
                self.node_is_pure_int_ast(left, recursive_name, locals)
                    && self.node_is_pure_int_ast(right, recursive_name, locals)
            }
            NodeKind::FunctionCall(func, args) => {
                let args_are_pure = args
                    .iter()
                    .all(|arg| self.node_is_pure_int_ast(arg, recursive_name, locals));
                if !args_are_pure {
                    return false;
                }

                match func.kind() {
                    NodeKind::Ident(name) => {
                        !locals.contains(name)
                            && (name == recursive_name
                                || self.known_pure_int_functions.contains(name))
                    }
                    NodeKind::MemberAccess(object, method_name) => {
                        matches!(object.kind(), NodeKind::Ident(module_name) if module_name == "core")
                            && method_name == "len"
                    }
                    _ => false,
                }
            }
            NodeKind::If(condition, then_branch, else_branch) => {
                self.node_is_pure_int_ast(condition, recursive_name, locals)
                    && self.node_is_pure_int_ast(then_branch, recursive_name, locals)
                    && else_branch.as_deref().is_none_or(|branch| {
                        self.node_is_pure_int_ast(branch, recursive_name, locals)
                    })
            }
            NodeKind::Ternary(condition, when_true, when_false) => {
                self.node_is_pure_int_ast(condition, recursive_name, locals)
                    && self.node_is_pure_int_ast(when_true, recursive_name, locals)
                    && self.node_is_pure_int_ast(when_false, recursive_name, locals)
            }
            NodeKind::While(condition, body) => {
                self.node_is_pure_int_ast(condition, recursive_name, locals)
                    && self.node_is_pure_int_ast(body, recursive_name, locals)
            }
            NodeKind::For(name, iter, body) => {
                if !self.node_is_pure_int_ast(iter, recursive_name, locals) {
                    return false;
                }
                locals.insert(name.clone());
                self.node_is_pure_int_ast(body, recursive_name, locals)
            }
            NodeKind::Print(_)
            | NodeKind::Println(_)
            | NodeKind::Throw(_)
            | NodeKind::Free(_)
            | NodeKind::Defer(_)
            | NodeKind::Await(_)
            | NodeKind::IndexAssign(_, _, _)
            | NodeKind::AnonFunctionDefinition(_, _)
            | NodeKind::AsyncAnonFunctionDefinition(_, _)
            | NodeKind::FunctionDefinition(_, _, _)
            | NodeKind::AsyncFunctionDefinition(_, _, _)
            | NodeKind::ExternFunctionDefinition(_, _)
            | NodeKind::StructDefinition(_, _)
            | NodeKind::StructFunctionDefinition(_, _, _)
            | NodeKind::PackageImport(_, _)
            | NodeKind::ModuleImport(_, _)
            | NodeKind::Break
            | NodeKind::Continue => false,
        }
    }

    fn function_is_pure_int_ast(&self, name: &str, args: &[String], body: &Node) -> bool {
        let mut locals: FxHashSet<String> = args.iter().cloned().collect();
        self.node_is_pure_int_ast(body, name, &mut locals)
    }

    fn expr_is_cloneable_ast(&self, node: &Node, recursive_name: &str) -> bool {
        match node.kind() {
            NodeKind::Int(_)
            | NodeKind::Float(_)
            | NodeKind::String(_)
            | NodeKind::Bool(_)
            | NodeKind::Void
            | NodeKind::Ident(_) => true,
            NodeKind::List(nodes) => nodes
                .iter()
                .all(|node| self.expr_is_cloneable_ast(node, recursive_name)),
            NodeKind::Dict(entries) => entries.iter().all(|(key, value)| {
                self.expr_is_cloneable_ast(key, recursive_name)
                    && self.expr_is_cloneable_ast(value, recursive_name)
            }),
            NodeKind::BinOp(left, _, right)
            | NodeKind::Index(left, right)
            | NodeKind::Range(Some(left), right)
            | NodeKind::Try(left, _, right) => {
                self.expr_is_cloneable_ast(left, recursive_name)
                    && self.expr_is_cloneable_ast(right, recursive_name)
            }
            NodeKind::UnaryOp(_, expr)
            | NodeKind::ExpressionStatement(expr)
            | NodeKind::Return(expr)
            | NodeKind::Range(None, expr) => self.expr_is_cloneable_ast(expr, recursive_name),
            NodeKind::FString(parts) => parts.iter().all(|part| match part {
                FStringPart::Literal(_) => true,
                FStringPart::Expr(expr) => self.expr_is_cloneable_ast(expr, recursive_name),
            }),
            NodeKind::FunctionCall(func, args) => {
                args.iter()
                    .all(|arg| self.expr_is_cloneable_ast(arg, recursive_name))
                    && matches!(
                        func.kind(),
                        NodeKind::Ident(name)
                            if name == recursive_name
                                || self.known_pure_cloneable_functions.contains(name)
                    )
            }
            NodeKind::If(condition, then_branch, else_branch) => {
                self.expr_is_cloneable_ast(condition, recursive_name)
                    && self.expr_is_cloneable_ast(then_branch, recursive_name)
                    && else_branch
                        .as_deref()
                        .is_none_or(|branch| self.expr_is_cloneable_ast(branch, recursive_name))
            }
            NodeKind::Ternary(condition, when_true, when_false) => {
                self.expr_is_cloneable_ast(condition, recursive_name)
                    && self.expr_is_cloneable_ast(when_true, recursive_name)
                    && self.expr_is_cloneable_ast(when_false, recursive_name)
            }
            NodeKind::Program(nodes)
            | NodeKind::Statements(nodes)
            | NodeKind::UnscopedStatements(nodes)
            | NodeKind::Block(nodes) => nodes
                .iter()
                .all(|node| self.expr_is_cloneable_ast(node, recursive_name)),
            _ => false,
        }
    }

    fn node_guarantees_cloneable_return(&self, node: &Node, recursive_name: &str) -> bool {
        match Self::statement_expr(node).kind() {
            NodeKind::Return(expr) => self.expr_is_cloneable_ast(expr, recursive_name),
            NodeKind::If(_, then_branch, Some(else_branch)) => {
                self.node_guarantees_cloneable_return(then_branch, recursive_name)
                    && self.node_guarantees_cloneable_return(else_branch, recursive_name)
            }
            NodeKind::Program(nodes)
            | NodeKind::Statements(nodes)
            | NodeKind::UnscopedStatements(nodes)
            | NodeKind::Block(nodes) => nodes
                .last()
                .is_some_and(|node| self.node_guarantees_cloneable_return(node, recursive_name)),
            _ => false,
        }
    }

    fn function_is_pure_cloneable_ast(&self, name: &str, args: &[String], body: &Node) -> bool {
        self.function_is_pure_int_ast(name, args, body)
            && self.node_guarantees_cloneable_return(body, name)
    }

    fn count_recursive_calls(&self, node: &Node, recursive_name: &str, count: &mut usize) {
        if *count >= 2 {
            return;
        }
        match node.kind() {
            NodeKind::FunctionCall(func, args) => {
                if matches!(func.kind(), NodeKind::Ident(name) if name == recursive_name) {
                    *count += 1;
                    if *count >= 2 {
                        return;
                    }
                }
                self.count_recursive_calls(func, recursive_name, count);
                for arg in args {
                    self.count_recursive_calls(arg, recursive_name, count);
                }
            }
            NodeKind::Program(nodes)
            | NodeKind::Statements(nodes)
            | NodeKind::UnscopedStatements(nodes)
            | NodeKind::Block(nodes)
            | NodeKind::List(nodes) => {
                for node in nodes {
                    self.count_recursive_calls(node, recursive_name, count);
                }
            }
            NodeKind::Dict(entries) => {
                for (key, value) in entries {
                    self.count_recursive_calls(key, recursive_name, count);
                    self.count_recursive_calls(value, recursive_name, count);
                }
            }
            NodeKind::BinOp(left, _, right)
            | NodeKind::Index(left, right)
            | NodeKind::Range(Some(left), right)
            | NodeKind::Try(left, _, right) => {
                self.count_recursive_calls(left, recursive_name, count);
                self.count_recursive_calls(right, recursive_name, count);
            }
            NodeKind::UnaryOp(_, expr)
            | NodeKind::ExpressionStatement(expr)
            | NodeKind::Return(expr)
            | NodeKind::Range(None, expr)
            | NodeKind::Await(expr)
            | NodeKind::Defer(expr)
            | NodeKind::Free(expr)
            | NodeKind::Print(expr)
            | NodeKind::Println(expr)
            | NodeKind::Throw(expr)
            | NodeKind::MemberAccess(expr, _) => {
                self.count_recursive_calls(expr, recursive_name, count);
            }
            NodeKind::If(condition, then_branch, else_branch) => {
                self.count_recursive_calls(condition, recursive_name, count);
                self.count_recursive_calls(then_branch, recursive_name, count);
                if let Some(branch) = else_branch.as_deref() {
                    self.count_recursive_calls(branch, recursive_name, count);
                }
            }
            NodeKind::Ternary(condition, when_true, when_false) => {
                self.count_recursive_calls(condition, recursive_name, count);
                self.count_recursive_calls(when_true, recursive_name, count);
                self.count_recursive_calls(when_false, recursive_name, count);
            }
            NodeKind::While(condition, body) | NodeKind::For(_, condition, body) => {
                self.count_recursive_calls(condition, recursive_name, count);
                self.count_recursive_calls(body, recursive_name, count);
            }
            NodeKind::FString(parts) => {
                for part in parts {
                    if let FStringPart::Expr(expr) = part {
                        self.count_recursive_calls(expr, recursive_name, count);
                    }
                }
            }
            NodeKind::Assign(_, expr) | NodeKind::Reassign(_, expr, _) => {
                self.count_recursive_calls(expr, recursive_name, count);
            }
            NodeKind::IndexAssign(object, index, value) => {
                self.count_recursive_calls(object, recursive_name, count);
                self.count_recursive_calls(index, recursive_name, count);
                self.count_recursive_calls(value, recursive_name, count);
            }
            _ => {}
        }
    }

    fn has_multiple_recursive_calls(&self, node: &Node, recursive_name: &str) -> bool {
        let mut count = 0;
        self.count_recursive_calls(node, recursive_name, &mut count);
        count >= 2
    }

    fn instruction_set_is_pure_int(&self, name: &str, instructions: &InstructionSet) -> bool {
        instructions
            .instructions
            .iter()
            .all(|instruction| match instruction.opcode() {
                Opcode::LoadConst(_)
                | Opcode::LoadConst0
                | Opcode::LoadConst1
                | Opcode::Load(_)
                | Opcode::LoadLocal0
                | Opcode::LoadLocal1
                | Opcode::LoadLocal2
                | Opcode::LoadLocal3
                | Opcode::LoadLocal4
                | Opcode::LoadLocal5
                | Opcode::LoadLocal6
                | Opcode::LoadLocal7
                | Opcode::LoadLocal8
                | Opcode::LoadLocal9
                | Opcode::LoadLocal10
                | Opcode::LoadLocal11
                | Opcode::StoreAt(_)
                | Opcode::StoreLocal0
                | Opcode::StoreLocal1
                | Opcode::StoreLocal2
                | Opcode::StoreLocal3
                | Opcode::StoreLocal4
                | Opcode::StoreLocal5
                | Opcode::StoreLocal6
                | Opcode::StoreLocal7
                | Opcode::StoreLocal8
                | Opcode::StoreLocal9
                | Opcode::StoreLocal10
                | Opcode::StoreLocal11
                | Opcode::Reassign(_)
                | Opcode::ReassignLocal0
                | Opcode::ReassignLocal1
                | Opcode::ReassignLocal2
                | Opcode::ReassignLocal3
                | Opcode::ReassignLocal4
                | Opcode::ReassignLocal5
                | Opcode::ReassignLocal6
                | Opcode::ReassignLocal7
                | Opcode::ReassignLocal8
                | Opcode::ReassignLocal9
                | Opcode::ReassignLocal10
                | Opcode::ReassignLocal11
                | Opcode::AddAssignLocal(_)
                | Opcode::AddAssignLocalInt(_)
                | Opcode::IncrementLocal(_)
                | Opcode::DecrementLocal(_)
                | Opcode::JumpIfFalse(_)
                | Opcode::Jump(_)
                | Opcode::PopLocal(_)
                | Opcode::Index
                | Opcode::IndexConst(_)
                | Opcode::IndexLocal(_)
                | Opcode::IndexLocalConst(_, _)
                | Opcode::IndexLocalLocal(_, _)
                | Opcode::IndexLocalLocalAdd1(_, _)
                | Opcode::List(_)
                | Opcode::Dict(_)
                | Opcode::Range
                | Opcode::True
                | Opcode::False
                | Opcode::Void
                | Opcode::Return
                | Opcode::Pop
                | Opcode::Add
                | Opcode::Subtract
                | Opcode::Multiply
                | Opcode::Divide
                | Opcode::Modulo
                | Opcode::Negate
                | Opcode::Equal
                | Opcode::NotEqual
                | Opcode::Greater
                | Opcode::GreaterEqual
                | Opcode::Less
                | Opcode::LessEqual
                | Opcode::AddInt
                | Opcode::AddInt1
                | Opcode::SubtractInt
                | Opcode::SubtractInt1
                | Opcode::SubtractInt2
                | Opcode::MultiplyInt
                | Opcode::DivideInt
                | Opcode::ModuloInt
                | Opcode::LessInt
                | Opcode::LessEqualInt
                | Opcode::LessEqualInt1
                | Opcode::ForRangeInit(_)
                | Opcode::ForRangeNext(_, _)
                | Opcode::ForRangeNextDiscard(_, _)
                | Opcode::Nop => true,
                Opcode::CallSelf(_) | Opcode::CallSelf1 => true,
                Opcode::CallGlobal1(index) => instructions
                    .globals
                    .get_name(index as usize)
                    .is_some_and(|callee| {
                        callee == name || self.known_pure_int_functions.contains(callee)
                    }),
                Opcode::CallGlobal(index, _) => instructions
                    .globals
                    .get_name(index as usize)
                    .is_some_and(|callee| {
                        callee == name || self.known_pure_int_functions.contains(callee)
                    }),
                _ => false,
            })
    }

    fn references_name(node: &Node, name: &str) -> bool {
        match node.kind() {
            NodeKind::Program(nodes)
            | NodeKind::Statements(nodes)
            | NodeKind::UnscopedStatements(nodes)
            | NodeKind::List(nodes)
            | NodeKind::Block(nodes)
            | NodeKind::StructDefinition(_, nodes) => {
                nodes.iter().any(|node| Self::references_name(node, name))
            }
            NodeKind::FString(parts) => parts.iter().any(|part| match part {
                FStringPart::Literal(_) => false,
                FStringPart::Expr(expr) => Self::references_name(expr, name),
            }),
            NodeKind::Dict(entries) => entries.iter().any(|(key, value)| {
                Self::references_name(key, name) || Self::references_name(value, name)
            }),
            NodeKind::BinOp(left, _, right)
            | NodeKind::Index(left, right)
            | NodeKind::Range(Some(left), right)
            | NodeKind::Try(left, _, right) => {
                Self::references_name(left, name) || Self::references_name(right, name)
            }
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
            | NodeKind::MemberAccess(expr, _) => Self::references_name(expr, name),
            NodeKind::Reassign(ident, expr, _) => {
                ident.value() == name || Self::references_name(expr, name)
            }
            NodeKind::IndexAssign(target, index, value) => {
                Self::references_name(target, name)
                    || Self::references_name(index, name)
                    || Self::references_name(value, name)
            }
            NodeKind::FunctionCall(func, args) => {
                Self::references_name(func, name)
                    || args.iter().any(|arg| Self::references_name(arg, name))
            }
            NodeKind::AnonFunctionDefinition(_, body)
            | NodeKind::AsyncAnonFunctionDefinition(_, body)
            | NodeKind::FunctionDefinition(_, _, body)
            | NodeKind::AsyncFunctionDefinition(_, _, body)
            | NodeKind::StructFunctionDefinition(_, _, body) => Self::references_name(body, name),
            NodeKind::While(condition, body) => {
                Self::references_name(condition, name) || Self::references_name(body, name)
            }
            NodeKind::If(condition, then_branch, else_branch) => {
                Self::references_name(condition, name)
                    || Self::references_name(then_branch, name)
                    || else_branch
                        .as_deref()
                        .is_some_and(|branch| Self::references_name(branch, name))
            }
            NodeKind::Ternary(condition, when_true, when_false) => {
                Self::references_name(condition, name)
                    || Self::references_name(when_true, name)
                    || Self::references_name(when_false, name)
            }
            NodeKind::For(_, iter, body) => {
                Self::references_name(iter, name) || Self::references_name(body, name)
            }
            NodeKind::Range(None, end) => Self::references_name(end, name),
            NodeKind::Ident(ident) => ident == name,
            NodeKind::PackageImport(_, Some(alias)) | NodeKind::ModuleImport(_, Some(alias)) => {
                alias == name
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

    fn collect_accumulator_add_terms(node: &Node, name: &str, terms: &mut Vec<Node>) -> bool {
        match node.kind() {
            NodeKind::Ident(ident) => ident == name,
            NodeKind::BinOp(left, Opcode::Add, right) => {
                if Self::collect_accumulator_add_terms(left, name, terms) {
                    terms.push((**right).clone());
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn string_constant_index(&mut self, node: &Node) -> Option<u32> {
        match node.kind() {
            NodeKind::String(value) => {
                let string_value = self
                    .instructions
                    .get_heap_mut()
                    .push(HeapValue::String(value));
                Some(self.instructions.push_constant(string_value))
            }
            _ => None,
        }
    }

    fn local_int_index_pattern(&self, node: &Node) -> Option<(u32, bool)> {
        match node.kind() {
            NodeKind::Ident(name) if self.is_known_int_name(name) => self
                .instructions
                .resolve_local_index(name)
                .map(|index| (index as u32, false)),
            NodeKind::BinOp(left, Opcode::Add, right)
                if matches!(right.kind(), NodeKind::Int(1)) =>
            {
                match left.kind() {
                    NodeKind::Ident(name) if self.is_known_int_name(name) => self
                        .instructions
                        .resolve_local_index(name)
                        .map(|index| (index as u32, true)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn local_const_index_pattern(&mut self, node: &Node) -> Option<(u32, u32)> {
        let NodeKind::Index(object, index) = node.kind() else {
            return None;
        };
        let NodeKind::Ident(name) = object.kind() else {
            return None;
        };
        let local_index = self.instructions.resolve_local_index(name)? as u32;
        let constant = optimize::try_get_constant(index)?;
        let constant_index = self.instructions.push_constant(constant);
        Some((local_index, constant_index))
    }

    fn local_void_condition_pattern(&self, node: &Node) -> Option<u32> {
        let NodeKind::BinOp(left, Opcode::Equal, right) = node.kind() else {
            return None;
        };

        match (left.kind(), right.kind()) {
            (NodeKind::Ident(name), NodeKind::Void) | (NodeKind::Void, NodeKind::Ident(name)) => {
                self.instructions
                    .resolve_local_index(name)
                    .map(|index| index as u32)
            }
            _ => None,
        }
    }

    fn local_less_equal_zero_condition_pattern(&self, node: &Node) -> Option<u32> {
        let NodeKind::BinOp(left, Opcode::LessEqual, right) = node.kind() else {
            return None;
        };

        match (left.kind(), right.kind()) {
            (NodeKind::Ident(name), NodeKind::Int(0)) => self
                .instructions
                .resolve_local_index(name)
                .map(|index| index as u32),
            _ => None,
        }
    }

    fn adjacent_local_compare_pattern(
        &self,
        left: &Node,
        op: Opcode,
        right: &Node,
    ) -> Option<Opcode> {
        if op != Opcode::Greater {
            return None;
        }

        let NodeKind::Index(left_object, left_index) = left.kind() else {
            return None;
        };
        let NodeKind::Index(right_object, right_index) = right.kind() else {
            return None;
        };
        let NodeKind::Ident(left_name) = left_object.kind() else {
            return None;
        };
        let NodeKind::Ident(right_name) = right_object.kind() else {
            return None;
        };

        if left_name != right_name {
            return None;
        }

        let local_index = self.instructions.resolve_local_index(left_name)? as u32;
        let Some((left_index_local, false)) = self.local_int_index_pattern(left_index) else {
            return None;
        };
        let Some((right_index_local, true)) = self.local_int_index_pattern(right_index) else {
            return None;
        };

        if left_index_local != right_index_local {
            return None;
        }

        Some(Opcode::GreaterIndexLocalLocalAdd1(
            local_index,
            left_index_local,
        ))
    }

    fn statement_expr(node: &Node) -> &Node {
        match node.kind() {
            NodeKind::ExpressionStatement(expr) => expr,
            _ => node,
        }
    }

    fn is_return_int_zero(node: &Node) -> bool {
        matches!(
            Self::statement_expr(node).kind(),
            NodeKind::Return(expr) if matches!(expr.kind(), NodeKind::Int(0))
        )
    }

    fn is_return_void(node: &Node) -> bool {
        matches!(
            Self::statement_expr(node).kind(),
            NodeKind::Return(expr) if matches!(expr.kind(), NodeKind::Void)
        )
    }

    fn adjacent_local_swap_pattern(&self, node: &Node) -> Option<(u32, u32)> {
        let nodes = match node.kind() {
            NodeKind::Program(nodes)
            | NodeKind::Statements(nodes)
            | NodeKind::UnscopedStatements(nodes)
            | NodeKind::Block(nodes) => nodes,
            _ => return None,
        };

        let [first, second, third] = nodes.as_slice() else {
            return None;
        };

        let first = Self::statement_expr(first);
        let second = Self::statement_expr(second);
        let third = Self::statement_expr(third);

        let NodeKind::Assign(temp_name, first_value) = first.kind() else {
            return None;
        };
        let NodeKind::Index(first_object, first_index) = first_value.kind() else {
            return None;
        };
        let NodeKind::Ident(list_name) = first_object.kind() else {
            return None;
        };
        let list_local = self.instructions.resolve_local_index(list_name)? as u32;
        let (index_local, false) = self.local_int_index_pattern(first_index)? else {
            return None;
        };

        let NodeKind::IndexAssign(second_object, second_index, second_value) = second.kind() else {
            return None;
        };
        let NodeKind::Ident(second_list_name) = second_object.kind() else {
            return None;
        };
        if second_list_name != list_name {
            return None;
        }
        let (second_index_local, false) = self.local_int_index_pattern(second_index)? else {
            return None;
        };
        if second_index_local != index_local {
            return None;
        }
        let NodeKind::Index(second_rhs_object, second_rhs_index) = second_value.kind() else {
            return None;
        };
        let NodeKind::Ident(second_rhs_list_name) = second_rhs_object.kind() else {
            return None;
        };
        if second_rhs_list_name != list_name {
            return None;
        }
        let (second_rhs_index_local, true) = self.local_int_index_pattern(second_rhs_index)? else {
            return None;
        };
        if second_rhs_index_local != index_local {
            return None;
        }

        let NodeKind::IndexAssign(third_object, third_index, third_value) = third.kind() else {
            return None;
        };
        let NodeKind::Ident(third_list_name) = third_object.kind() else {
            return None;
        };
        if third_list_name != list_name {
            return None;
        }
        let (third_index_local, true) = self.local_int_index_pattern(third_index)? else {
            return None;
        };
        if third_index_local != index_local {
            return None;
        }
        let NodeKind::Ident(third_value_name) = third_value.kind() else {
            return None;
        };
        if third_value_name != temp_name {
            return None;
        }

        Some((list_local, index_local))
    }

    pub fn emit(&mut self, node: Node) -> WalrusResult<()> {
        let kind = node.kind().to_string();
        let span = *node.span();

        match node.into_kind() {
            NodeKind::Program(nodes) => {
                // Two-pass compilation for forward declarations:
                self.prepare_top_level_scope(&nodes);
                // Pass 1: Pre-register all top-level function and struct names
                for node in &nodes {
                    self.pre_register_declarations(node);
                }
                // Pass 2: Compile everything normally
                for node in nodes {
                    self.emit(node)?;
                }
            }
            NodeKind::Int(value) => {
                let index = self.instructions.push_constant(Value::Int(value));
                let opcode = match index {
                    0 => Opcode::LoadConst0,
                    1 => Opcode::LoadConst1,
                    _ => Opcode::LoadConst(index),
                };
                self.instructions.push(Instruction::new(opcode, span));
            }
            NodeKind::Float(value) => {
                let index = self.instructions.push_constant(Value::Float(value));
                let opcode = match index {
                    0 => Opcode::LoadConst0,
                    1 => Opcode::LoadConst1,
                    _ => Opcode::LoadConst(index),
                };
                self.instructions.push(Instruction::new(opcode, span));
            }
            NodeKind::Bool(value) => {
                self.instructions.push(Instruction::new(
                    if value { Opcode::True } else { Opcode::False },
                    span,
                ));
            }
            NodeKind::String(value) => {
                let value = self
                    .instructions
                    .get_heap_mut()
                    .push(HeapValue::String(&value));
                let index = self.instructions.push_constant(value);
                let opcode = match index {
                    0 => Opcode::LoadConst0,
                    1 => Opcode::LoadConst1,
                    _ => Opcode::LoadConst(index),
                };
                self.instructions.push(Instruction::new(opcode, span));
            }
            NodeKind::FString(parts) => {
                // Compile f-strings by emitting each part and concatenating them
                let mut part_count = 0;

                for part in parts {
                    match part {
                        FStringPart::Literal(s) => {
                            // Push literal string constant
                            let value =
                                self.instructions.get_heap_mut().push(HeapValue::String(&s));
                            let index = self.instructions.push_constant(value);
                            let opcode = match index {
                                0 => Opcode::LoadConst0,
                                1 => Opcode::LoadConst1,
                                _ => Opcode::LoadConst(index),
                            };
                            self.instructions.push(Instruction::new(opcode, span));
                            part_count += 1;
                        }
                        FStringPart::Expr(node) => {
                            // Expression is already parsed with proper span
                            self.emit(*node)?;
                            // Convert expression result to string for interpolation
                            self.instructions.push(Instruction::new(Opcode::Str, span));
                            part_count += 1;
                        }
                    }
                }

                if part_count == 0 {
                    // Ensure empty f-strings still push an empty string value.
                    let value = self.instructions.get_heap_mut().push(HeapValue::String(""));
                    let index = self.instructions.push_constant(value);
                    let opcode = match index {
                        0 => Opcode::LoadConst0,
                        1 => Opcode::LoadConst1,
                        _ => Opcode::LoadConst(index),
                    };
                    self.instructions.push(Instruction::new(opcode, span));
                    return Ok(());
                }

                // Concatenate all parts using Add operations
                for _ in 1..part_count {
                    self.instructions.push(Instruction::new(Opcode::Add, span));
                }
            }
            NodeKind::List(nodes) => {
                let cap = nodes.len();

                // Emit items in correct order (no need to reverse at runtime)
                for node in nodes {
                    self.emit(node)?;
                }

                self.instructions
                    .push(Instruction::new(Opcode::List(cap as u32), span));
            }
            NodeKind::Dict(nodes) => {
                let constant_keys: Option<Vec<Value>> = nodes
                    .iter()
                    .map(|(key, _)| optimize::try_get_constant(key))
                    .collect();

                if let Some(keys) = constant_keys {
                    for (_, value) in nodes {
                        self.emit(value)?;
                    }

                    let key_tuple = self
                        .instructions
                        .get_heap_mut()
                        .push(HeapValue::Tuple(&keys));
                    let key_index = self.instructions.push_constant(key_tuple);
                    self.instructions
                        .push(Instruction::new(Opcode::DictConstKeys(key_index), span));
                    return Ok(());
                }

                let cap = nodes.len();

                for (key, value) in nodes {
                    self.emit(key)?;
                    self.emit(value)?;
                }

                self.instructions
                    .push(Instruction::new(Opcode::Dict(cap as u32), span));
            }
            NodeKind::Range(left, right) => {
                // Right is always present
                self.emit(*right)?;

                if let Some(left) = left {
                    self.emit(*left)?;
                } else {
                    self.instructions.push(Instruction::new(Opcode::Void, span));
                }

                self.instructions
                    .push(Instruction::new(Opcode::Range, span));
            }
            NodeKind::Void => {
                self.instructions.push(Instruction::new(Opcode::Void, span));
            }
            NodeKind::BinOp(left, op, right) => {
                // Try constant folding first
                if let Some(folded) = optimize::try_fold_binop(&left, op, &right) {
                    self.emit_constant(folded, span);
                    return Ok(());
                }

                // Short-circuit evaluation for And/Or
                match op {
                    Opcode::And => {
                        // Evaluate left
                        self.emit(*left)?;
                        // Duplicate for conditional check
                        self.instructions.push(Instruction::new(Opcode::Dup, span));
                        // If false, skip right and keep false
                        let jump = self.instructions.len();
                        self.instructions
                            .push(Instruction::new(Opcode::JumpIfFalse(0), span));
                        // Pop the duplicated left value (it was true)
                        self.instructions.push(Instruction::new(Opcode::Pop, span));
                        // Evaluate right
                        self.emit(*right)?;
                        // Set jump target to current position
                        self.instructions.set(
                            jump,
                            Instruction::new(
                                Opcode::JumpIfFalse(self.instructions.len() as u32),
                                span,
                            ),
                        );
                    }
                    Opcode::Or => {
                        // Evaluate left
                        self.emit(*left)?;
                        // Duplicate for conditional check
                        self.instructions.push(Instruction::new(Opcode::Dup, span));
                        // If true, skip right and keep true
                        let jump_if_false = self.instructions.len();
                        self.instructions
                            .push(Instruction::new(Opcode::JumpIfFalse(0), span));
                        // Left was true, skip right evaluation
                        let jump_end = self.instructions.len();
                        self.instructions
                            .push(Instruction::new(Opcode::Jump(0), span));
                        // Left was false, pop it and evaluate right
                        self.instructions.set(
                            jump_if_false,
                            Instruction::new(
                                Opcode::JumpIfFalse(self.instructions.len() as u32),
                                span,
                            ),
                        );
                        self.instructions.push(Instruction::new(Opcode::Pop, span));
                        self.emit(*right)?;
                        // Set end jump target
                        self.instructions.set(
                            jump_end,
                            Instruction::new(Opcode::Jump(self.instructions.len() as u32), span),
                        );
                    }
                    _ => {
                        if let Some(opcode) = self.adjacent_local_compare_pattern(&left, op, &right)
                        {
                            self.instructions.push(Instruction::new(opcode, span));
                            return Ok(());
                        }

                        // Normal binary operations
                        let specialized = self.specialized_int_opcode(op, &left, &right);
                        self.emit(*left)?;
                        if !matches!(
                            specialized,
                            Some(
                                Opcode::AddInt1
                                    | Opcode::SubtractInt1
                                    | Opcode::SubtractInt2
                                    | Opcode::LessEqualInt1
                            )
                        ) {
                            self.emit(*right)?;
                        }
                        self.instructions
                            .push(Instruction::new(specialized.unwrap_or(op), span));
                    }
                }
            }
            NodeKind::UnaryOp(op, node) => {
                // Try constant folding first
                if let Some(folded) = optimize::try_fold_unary(op, &node) {
                    self.emit_constant(folded, span);
                    return Ok(());
                }

                self.emit(*node)?;

                self.instructions.push(Instruction::new(op, span));
            }
            NodeKind::If(cond, then, otherwise) => {
                let swap_adjacent = self.adjacent_local_swap_pattern(&then);

                if otherwise.is_none() {
                    if let Some(local_index) = self.local_void_condition_pattern(&cond) {
                        if Self::is_return_int_zero(&then) {
                            self.instructions.push(Instruction::new(
                                Opcode::ReturnInt0IfLocalVoid(local_index),
                                span,
                            ));
                            return Ok(());
                        }
                    }

                    if let Some(local_index) = self.local_less_equal_zero_condition_pattern(&cond) {
                        if Self::is_return_void(&then) {
                            self.instructions.push(Instruction::new(
                                Opcode::ReturnVoidIfLocalLessEqualZero(local_index),
                                span,
                            ));
                            return Ok(());
                        }
                    }
                }

                if let Some(local_index) = self.local_void_condition_pattern(&cond) {
                    let jump = self.instructions.len();
                    self.instructions.push(Instruction::new(
                        Opcode::JumpIfLocalNotVoid(local_index, 0),
                        span,
                    ));

                    if let Some((list_local, index_local)) = swap_adjacent {
                        self.instructions.push(Instruction::new(
                            Opcode::SwapAdjacentLocal(list_local, index_local),
                            span,
                        ));
                    } else {
                        self.emit(*then)?;
                    }

                    let jump_else = self.instructions.len();
                    self.instructions
                        .push(Instruction::new(Opcode::Jump(0), span));

                    self.instructions.set(
                        jump,
                        Instruction::new(
                            Opcode::JumpIfLocalNotVoid(local_index, self.instructions.len() as u32),
                            span,
                        ),
                    );

                    if let Some(otherwise) = otherwise {
                        self.emit(*otherwise)?;
                    }

                    self.instructions.set(
                        jump_else,
                        Instruction::new(Opcode::Jump(self.instructions.len() as u32), span),
                    );
                    return Ok(());
                }

                self.emit(*cond)?;

                let jump = self.instructions.len();

                self.instructions
                    .push(Instruction::new(Opcode::JumpIfFalse(0), span));

                if let Some((list_local, index_local)) = swap_adjacent {
                    self.instructions.push(Instruction::new(
                        Opcode::SwapAdjacentLocal(list_local, index_local),
                        span,
                    ));
                } else {
                    self.emit(*then)?;
                }

                let jump_else = self.instructions.len();

                self.instructions
                    .push(Instruction::new(Opcode::Jump(0), span));

                self.instructions.set(
                    jump,
                    Instruction::new(Opcode::JumpIfFalse(self.instructions.len() as u32), span),
                );

                if let Some(otherwise) = otherwise {
                    self.emit(*otherwise)?;
                }

                self.instructions.set(
                    jump_else,
                    // todo: check if span is right
                    Instruction::new(Opcode::Jump(self.instructions.len() as u32), span),
                );
            }
            NodeKind::While(cond, body) => {
                let start = self.instructions.len();

                // Push loop context - for while loops, body handles its own scoping,
                // so locals_at_start is the current count
                self.loop_stack.push(LoopContext {
                    start,
                    breaks: Vec::new(),
                    has_stack_iterator: false,
                    locals_at_start: self.instructions.local_len(),
                });

                self.emit(*cond)?;

                let jump = self.instructions.len();

                self.instructions
                    .push(Instruction::new(Opcode::JumpIfFalse(0), span));

                self.emit(*body)?;

                let back_edge = self.instructions.len();
                self.instructions
                    .push(Instruction::new(Opcode::Jump(start as u32), span));

                let exit_ip = self.instructions.len();
                self.instructions.set(
                    jump,
                    Instruction::new(Opcode::JumpIfFalse(exit_ip as u32), span),
                );

                // JIT: Register the while loop for hot-spot detection
                self.instructions
                    .register_loop(start, back_edge, exit_ip, false);

                // Pop loop context and patch break jumps
                if let Some(loop_ctx) = self.loop_stack.pop() {
                    let end = self.instructions.len();
                    for break_addr in loop_ctx.breaks {
                        self.instructions
                            .set(break_addr, Instruction::new(Opcode::Jump(end as u32), span));
                    }
                }
            }
            NodeKind::For(name, iter, body) => {
                // Check if this is a range-based for loop that we can optimize
                if let NodeKind::Range(start_opt, end_node) = iter.kind() {
                    // Optimized range loop - no heap allocation!
                    // Emit start (default to 0) and end
                    if let Some(start_node) = start_opt {
                        self.emit(*start_node.clone())?;
                    } else {
                        // Push the integer 0
                        let index = self.instructions.push_constant(Value::Int(0));
                        let opcode = match index {
                            0 => Opcode::LoadConst0,
                            1 => Opcode::LoadConst1,
                            _ => Opcode::LoadConst(index),
                        };
                        self.instructions.push(Instruction::new(opcode, span));
                    }
                    self.emit(*end_node.clone())?;

                    // Reserve two locals for range tracking (OUTSIDE of depth scope so they persist)
                    let range_idx = self.instructions.push_local(format!("__range_{}", name));
                    let _ = self
                        .instructions
                        .push_local(format!("__range_end_{}", name));

                    // Initialize the range locals
                    self.instructions
                        .push(Instruction::new(Opcode::ForRangeInit(range_idx), span));

                    let uses_loop_var = Self::references_name(&body, &name);
                    let var_idx = if uses_loop_var {
                        let var_idx = self.instructions.push_local(name.clone());
                        self.mark_known_int(&name, false, true);
                        Some(var_idx)
                    } else {
                        None
                    };

                    let jump = self.instructions.len();

                    let loop_opcode = if uses_loop_var {
                        Opcode::ForRangeNext(0, range_idx)
                    } else {
                        Opcode::ForRangeNextDiscard(0, range_idx)
                    };
                    self.instructions.push(Instruction::new(loop_opcode, span));

                    if let Some(var_idx) = var_idx {
                        self.instructions
                            .push(Instruction::new(Self::local_store_opcode(var_idx), span));
                    }

                    // Now increase depth for the loop body
                    self.inc_depth();

                    // Push loop context - range loops don't have an iterator on the stack
                    self.loop_stack.push(LoopContext {
                        start: jump,
                        breaks: Vec::new(),
                        has_stack_iterator: false,
                        locals_at_start: self.instructions.local_len(),
                    });

                    self.emit(*body)?;

                    let loop_ctx = self.loop_stack.pop();

                    // Only pop body locals, not range locals
                    self.dec_depth(span);

                    let back_edge = self.instructions.len();
                    self.instructions
                        .push(Instruction::new(Opcode::Jump(jump as u32), span));

                    // Patch the ForRangeNext to jump past the loop cleanup locals.
                    let end_pos = self.instructions.len();
                    self.instructions.set(
                        jump,
                        Instruction::new(
                            if uses_loop_var {
                                Opcode::ForRangeNext(end_pos as u32, range_idx)
                            } else {
                                Opcode::ForRangeNextDiscard(end_pos as u32, range_idx)
                            },
                            span,
                        ),
                    );

                    // JIT: Register the range-based for loop for hot-spot detection
                    // The header is at 'jump' where ForRangeNext is
                    self.instructions
                        .register_loop(jump, back_edge, end_pos, true);

                    let range_local_count: u32 = if uses_loop_var { 3 } else { 2 };
                    self.instructions
                        .push(Instruction::new(Opcode::PopLocal(range_local_count), span));

                    // Patch break jumps (they need to jump past the PopLocal too)
                    if let Some(loop_ctx) = loop_ctx {
                        let end = self.instructions.len();
                        for break_addr in loop_ctx.breaks {
                            self.instructions
                                .set(break_addr, Instruction::new(Opcode::Jump(end as u32), span));
                        }
                    }

                    self.instructions.pop_locals(range_local_count as usize);
                } else {
                    // Generic iterator protocol path:
                    //   1. Get iterator via GetIter
                    //   2. Call iterator.next() each iteration
                    //   3. Stop when next() returns void
                    self.emit(*iter)?;

                    self.instructions
                        .push(Instruction::new(Opcode::GetIter, span));

                    // Method name constant for iterator.next()
                    let next_method = self
                        .instructions
                        .get_heap_mut()
                        .push(HeapValue::String("next"));
                    let next_method_const = self.instructions.push_constant(next_method);
                    let next_method_opcode = match next_method_const {
                        0 => Opcode::LoadConst0,
                        1 => Opcode::LoadConst1,
                        _ => Opcode::LoadConst(next_method_const),
                    };

                    let jump = self.instructions.len();

                    // Keep one iterator copy on stack, call next() on the duplicate.
                    self.instructions.push(Instruction::new(Opcode::Dup, span));
                    self.instructions
                        .push(Instruction::new(next_method_opcode, span));
                    self.instructions
                        .push(Instruction::new(Opcode::CallMethod(0), span));

                    // If result == void, iteration is finished.
                    self.instructions.push(Instruction::new(Opcode::Dup, span));
                    self.instructions.push(Instruction::new(Opcode::Void, span));
                    self.instructions
                        .push(Instruction::new(Opcode::Equal, span));
                    let continue_jump_addr = self.instructions.len();
                    self.instructions
                        .push(Instruction::new(Opcode::JumpIfFalse(0), span));
                    // Done path: remove (void result, iterator) and exit loop.
                    self.instructions.push(Instruction::new(Opcode::Pop, span));
                    self.instructions.push(Instruction::new(Opcode::Pop, span));
                    let done_jump_addr = self.instructions.len();
                    self.instructions
                        .push(Instruction::new(Opcode::Jump(0), span));

                    let continue_ip = self.instructions.len();
                    self.instructions.set(
                        continue_jump_addr,
                        Instruction::new(Opcode::JumpIfFalse(continue_ip as u32), span),
                    );

                    self.inc_depth();

                    let index = self.instructions.push_local(name);

                    self.instructions
                        .push(Instruction::new(Self::local_store_opcode(index), span));

                    // Push loop context AFTER defining loop variable
                    // locals_at_start is the count including the loop variable,
                    // so continue will pop body-declared vars but keep the loop var
                    self.loop_stack.push(LoopContext {
                        start: jump,
                        breaks: Vec::new(),
                        has_stack_iterator: true,
                        locals_at_start: self.instructions.local_len(),
                    });

                    self.emit(*body)?;

                    // Pop loop context before dec_depth so continue can reference it
                    let loop_ctx = self.loop_stack.pop();

                    // Pop locals declared in the loop body (but not the loop variable itself)
                    // This is crucial: without this, variables declared inside the loop body
                    // would reuse their slots on subsequent iterations, seeing stale values
                    self.dec_depth(span);

                    let back_edge = self.instructions.len();
                    self.instructions
                        .push(Instruction::new(Opcode::Jump(jump as u32), span));

                    let exit_ip = self.instructions.len();
                    self.instructions.set(
                        done_jump_addr,
                        Instruction::new(Opcode::Jump(exit_ip as u32), span),
                    );

                    // JIT: Register the iterator-based for loop for hot-spot detection
                    self.instructions
                        .register_loop(jump, back_edge, exit_ip, false);

                    // Patch break jumps
                    if let Some(loop_ctx) = loop_ctx {
                        let end = self.instructions.len();
                        for break_addr in loop_ctx.breaks {
                            self.instructions
                                .set(break_addr, Instruction::new(Opcode::Jump(end as u32), span));
                        }
                    }
                }
            }
            NodeKind::FunctionDefinition(name, args, body) => {
                self.emit_named_function_definition(name, args, *body, span, false)?;
            }
            NodeKind::AsyncFunctionDefinition(name, args, body) => {
                self.emit_named_function_definition(name, args, *body, span, true)?;
            }
            NodeKind::Await(value) => {
                self.emit(*value)?;
                self.instructions
                    .push(Instruction::new(Opcode::Await, span));
            }
            NodeKind::AnonFunctionDefinition(args, body) => {
                self.emit_anon_function_definition(args, *body, span, false)?;
            }
            NodeKind::AsyncAnonFunctionDefinition(args, body) => {
                self.emit_anon_function_definition(args, *body, span, true)?;
            }
            NodeKind::FunctionCall(func, args) => {
                self.emit_function_call(func, args, span, false)?;
            }
            NodeKind::Index(object, index) => {
                if let NodeKind::Ident(name) = object.kind() {
                    if let Some(local_index) = self.instructions.resolve_local_index(name) {
                        if let Some((index_local, add_one)) = self.local_int_index_pattern(&index) {
                            let opcode = if add_one {
                                Opcode::IndexLocalLocalAdd1(local_index as u32, index_local)
                            } else {
                                Opcode::IndexLocalLocal(local_index as u32, index_local)
                            };
                            self.instructions.push(Instruction::new(opcode, span));
                            return Ok(());
                        }

                        if let Some(constant) = optimize::try_get_constant(&index) {
                            let constant_index = self.instructions.push_constant(constant);
                            self.instructions.push(Instruction::new(
                                Opcode::IndexLocalConst(local_index as u32, constant_index),
                                span,
                            ));
                        } else {
                            self.emit(*index)?;
                            self.instructions.push(Instruction::new(
                                Opcode::IndexLocal(local_index as u32),
                                span,
                            ));
                        }
                        return Ok(());
                    }
                }

                self.emit(*object)?;
                if let Some(constant) = optimize::try_get_constant(&index) {
                    let constant_index = self.instructions.push_constant(constant);
                    self.instructions
                        .push(Instruction::new(Opcode::IndexConst(constant_index), span));
                } else {
                    self.emit(*index)?;
                    self.instructions
                        .push(Instruction::new(Opcode::Index, span));
                }
            }
            NodeKind::Println(node) => {
                self.emit(*node)?;

                self.instructions
                    .push(Instruction::new(Opcode::Println, span));
            }
            NodeKind::Print(node) => {
                self.emit(*node)?;

                self.instructions
                    .push(Instruction::new(Opcode::Print, span));
            }
            NodeKind::ExpressionStatement(expr) => {
                self.emit(*expr)?;

                self.instructions.push(Instruction::new(Opcode::Pop, span));
            }
            NodeKind::Ident(name) => {
                // Check locals first, then globals
                if let Some(index) = self.instructions.resolve_local_index(&name) {
                    let opcode = Self::local_load_opcode(index as u32);
                    self.instructions.push(Instruction::new(opcode, span));
                } else if let Some(index) = self.instructions.resolve_global_index(&name) {
                    let opcode = Self::global_load_opcode(index as u32);
                    self.instructions.push(Instruction::new(opcode, span));
                } else if let Some(ref struct_name) = self.current_struct {
                    // In struct methods, bare identifiers may reference methods on the current
                    // struct. Only resolve names we know are methods; otherwise report the
                    // clearer undefined-variable error.
                    let is_known_method = self
                        .current_struct_methods
                        .as_ref()
                        .map_or(false, |methods| methods.contains(&name));

                    if !is_known_method {
                        return Err(WalrusError::UndefinedVariable {
                            name,
                            span,
                            src: self.source_ref.source().to_string(),
                            filename: self.source_ref.filename().to_string(),
                        });
                    }

                    // Load the struct definition
                    if let Some(struct_index) = self.instructions.resolve_global_index(struct_name)
                    {
                        let opcode = Self::global_load_opcode(struct_index as u32);
                        self.instructions.push(Instruction::new(opcode, span));

                        // Push the method name as a string
                        let method_str = self
                            .instructions
                            .get_heap_mut()
                            .push(HeapValue::String(&name));
                        let index = self.instructions.push_constant(method_str);
                        let opcode = match index {
                            0 => Opcode::LoadConst0,
                            1 => Opcode::LoadConst1,
                            _ => Opcode::LoadConst(index),
                        };
                        self.instructions.push(Instruction::new(opcode, span));

                        // Get the method from the struct
                        self.instructions
                            .push(Instruction::new(Opcode::GetMethod, span));
                    } else {
                        return Err(WalrusError::UndefinedVariable {
                            name,
                            span,
                            src: self.source_ref.source().to_string(),
                            filename: self.source_ref.filename().to_string(),
                        });
                    }
                } else {
                    return Err(WalrusError::UndefinedVariable {
                        name,
                        span,
                        src: self.source_ref.source().to_string(),
                        filename: self.source_ref.filename().to_string(),
                    });
                }
            }
            NodeKind::Assign(name, node) => {
                let is_global = self.depth == 0 && self.top_level_global_names.contains(&name);
                let value_is_known_int = self.is_known_int_expr(&node);

                if !is_global {
                    // Check for redefinition only in local scopes
                    if let Some(depth) = self.instructions.resolve_depth(&name) {
                        if depth >= self.instructions.local_depth() {
                            return Err(WalrusError::RedefinedLocal {
                                name,
                                span,
                                src: self.source_ref.source().to_string(),
                                filename: self.source_ref.filename().to_string(),
                            });
                        }
                    }
                }

                self.emit(*node)?;

                if is_global {
                    self.define_global_variable(name.clone(), span);
                    self.mark_known_int(&name, true, value_is_known_int);
                } else {
                    self.define_variable(name.clone(), span);
                    self.mark_known_int(&name, false, value_is_known_int);
                }
            }
            NodeKind::Reassign(name, node, op) => {
                // Check locals first, then globals
                let (index, is_global) = if let Some(index) =
                    self.instructions.resolve_local_index(name.value())
                {
                    (index as u32, false)
                } else if let Some(index) = self.instructions.resolve_global_index(name.value()) {
                    (index as u32, true)
                } else {
                    return Err(WalrusError::UndefinedVariable {
                        name: name.value().to_string(),
                        span,
                        src: self.source_ref.source().to_string(),
                        filename: self.source_ref.filename().to_string(),
                    });
                };

                // Optimization: use specialized increment/decrement opcodes for local variables
                if !is_global {
                    match optimize::analyze_reassign_for_increment(name.value(), &node, op) {
                        optimize::ReassignOptimization::Increment => {
                            self.instructions
                                .push(Instruction::new(Opcode::IncrementLocal(index), span));
                            if self.is_known_int_name(name.value()) {
                                self.mark_known_int(name.value(), false, true);
                            }
                            return Ok(());
                        }
                        optimize::ReassignOptimization::Decrement => {
                            self.instructions
                                .push(Instruction::new(Opcode::DecrementLocal(index), span));
                            if self.is_known_int_name(name.value()) {
                                self.mark_known_int(name.value(), false, true);
                            }
                            return Ok(());
                        }
                        optimize::ReassignOptimization::None => {}
                    }
                }

                let current_is_known_int = self.is_known_int_name(name.value());
                let rhs_is_known_int = self.is_known_int_expr(&node);
                let new_is_known_int = match op {
                    Opcode::Equal => rhs_is_known_int,
                    Opcode::Add
                    | Opcode::Subtract
                    | Opcode::Multiply
                    | Opcode::Divide
                    | Opcode::Modulo => current_is_known_int && rhs_is_known_int,
                    _ => false,
                };

                if op == Opcode::Add {
                    if !is_global {
                        if let Some(const_index) = self.string_constant_index(&node) {
                            self.instructions.push(Instruction::new(
                                Opcode::AppendStringLocalConst(index, const_index),
                                span,
                            ));
                            return Ok(());
                        }
                    }

                    self.emit(*node)?;
                    self.instructions.push(Instruction::new(
                        Self::add_assign_opcode(index, is_global, new_is_known_int),
                        span,
                    ));
                    self.mark_known_int(name.value(), is_global, new_is_known_int);
                    return Ok(());
                }

                if op == Opcode::Equal {
                    if !is_global {
                        if let NodeKind::BinOp(lhs, Opcode::Add, rhs) = node.kind() {
                            if matches!(lhs.kind(), NodeKind::Ident(var) if var == name.value()) {
                                if let Some(const_index) = self.string_constant_index(rhs) {
                                    self.instructions.push(Instruction::new(
                                        Opcode::AppendStringLocalConst(index, const_index),
                                        span,
                                    ));
                                    return Ok(());
                                }
                            }
                        }
                    }

                    let mut add_terms = Vec::new();
                    if Self::collect_accumulator_add_terms(&node, name.value(), &mut add_terms)
                        && current_is_known_int
                        && add_terms.iter().all(|term| self.is_known_int_expr(term))
                    {
                        for term in add_terms {
                            self.emit(term)?;
                            self.instructions.push(Instruction::new(
                                Self::add_assign_opcode(index, is_global, true),
                                span,
                            ));
                        }
                        self.mark_known_int(name.value(), is_global, true);
                        return Ok(());
                    }
                }

                // For compound assignments (+=, -=, etc.), we need to load the current value first
                match op {
                    Opcode::Subtract | Opcode::Multiply | Opcode::Divide | Opcode::Modulo => {
                        // Load current value
                        if is_global {
                            self.instructions
                                .push(Instruction::new(Self::global_load_opcode(index), span));
                        } else {
                            self.instructions
                                .push(Instruction::new(Self::local_load_opcode(index), span));
                        }
                        // Emit the right-hand side
                        self.emit(*node)?;
                        // Perform the operation
                        let opcode = if current_is_known_int && rhs_is_known_int {
                            match op {
                                Opcode::Add => Opcode::AddInt,
                                Opcode::Subtract => Opcode::SubtractInt,
                                Opcode::Multiply => Opcode::MultiplyInt,
                                Opcode::Divide => Opcode::DivideInt,
                                Opcode::Modulo => Opcode::ModuloInt,
                                _ => op,
                            }
                        } else {
                            op
                        };
                        self.instructions.push(Instruction::new(opcode, span));
                    }
                    _ => {
                        // Simple assignment (=), just emit the new value
                        self.emit(*node)?;
                    }
                }

                // Store the result back
                if is_global {
                    self.instructions
                        .push(Instruction::new(Opcode::ReassignGlobal(index), span));
                    self.mark_known_int(name.value(), true, new_is_known_int);
                } else {
                    self.instructions
                        .push(Instruction::new(Self::local_reassign_opcode(index), span));
                    self.mark_known_int(name.value(), false, new_is_known_int);
                }
            }
            NodeKind::Statements(nodes) => {
                // Two-pass compilation for forward declarations at global scope:
                if self.depth == 0 {
                    self.prepare_top_level_scope(&nodes);
                    // Pass 1: Pre-register all top-level function and struct names
                    for node in &nodes {
                        self.pre_register_declarations(node);
                    }

                    // Pass 2a: Compile function and struct definitions first (hoisting)
                    for node in &nodes {
                        match node.kind() {
                            NodeKind::FunctionDefinition(_, _, _)
                            | NodeKind::AsyncFunctionDefinition(_, _, _)
                            | NodeKind::StructDefinition(_, _) => {
                                self.emit(node.clone())?;
                            }
                            _ => {}
                        }
                    }

                    // Pass 2b: Compile everything else
                    for node in nodes {
                        match node.kind() {
                            NodeKind::FunctionDefinition(_, _, _)
                            | NodeKind::AsyncFunctionDefinition(_, _, _)
                            | NodeKind::StructDefinition(_, _) => {
                                // Already compiled above
                            }
                            _ => {
                                self.emit(node)?;
                            }
                        }
                    }
                } else {
                    // Non-global scope: just compile in order
                    self.inc_depth();
                    for node in nodes {
                        self.emit(node)?;
                    }
                    self.dec_depth(span);
                }
            }
            NodeKind::UnscopedStatements(nodes) => {
                for node in nodes {
                    self.emit(node)?;
                }
            }
            NodeKind::Return(node) => {
                // Check if this is a tail call (returning a function call directly)
                if let NodeKind::FunctionCall(func, args) = node.kind().clone() {
                    // This is a tail call - emit TailCall instead of Call + Return
                    self.emit_function_call(func, args, *node.span(), true)?;
                    return Ok(());
                }

                // Regular return: emit expression then Return opcode
                self.emit(*node)?;

                self.instructions
                    .push(Instruction::new(Opcode::Return, span));
            }
            NodeKind::Break => {
                if let Some(loop_ctx) = self.loop_stack.last_mut() {
                    // Pop any locals declared inside the loop body before breaking
                    let current_locals = self.instructions.local_len();
                    let to_pop = current_locals - loop_ctx.locals_at_start;
                    if to_pop > 0 {
                        self.instructions
                            .push(Instruction::new(Opcode::PopLocal(to_pop as u32), span));
                    }
                    // For iterator-based for-loops, pop the iterator from the stack before breaking
                    // Range-based for-loops don't have an iterator on the stack (they use locals)
                    if loop_ctx.has_stack_iterator {
                        self.instructions.push(Instruction::new(Opcode::Pop, span));
                    }
                    // Add a placeholder jump that will be patched later
                    let jump_addr = self.instructions.len();
                    self.instructions
                        .push(Instruction::new(Opcode::Jump(0), span));
                    loop_ctx.breaks.push(jump_addr);
                } else {
                    return Err(WalrusError::BreakOutsideLoop {
                        span,
                        src: self.source_ref.source().to_string(),
                        filename: self.source_ref.filename().to_string(),
                    });
                }
            }
            NodeKind::Continue => {
                if let Some(loop_ctx) = self.loop_stack.last() {
                    // Pop any locals declared inside the loop body before jumping back
                    // This ensures the locals vector is in the correct state for the next iteration
                    let current_locals = self.instructions.local_len();
                    let to_pop = current_locals - loop_ctx.locals_at_start;
                    if to_pop > 0 {
                        self.instructions
                            .push(Instruction::new(Opcode::PopLocal(to_pop as u32), span));
                    }
                    // Jump back to the loop start
                    self.instructions
                        .push(Instruction::new(Opcode::Jump(loop_ctx.start as u32), span));
                } else {
                    return Err(WalrusError::ContinueOutsideLoop {
                        span,
                        src: self.source_ref.source().to_string(),
                        filename: self.source_ref.filename().to_string(),
                    });
                }
            }
            NodeKind::StructDefinition(name, members) => {
                let is_global = self.depth == 0;

                // Get the struct index - use pre-registered index if it exists (forward declaration),
                // otherwise register it now
                let struct_global_index = if is_global {
                    Some(
                        if let Some(index) = self.instructions.resolve_global_index(&name) {
                            // Already pre-registered in Pass 1
                            index as u32
                        } else {
                            // Not pre-registered (shouldn't happen for top-level, but handle it)
                            self.instructions.push_global(name.clone())
                        },
                    )
                } else {
                    None
                };

                // Create a new struct definition
                let mut struct_def = crate::structs::StructDefinition::new(name.clone());
                let method_names: FxHashSet<String> = members
                    .iter()
                    .filter_map(|member| match member.kind() {
                        NodeKind::StructFunctionDefinition(method_name, _, _) => {
                            Some(method_name.clone())
                        }
                        _ => None,
                    })
                    .collect();

                // Process struct members (methods)
                for member in members {
                    let member_kind = member.kind().to_string();
                    match member.into_kind() {
                        NodeKind::StructFunctionDefinition(method_name, args, body) => {
                            // Create a child emitter for the method with struct context
                            let mut emitter = self.new_child();
                            emitter.current_struct = Some(name.clone());
                            emitter.current_struct_methods = Some(method_names.clone());
                            let arg_len = args.len();
                            emitter.current_function_name = Some(method_name.clone());
                            emitter.current_function_arity = Some(arg_len);
                            let int_params = Self::int_parameter_names(&args, &body);

                            // Define method parameters as locals
                            for arg in args {
                                let is_int = int_params.contains(&arg);
                                emitter.define_parameter(arg.clone());
                                if is_int {
                                    emitter.mark_known_int(&arg, false, true);
                                }
                            }

                            emitter.emit(*body)?;

                            // Add implicit void return if the method doesn't end with an explicit return
                            emitter.emit_void(span);
                            emitter.emit_return(span);

                            // Create the method function
                            let method_key =
                                self.instructions.get_heap_mut().push_ident(&method_name);
                            let func = crate::function::WalrusFunction::Vm(
                                crate::function::VmFunction::new(
                                    format!("{}::{}", name, method_name),
                                    arg_len,
                                    emitter.instruction_set(),
                                ),
                            );

                            struct_def.add_method(method_key, func);
                        }
                        _ => {
                            return Err(WalrusError::TodoError {
                                message: format!("Unexpected struct member type: {}", member_kind),
                            });
                        }
                    }
                }

                // Push the struct definition to the heap
                let struct_value = self
                    .instructions
                    .get_heap_mut()
                    .push(HeapValue::StructDef(struct_def));

                // Load the struct definition constant
                let index = self.instructions.push_constant(struct_value);
                let opcode = match index {
                    0 => Opcode::LoadConst0,
                    1 => Opcode::LoadConst1,
                    _ => Opcode::LoadConst(index),
                };
                self.instructions.push(Instruction::new(opcode, span));

                // Store the struct definition in the appropriate scope
                if is_global {
                    let struct_index = struct_global_index.unwrap();
                    self.instructions
                        .push(Instruction::new(Opcode::StoreGlobal(struct_index), span));
                } else {
                    self.define_variable(name, span);
                }
            }
            NodeKind::StructFunctionDefinition(name, _, _) => {
                return Err(WalrusError::TodoError {
                    message: format!(
                        "Struct function '{}' should only appear inside struct definitions",
                        name
                    ),
                });
            }
            NodeKind::MemberAccess(object, member) => {
                // Emit the object (should be a struct definition or instance)
                self.emit(*object)?;

                // Push the member name as a string constant
                let member_str = self
                    .instructions
                    .get_heap_mut()
                    .push(HeapValue::String(&member));
                let index = self.instructions.push_constant(member_str);
                let opcode = match index {
                    0 => Opcode::LoadConst0,
                    1 => Opcode::LoadConst1,
                    _ => Opcode::LoadConst(index),
                };
                self.instructions.push(Instruction::new(opcode, span));

                // Emit GetMethod opcode to retrieve the method from the struct
                self.instructions
                    .push(Instruction::new(Opcode::GetMethod, span));
            }
            NodeKind::IndexAssign(object, index, value) => {
                if let NodeKind::Ident(name) = object.kind() {
                    if let Some(local_index) = self.instructions.resolve_local_index(name) {
                        if let Some((index_local, add_one)) = self.local_int_index_pattern(&index) {
                            self.emit(*value)?;
                            let opcode = if add_one {
                                Opcode::StoreIndexLocalLocalAdd1(local_index as u32, index_local)
                            } else {
                                Opcode::StoreIndexLocalLocal(local_index as u32, index_local)
                            };
                            self.instructions.push(Instruction::new(opcode, span));
                            return Ok(());
                        }

                        self.emit(*index)?;
                        self.emit(*value)?;
                        self.instructions.push(Instruction::new(
                            Opcode::StoreIndexLocal(local_index as u32),
                            span,
                        ));
                        return Ok(());
                    }
                }

                // Emit the object to index into
                self.emit(*object)?;
                // Emit the index
                self.emit(*index)?;
                // Emit the value to store
                self.emit(*value)?;
                // Emit the StoreIndex opcode (pops value, index, object; performs assignment)
                self.instructions
                    .push(Instruction::new(Opcode::StoreIndex, span));
            }
            NodeKind::Throw(value) => {
                self.emit(*value)?;
                self.instructions
                    .push(Instruction::new(Opcode::Throw, span));
            }
            NodeKind::Try(try_block, catch_var, catch_block) => {
                // Install an exception handler for the lexical try range.
                // The catch target is patched once the try block has been emitted.
                let handler_addr = self.instructions.len();
                self.instructions
                    .push(Instruction::new(Opcode::PushExceptionHandler(0), span));

                // Emit try body.
                self.emit(*try_block)?;

                // Normal path: remove handler and skip catch block.
                self.instructions
                    .push(Instruction::new(Opcode::PopExceptionHandler, span));
                let jump_over_catch = self.instructions.len();
                self.instructions
                    .push(Instruction::new(Opcode::Jump(0), span));

                // Exception path starts here.
                let catch_ip = self.instructions.len();
                self.instructions.set(
                    handler_addr,
                    Instruction::new(Opcode::PushExceptionHandler(catch_ip as u32), span),
                );

                // Catch variable is scoped to the catch body.
                self.inc_depth();
                self.define_variable(catch_var, span);
                self.emit(*catch_block)?;
                self.dec_depth(span);

                let end_ip = self.instructions.len();
                self.instructions.set(
                    jump_over_catch,
                    Instruction::new(Opcode::Jump(end_ip as u32), span),
                );
            }
            NodeKind::ModuleImport(module_name, alias) => {
                // Push the module name as a string constant
                let name_val = self
                    .instructions
                    .get_heap_mut()
                    .push(HeapValue::String(&module_name));
                let index = self.instructions.push_constant(name_val);
                let opcode = match index {
                    0 => Opcode::LoadConst0,
                    1 => Opcode::LoadConst1,
                    _ => Opcode::LoadConst(index),
                };
                self.instructions.push(Instruction::new(opcode, span));

                // Emit the Import opcode
                self.instructions
                    .push(Instruction::new(Opcode::Import, span));

                // Store the module dict in a variable
                let var_name = alias.unwrap_or_else(|| default_import_alias(&module_name));

                if self.depth == 0 {
                    self.define_global_variable(var_name, span);
                } else {
                    self.define_variable(var_name, span);
                }
            }
            NodeKind::PackageImport(package_name, alias) => {
                // Lower package imports to "@package" module spec strings.
                let module_spec = format!("@{package_name}");
                let name_val = self
                    .instructions
                    .get_heap_mut()
                    .push(HeapValue::String(&module_spec));
                let index = self.instructions.push_constant(name_val);
                let opcode = match index {
                    0 => Opcode::LoadConst0,
                    1 => Opcode::LoadConst1,
                    _ => Opcode::LoadConst(index),
                };
                self.instructions.push(Instruction::new(opcode, span));
                self.instructions
                    .push(Instruction::new(Opcode::Import, span));

                let var_name = alias.unwrap_or(package_name);
                if self.depth == 0 {
                    self.define_global_variable(var_name, span);
                } else {
                    self.define_variable(var_name, span);
                }
            }
            _ => unimplemented!("{}", kind),
        }

        Ok(())
    }

    /// Pre-register top-level declarations (functions, structs) for forward references.
    /// This is called in Pass 1 before the main compilation Pass 2.
    fn pre_register_declarations(&mut self, node: &Node) {
        match node.kind() {
            NodeKind::FunctionDefinition(name, args, body) => {
                // Only pre-register at global scope
                if self.depth == 0 {
                    self.instructions.push_global(name.clone());
                    if self.function_guarantees_int_return(name, args, body) {
                        self.known_int_functions.insert(name.clone());
                        if self.function_is_pure_int_ast(name, args, body) {
                            self.known_pure_int_functions.insert(name.clone());
                        }
                    }
                    if self.function_is_pure_cloneable_ast(name, args, body)
                        && self.has_multiple_recursive_calls(body, name)
                    {
                        self.known_pure_cloneable_functions.insert(name.clone());
                    }
                }
            }
            NodeKind::AsyncFunctionDefinition(name, _, _) => {
                // Reserve the name so references resolve consistently before
                // async code generation lands.
                if self.depth == 0 {
                    self.instructions.push_global(name.clone());
                }
            }
            NodeKind::StructDefinition(name, _) => {
                // Structs are also global declarations
                if self.depth == 0 {
                    self.instructions.push_global(name.clone());
                }
            }
            _ => {
                // Other node types don't need pre-registration
            }
        }
    }

    fn inc_depth(&mut self) {
        self.instructions.inc_depth();
        self.known_int_scopes.push(FxHashSet::default());
    }

    fn dec_depth(&mut self, span: Span) {
        let popped = self.instructions.dec_depth();
        if self.known_int_scopes.len() > 1 {
            self.known_int_scopes.pop();
        }

        if popped > 0 {
            self.instructions
                .push(Instruction::new(Opcode::PopLocal(popped as u32), span));
        }
    }

    fn define_variable(&mut self, name: String, span: Span) {
        let index = self.instructions.push_local(name);

        self.instructions
            .push(Instruction::new(Self::local_store_opcode(index), span));
    }

    fn define_global_variable(&mut self, name: String, span: Span) {
        let index = self.instructions.push_global(name);

        self.instructions
            .push(Instruction::new(Opcode::StoreGlobal(index), span));
    }

    fn define_parameter(&mut self, name: String) {
        // Define parameter in locals without emitting Store opcode
        // The value will already be in locals when the function is called
        self.instructions.push_local(name);
    }

    fn emit_named_function_definition(
        &mut self,
        name: String,
        args: Vec<String>,
        body: Node,
        span: Span,
        is_async: bool,
    ) -> WalrusResult<()> {
        let is_global = self.depth == 0;

        // Get the function index - use pre-registered index if it exists (forward declaration),
        // otherwise register it now.
        let func_index = if is_global {
            if let Some(index) = self.instructions.resolve_global_index(&name) {
                index as u32
            } else {
                self.instructions.push_global(name.clone())
            }
        } else {
            self.instructions.push_local(name.clone())
        };

        let mut emitter = self.new_child();
        let arg_len = args.len();
        emitter.current_function_name = Some(name.clone());
        emitter.current_function_arity = Some(arg_len);
        let int_params = Self::int_parameter_names(&args, &body);

        for arg in args {
            let is_int = int_params.contains(&arg);
            emitter.define_parameter(arg.clone());
            if is_int {
                emitter.mark_known_int(&arg, false, true);
            }
        }

        emitter.emit(body)?;
        emitter.emit_void(span);
        emitter.emit_return(span);

        let func_instructions = emitter.instruction_set();
        if is_global
            && self.known_int_functions.contains(&name)
            && self.instruction_set_is_pure_int(&name, &func_instructions)
        {
            self.known_pure_int_functions.insert(name.clone());
        }
        self.instructions
            .register_function(name.clone(), 0, arg_len);

        let vm_function = if is_async {
            VmFunction::new_async(name.clone(), arg_len, func_instructions)
        } else {
            VmFunction::new(name.clone(), arg_len, func_instructions)
        };
        let func = self
            .instructions
            .get_heap_mut()
            .push(HeapValue::Function(WalrusFunction::Vm(vm_function)));

        let index = self.instructions.push_constant(func);
        let opcode = match index {
            0 => Opcode::LoadConst0,
            1 => Opcode::LoadConst1,
            _ => Opcode::LoadConst(index),
        };
        self.instructions.push(Instruction::new(opcode, span));

        if is_global {
            self.instructions
                .push(Instruction::new(Opcode::StoreGlobal(func_index), span));
        } else {
            self.instructions
                .push(Instruction::new(Self::local_store_opcode(func_index), span));
        }

        Ok(())
    }

    fn emit_anon_function_definition(
        &mut self,
        args: Vec<String>,
        body: Node,
        span: Span,
        is_async: bool,
    ) -> WalrusResult<()> {
        let mut emitter = self.new_child();
        let arg_len = args.len();

        for arg in args {
            emitter.define_parameter(arg);
        }

        emitter.emit(body)?;
        emitter.emit_void(span);
        emitter.emit_return(span);

        let name = format!("[{:p}]", &emitter.instructions);
        let vm_function = if is_async {
            VmFunction::new_async(name, arg_len, emitter.instruction_set())
        } else {
            VmFunction::new(name, arg_len, emitter.instruction_set())
        };
        let func = self
            .instructions
            .get_heap_mut()
            .push(HeapValue::Function(WalrusFunction::Vm(vm_function)));

        let index = self.instructions.push_constant(func);
        let opcode = match index {
            0 => Opcode::LoadConst0,
            1 => Opcode::LoadConst1,
            _ => Opcode::LoadConst(index),
        };
        self.instructions.push(Instruction::new(opcode, span));

        Ok(())
    }

    /// Emit bytecode for a function call.
    /// If `is_tail_call` is true, emits TailCall opcode (for tail call optimization).
    /// Otherwise emits regular Call opcode.
    fn emit_function_call(
        &mut self,
        func: Box<Node>,
        args: Vec<Node>,
        span: Span,
        is_tail_call: bool,
    ) -> WalrusResult<()> {
        // Check if this is a method call (e.g., arr.push(x) or Calculator.add(a, b))
        if let NodeKind::MemberAccess(object, method_name) = func.kind() {
            let arg_len = args.len();

            if method_name == "push" && arg_len == 1 && !is_tail_call {
                self.emit(*object.clone())?;
                self.emit(args.into_iter().next().expect("push has exactly one arg"))?;
                self.instructions
                    .push(Instruction::new(Opcode::ListPush, span));
                return Ok(());
            }

            // Emit the object first
            self.emit(*object.clone())?;

            // Emit all arguments
            for arg in args {
                self.emit(arg)?;
            }

            // Push the method name as a string constant
            let method_str = self
                .instructions
                .get_heap_mut()
                .push(HeapValue::String(method_name));
            let index = self.instructions.push_constant(method_str);
            let opcode = match index {
                0 => Opcode::LoadConst0,
                1 => Opcode::LoadConst1,
                _ => Opcode::LoadConst(index),
            };
            self.instructions.push(Instruction::new(opcode, span));

            // Emit CallMethod with argument count
            self.instructions
                .push(Instruction::new(Opcode::CallMethod(arg_len as u32), span));

            // Method calls can't be tail-optimized (for now)
            if is_tail_call {
                self.instructions
                    .push(Instruction::new(Opcode::Return, span));
            }
            return Ok(());
        }

        if !is_tail_call {
            if let NodeKind::Ident(name) = func.kind() {
                if self.instructions.resolve_local_index(name).is_none() {
                    if self.current_function_name.as_deref() == Some(name)
                        && self.current_function_arity == Some(args.len())
                    {
                        let arg_len = args.len();

                        if arg_len == 1 {
                            if let Some((local_index, const_index)) =
                                self.local_const_index_pattern(&args[0])
                            {
                                self.instructions.push(Instruction::new(
                                    Opcode::CallSelfIndexLocalConst1(local_index, const_index),
                                    span,
                                ));
                                return Ok(());
                            }
                        }

                        for arg in args {
                            self.emit(arg)?;
                        }

                        let opcode = if arg_len == 1 {
                            if self.known_pure_int_functions.contains(name) {
                                Opcode::CallMemoizedSelf1
                            } else if self.known_pure_cloneable_functions.contains(name) {
                                Opcode::CallMemoizedCloneSelf1
                            } else {
                                Opcode::CallSelf1
                            }
                        } else {
                            Opcode::CallSelf(arg_len as u32)
                        };
                        self.instructions.push(Instruction::new(opcode, span));
                        return Ok(());
                    }

                    if let Some(index) = self.instructions.resolve_global_index(name) {
                        let arg_len = args.len();

                        if arg_len == 1
                            && !self.loop_stack.is_empty()
                            && self.known_pure_int_functions.contains(name)
                            && optimize::try_get_constant(&args[0]).is_some()
                        {
                            self.emit(args.into_iter().next().expect("arg_len checked above"))?;
                            self.instructions.push(Instruction::new(
                                Opcode::CallPureGlobal1(index as u32),
                                span,
                            ));
                            return Ok(());
                        }

                        for arg in args {
                            self.emit(arg)?;
                        }

                        let opcode = if arg_len == 1 {
                            Opcode::CallGlobal1(index as u32)
                        } else {
                            Opcode::CallGlobal(index as u32, arg_len as u32)
                        };
                        self.instructions.push(Instruction::new(opcode, span));
                        return Ok(());
                    }
                }
            }
        }

        // Regular or tail function call
        let arg_len = args.len();

        for arg in args {
            self.emit(arg)?;
        }

        self.emit(*func)?;

        if is_tail_call {
            self.instructions
                .push(Instruction::new(Opcode::TailCall(arg_len as u32), span));
        } else {
            self.instructions
                .push(Instruction::new(Opcode::Call(arg_len as u32), span));
        }

        Ok(())
    }

    pub fn instruction_set(self) -> InstructionSet {
        self.instructions.disassemble();
        self.instructions
    }

    pub fn emit_void(&mut self, span: Span) {
        self.instructions.push(Instruction::new(Opcode::Void, span));
    }

    pub fn emit_return(&mut self, span: Span) {
        self.instructions
            .push(Instruction::new(Opcode::Return, span));
    }

    /// Emit a constant value (used by constant folding).
    fn emit_constant(&mut self, value: Value, span: Span) {
        match value {
            Value::Bool(true) => {
                self.instructions.push(Instruction::new(Opcode::True, span));
            }
            Value::Bool(false) => {
                self.instructions
                    .push(Instruction::new(Opcode::False, span));
            }
            _ => {
                let index = self.instructions.push_constant(value);
                let opcode = match index {
                    0 => Opcode::LoadConst0,
                    1 => Opcode::LoadConst1,
                    _ => Opcode::LoadConst(index),
                };
                self.instructions.push(Instruction::new(opcode, span));
            }
        }
    }

    /// Build debug information for the instruction set.
    /// This maps instruction pointers to source line numbers and stores variable names.
    pub fn build_debug_info(&mut self) {
        use crate::vm::instruction_set::{DebugInfo, LineTable};

        let source = self.source_ref.source();

        // Precompute line offsets (byte offset where each line starts)
        let line_offsets: Vec<usize> = std::iter::once(0)
            .chain(source.match_indices('\n').map(|(i, _)| i + 1))
            .collect();

        // Helper to convert byte offset to line number (1-indexed)
        let byte_to_line = |byte_offset: usize| -> usize {
            match line_offsets.binary_search(&byte_offset) {
                Ok(i) => i + 1, // Exact match at start of line
                Err(i) => i,    // Between lines, return the line we're in
            }
        };

        // Build line table by walking instructions
        let mut line_table = LineTable::new();
        let mut current_line: Option<usize> = None;
        let mut line_start_ip: usize = 0;

        for (ip, instr) in self.instructions.instructions.iter().enumerate() {
            let span = instr.span();
            let line = byte_to_line(span.0);

            match current_line {
                None => {
                    current_line = Some(line);
                    line_start_ip = ip;
                }
                Some(prev_line) if prev_line != line => {
                    // New line, close the previous entry
                    line_table.add_entry(prev_line, line_start_ip, ip);
                    current_line = Some(line);
                    line_start_ip = ip;
                }
                _ => {} // Same line, continue
            }
        }

        // Close final line entry
        if let Some(line) = current_line {
            line_table.add_entry(line, line_start_ip, self.instructions.instructions.len());
        }

        // Copy variable names from symbol tables
        let local_names = self.instructions.locals.get_all_names();
        let global_names = self.instructions.globals.get_all_names();

        self.instructions.debug_info = Some(DebugInfo {
            local_names,
            global_names,
            line_table,
            source: source.to_string(),
        });
    }
}

fn default_import_alias(module_name: &str) -> String {
    let tail = module_name.rsplit('/').next().unwrap_or(module_name);
    tail.strip_suffix(".walrus").unwrap_or(tail).to_string()
}
