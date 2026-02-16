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
use crate::WalrusResult;
use rustc_hash::FxHashSet;

// Builtin function metadata
struct BuiltinInfo {
    opcode: Opcode,
    arity: usize,
}

// Registry of builtin functions
fn get_builtin(name: &str) -> Option<BuiltinInfo> {
    match name {
        "len" => Some(BuiltinInfo {
            opcode: Opcode::Len,
            arity: 1,
        }),
        "str" => Some(BuiltinInfo {
            opcode: Opcode::Str,
            arity: 1,
        }),
        "type" => Some(BuiltinInfo {
            opcode: Opcode::Type,
            arity: 1,
        }),
        "__gc__" => Some(BuiltinInfo {
            opcode: Opcode::Gc,
            arity: 0,
        }),
        "__heap_stats__" => Some(BuiltinInfo {
            opcode: Opcode::HeapStats,
            arity: 0,
        }),
        "__gc_threshold__" => Some(BuiltinInfo {
            opcode: Opcode::GcConfig,
            arity: 1,
        }),
        _ => None,
    }
}

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
}

struct LoopContext {
    start: usize,             // Address of loop start (for continue)
    breaks: Vec<usize>,       // Addresses of break jumps to patch
    has_stack_iterator: bool, // True if loop has an iterator on the operand stack (iterator-based for loops only)
    locals_at_start: usize,   // Number of locals at loop body start (for continue cleanup)
}

impl<'a> BytecodeEmitter<'a> {
    pub fn new(source_ref: SourceRef<'a>) -> Self {
        Self {
            instructions: InstructionSet::new(),
            source_ref,
            depth: 0, // Start at global scope
            loop_stack: Vec::new(),
            current_struct: None,
            current_struct_methods: None,
        }
    }

    fn new_child(&self) -> Self {
        Self {
            instructions: InstructionSet::new_child_with_globals(self.instructions.globals.clone()),
            source_ref: self.source_ref,
            depth: 1,               // Functions start at local scope depth 1
            loop_stack: Vec::new(), // Functions get their own loop stack
            current_struct: self.current_struct.clone(),
            current_struct_methods: self.current_struct_methods.clone(),
        }
    }

    pub fn emit(&mut self, node: Node) -> WalrusResult<()> {
        let kind = node.kind().to_string();
        let span = *node.span();

        match node.into_kind() {
            NodeKind::Program(nodes) => {
                // Two-pass compilation for forward declarations:
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
                            // Convert to string using builtin str function
                            self.instructions.push(Instruction::new(Opcode::Str, span));
                            part_count += 1;
                        }
                    }
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
                        // Normal binary operations
                        self.emit(*left)?;
                        self.emit(*right)?;
                        self.instructions.push(Instruction::new(op, span));
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
                self.emit(*cond)?;

                let jump = self.instructions.len();

                self.instructions
                    .push(Instruction::new(Opcode::JumpIfFalse(0), span));

                self.emit(*then)?;

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

                    // The loop variable slot (also outside depth scope for simplicity)
                    let var_idx = self.instructions.push_local(name.clone());

                    let jump = self.instructions.len();

                    // Placeholder for ForRangeNext - will be patched
                    self.instructions
                        .push(Instruction::new(Opcode::ForRangeNext(0, range_idx), span));

                    // Store the value into the loop variable
                    self.instructions
                        .push(Instruction::new(Opcode::StoreAt(var_idx), span));

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

                    // Patch the ForRangeNext to jump past the loop AND pop 3 locals (range_start, range_end, var)
                    let end_pos = self.instructions.len();
                    self.instructions.set(
                        jump,
                        Instruction::new(Opcode::ForRangeNext(end_pos as u32, range_idx), span),
                    );

                    // JIT: Register the range-based for loop for hot-spot detection
                    // The header is at 'jump' where ForRangeNext is
                    self.instructions
                        .register_loop(jump, back_edge, end_pos, true);

                    // Pop the 3 range locals at the end
                    self.instructions
                        .push(Instruction::new(Opcode::PopLocal(3), span));

                    // Patch break jumps (they need to jump past the PopLocal too)
                    if let Some(loop_ctx) = loop_ctx {
                        let end = self.instructions.len();
                        for break_addr in loop_ctx.breaks {
                            self.instructions
                                .set(break_addr, Instruction::new(Opcode::Jump(end as u32), span));
                        }
                    }

                    // Remove the 3 locals from symbol table
                    self.instructions.pop_locals(3);
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
                        .push(Instruction::new(Opcode::StoreAt(index), span));

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
                let is_global = self.depth == 0;

                // Get the function index - use pre-registered index if it exists (forward declaration),
                // otherwise register it now
                let func_index = if is_global {
                    if let Some(index) = self.instructions.resolve_global_index(&name) {
                        // Already pre-registered in Pass 1
                        index as u32
                    } else {
                        // Not pre-registered (shouldn't happen for top-level, but handle it)
                        self.instructions.push_global(name.clone())
                    }
                } else {
                    self.instructions.push_local(name.clone())
                };

                // Create a child emitter - functions are always local scope
                let mut emitter = self.new_child();
                let arg_len = args.len();

                // Define function parameters as locals in the function scope
                // Don't emit Store opcodes - values will already be in locals when called
                for arg in args {
                    emitter.define_parameter(arg);
                }

                emitter.emit(*body)?;

                // Add implicit void return if the function doesn't end with an explicit return
                emitter.emit_void(span);
                emitter.emit_return(span);

                // Get the compiled function's instruction set
                let func_instructions = emitter.instruction_set();

                // JIT: Register the function for hot-spot detection
                // Note: We register it in the parent instruction set since that's where calls happen
                self.instructions
                    .register_function(name.clone(), 0, arg_len);

                // Create the function heap value
                let func =
                    self.instructions
                        .get_heap_mut()
                        .push(HeapValue::Function(WalrusFunction::Vm(VmFunction::new(
                            name.clone(),
                            arg_len,
                            func_instructions,
                        ))));

                // Load the function constant and store it
                let index = self.instructions.push_constant(func);
                let opcode = match index {
                    0 => Opcode::LoadConst0,
                    1 => Opcode::LoadConst1,
                    _ => Opcode::LoadConst(index),
                };
                self.instructions.push(Instruction::new(opcode, span));

                if is_global {
                    // Top-level function definitions go into globals
                    self.instructions
                        .push(Instruction::new(Opcode::StoreGlobal(func_index), span));
                } else {
                    // Nested function definitions go into locals
                    self.instructions
                        .push(Instruction::new(Opcode::StoreAt(func_index), span));
                }
            }
            NodeKind::AnonFunctionDefinition(args, body) => {
                // Create a child emitter - functions are always local scope
                let mut emitter = self.new_child();
                let arg_len = args.len();

                // Define function parameters as locals in the function scope
                // Don't emit Store opcodes - values will already be in locals when called
                for arg in args {
                    emitter.define_parameter(arg);
                }

                emitter.emit(*body)?;

                // Add implicit void return if the function doesn't end with an explicit return
                emitter.emit_void(span);
                emitter.emit_return(span);

                // TODO: Should this include the arity?
                let name = format!("[{:p}]", &emitter.instructions);

                // Create the function heap value
                let func =
                    self.instructions
                        .get_heap_mut()
                        .push(HeapValue::Function(WalrusFunction::Vm(VmFunction::new(
                            name,
                            arg_len,
                            emitter.instruction_set(),
                        ))));

                // Load the function constant
                let index = self.instructions.push_constant(func);
                let opcode = match index {
                    0 => Opcode::LoadConst0,
                    1 => Opcode::LoadConst1,
                    _ => Opcode::LoadConst(index),
                };
                self.instructions.push(Instruction::new(opcode, span));
            }
            NodeKind::FunctionCall(func, args) => {
                self.emit_function_call(func, args, span, false)?;
            }
            NodeKind::Index(node, index) => {
                self.emit(*node)?;
                self.emit(*index)?;

                self.instructions
                    .push(Instruction::new(Opcode::Index, span));
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
                    let opcode = match index {
                        0 => Opcode::LoadLocal0,
                        1 => Opcode::LoadLocal1,
                        2 => Opcode::LoadLocal2,
                        3 => Opcode::LoadLocal3,
                        _ => Opcode::Load(index as u32),
                    };
                    self.instructions.push(Instruction::new(opcode, span));
                } else if let Some(index) = self.instructions.resolve_global_index(&name) {
                    let opcode = match index {
                        0 => Opcode::LoadGlobal0,
                        1 => Opcode::LoadGlobal1,
                        2 => Opcode::LoadGlobal2,
                        3 => Opcode::LoadGlobal3,
                        _ => Opcode::LoadGlobal(index as u32),
                    };
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
                        let opcode = match struct_index {
                            0 => Opcode::LoadGlobal0,
                            1 => Opcode::LoadGlobal1,
                            2 => Opcode::LoadGlobal2,
                            3 => Opcode::LoadGlobal3,
                            _ => Opcode::LoadGlobal(struct_index as u32),
                        };
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
                let is_global = self.depth == 0;

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
                    self.define_global_variable(name, span);
                } else {
                    self.define_variable(name, span);
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
                            return Ok(());
                        }
                        optimize::ReassignOptimization::Decrement => {
                            self.instructions
                                .push(Instruction::new(Opcode::DecrementLocal(index), span));
                            return Ok(());
                        }
                        optimize::ReassignOptimization::None => {}
                    }
                }

                // For compound assignments (+=, -=, etc.), we need to load the current value first
                match op {
                    Opcode::Add
                    | Opcode::Subtract
                    | Opcode::Multiply
                    | Opcode::Divide
                    | Opcode::Modulo => {
                        // Load current value
                        if is_global {
                            self.instructions
                                .push(Instruction::new(Opcode::LoadGlobal(index), span));
                        } else {
                            self.instructions
                                .push(Instruction::new(Opcode::Load(index), span));
                        }
                        // Emit the right-hand side
                        self.emit(*node)?;
                        // Perform the operation
                        self.instructions.push(Instruction::new(op, span));
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
                } else {
                    self.instructions
                        .push(Instruction::new(Opcode::Reassign(index), span));
                }
            }
            NodeKind::Statements(nodes) => {
                // Two-pass compilation for forward declarations at global scope:
                if self.depth == 0 {
                    // Pass 1: Pre-register all top-level function and struct names
                    for node in &nodes {
                        self.pre_register_declarations(node);
                    }

                    // Pass 2a: Compile function and struct definitions first (hoisting)
                    for node in &nodes {
                        match node.kind() {
                            NodeKind::FunctionDefinition(_, _, _)
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
                    // Don't optimize builtin function calls - they don't use call frames
                    let is_builtin = if let NodeKind::Ident(name) = func.kind() {
                        get_builtin(name).is_some()
                    } else {
                        false
                    };

                    if !is_builtin {
                        // This is a tail call - emit TailCall instead of Call + Return
                        self.emit_function_call(func, args, *node.span(), true)?;
                        return Ok(());
                    }
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

                            // Define method parameters as locals
                            for arg in args {
                                emitter.define_parameter(arg);
                            }

                            emitter.emit(*body)?;

                            // Add implicit void return if the method doesn't end with an explicit return
                            emitter.emit_void(span);
                            emitter.emit_return(span);

                            // Create the method function
                            let func = crate::function::WalrusFunction::Vm(
                                crate::function::VmFunction::new(
                                    format!("{}::{}", name, method_name),
                                    arg_len,
                                    emitter.instruction_set(),
                                ),
                            );

                            struct_def.add_method(method_name, func);
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
                let var_name = alias.unwrap_or_else(|| {
                    // Extract last component of module path as default name
                    // e.g., "std/io" -> "io"
                    module_name
                        .rsplit('/')
                        .next()
                        .unwrap_or(&module_name)
                        .to_string()
                });

                if self.depth == 0 {
                    self.define_global_variable(var_name, span);
                } else {
                    self.define_variable(var_name, span);
                }
            }
            NodeKind::PackageImport(_, _) => {
                // Package imports not yet implemented
                return Err(WalrusError::PackageImportNotImplemented {
                    span,
                    src: self.source_ref.source().to_string(),
                    filename: self.source_ref.filename().to_string(),
                });
            }
            _ => unimplemented!("{}", kind),
        }

        Ok(())
    }

    /// Pre-register top-level declarations (functions, structs) for forward references.
    /// This is called in Pass 1 before the main compilation Pass 2.
    fn pre_register_declarations(&mut self, node: &Node) {
        match node.kind() {
            NodeKind::FunctionDefinition(name, _, _) => {
                // Only pre-register at global scope
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
    }

    fn dec_depth(&mut self, span: Span) {
        let popped = self.instructions.dec_depth();

        if popped > 0 {
            self.instructions
                .push(Instruction::new(Opcode::PopLocal(popped as u32), span));
        }
    }

    fn define_variable(&mut self, name: String, span: Span) {
        self.instructions.push_local(name);

        self.instructions
            .push(Instruction::new(Opcode::Store, span));
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
        // Check if this is a builtin function call
        if let NodeKind::Ident(name) = func.kind() {
            if let Some(builtin) = get_builtin(name) {
                // Emit builtin opcode (never tail-optimized since they don't use frames)
                if args.len() != builtin.arity {
                    return Err(WalrusError::InvalidArgCount {
                        name: name.to_string(),
                        expected: builtin.arity,
                        got: args.len(),
                        span,
                        src: self.source_ref.source().to_string(),
                        filename: self.source_ref.filename().to_string(),
                    });
                }

                // Emit arguments
                for arg in args {
                    self.emit(arg)?;
                }

                // Emit the builtin opcode
                self.instructions
                    .push(Instruction::new(builtin.opcode, span));

                // If this was supposed to be a tail call, we still need to return
                if is_tail_call {
                    self.instructions
                        .push(Instruction::new(Opcode::Return, span));
                }
                return Ok(());
            }
        }

        // Check if this is a method call (e.g., arr.push(x) or Calculator.add(a, b))
        if let NodeKind::MemberAccess(object, method_name) = func.kind() {
            let arg_len = args.len();

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
