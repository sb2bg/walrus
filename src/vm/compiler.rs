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
        _ => None,
    }
}

pub struct BytecodeEmitter<'a> {
    instructions: InstructionSet,
    source_ref: SourceRef<'a>,
    depth: usize,                   // 0 = global scope, >0 = local scope
    loop_stack: Vec<LoopContext>,   // Track nested loops for break/continue
    current_struct: Option<String>, // Name of struct currently being compiled (for method access)
}

struct LoopContext {
    start: usize,       // Address of loop start (for continue)
    breaks: Vec<usize>, // Addresses of break jumps to patch
    is_for_loop: bool,  // True if this is a for loop (needs to pop iterator on break)
}

impl<'a> BytecodeEmitter<'a> {
    pub fn new(source_ref: SourceRef<'a>) -> Self {
        Self {
            instructions: InstructionSet::new(),
            source_ref,
            depth: 0, // Start at global scope
            loop_stack: Vec::new(),
            current_struct: None,
        }
    }

    fn new_child(&self) -> Self {
        Self {
            instructions: InstructionSet::new_child_with_globals(self.instructions.globals.clone()),
            source_ref: self.source_ref,
            depth: 1,               // Functions start at local scope depth 1
            loop_stack: Vec::new(), // Functions get their own loop stack
            current_struct: self.current_struct.clone(),
        }
    }

    pub fn emit(&mut self, node: Node) -> WalrusResult<()> {
        let kind = node.kind().to_string();
        let span = *node.span();

        match node.into_kind() {
            NodeKind::Program(nodes) => {
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

                // Push loop context
                self.loop_stack.push(LoopContext {
                    start,
                    breaks: Vec::new(),
                    is_for_loop: false,
                });

                self.emit(*cond)?;

                let jump = self.instructions.len();

                self.instructions
                    .push(Instruction::new(Opcode::JumpIfFalse(0), span));

                self.emit(*body)?;

                self.instructions
                    .push(Instruction::new(Opcode::Jump(start as u32), span));

                self.instructions.set(
                    jump,
                    Instruction::new(Opcode::JumpIfFalse(self.instructions.len() as u32), span),
                );

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
                self.emit(*iter)?;

                self.instructions
                    .push(Instruction::new(Opcode::GetIter, span));

                let jump = self.instructions.len();

                self.instructions
                    .push(Instruction::new(Opcode::IterNext(0), span));

                // Push loop context (continue jumps back to the IterNext instruction)
                self.loop_stack.push(LoopContext {
                    start: jump,
                    breaks: Vec::new(),
                    is_for_loop: true,
                });

                self.inc_depth();

                let index = self.instructions.push_local(name);

                self.instructions
                    .push(Instruction::new(Opcode::StoreAt(index), span));

                self.emit(*body)?;

                // Pop locals declared in the loop body (but not the loop variable itself)
                // This is crucial: without this, variables declared inside the loop body
                // would reuse their slots on subsequent iterations, seeing stale values
                self.dec_depth(span);

                self.instructions
                    .push(Instruction::new(Opcode::Jump(jump as u32), span));

                self.instructions.set(
                    jump,
                    Instruction::new(Opcode::IterNext(self.instructions.len() as u32), span),
                );

                // Pop loop context and patch break jumps
                if let Some(loop_ctx) = self.loop_stack.pop() {
                    let end = self.instructions.len();
                    for break_addr in loop_ctx.breaks {
                        self.instructions
                            .set(break_addr, Instruction::new(Opcode::Jump(end as u32), span));
                    }
                }
            }
            NodeKind::FunctionDefinition(name, args, body) => {
                let is_global = self.depth == 0;

                // Pre-register the function name
                let func_index = if is_global {
                    self.instructions.push_global(name.clone())
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

                // Create the function heap value
                let func =
                    self.instructions
                        .get_heap_mut()
                        .push(HeapValue::Function(WalrusFunction::Vm(VmFunction::new(
                            name.clone(),
                            arg_len,
                            emitter.instruction_set(),
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
                // Check if this is a builtin function call
                if let NodeKind::Ident(name) = func.kind() {
                    if let Some(builtin) = get_builtin(name) {
                        // Emit builtin opcode
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
                        return Ok(());
                    }
                }

                // Regular function call
                let arg_len = args.len();

                for arg in args {
                    self.emit(arg)?;
                }

                self.emit(*func)?;

                self.instructions
                    .push(Instruction::new(Opcode::Call(arg_len as u32), span));
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
                    // If we're inside a struct method and the identifier isn't found,
                    // try to resolve it as a method of the current struct
                    // This will load the struct and then get the method

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
                self.inc_depth();

                for node in nodes {
                    self.emit(node)?;
                }

                self.dec_depth(span);
            }
            NodeKind::UnscopedStatements(nodes) => {
                for node in nodes {
                    self.emit(node)?;
                }
            }
            NodeKind::Return(node) => {
                self.emit(*node)?;

                self.instructions
                    .push(Instruction::new(Opcode::Return, span));
            }
            NodeKind::Break => {
                if let Some(loop_ctx) = self.loop_stack.last_mut() {
                    // For for-loops, pop the iterator from the stack before breaking
                    if loop_ctx.is_for_loop {
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

                // Pre-register the struct name in globals so methods can reference it
                let struct_global_index = if is_global {
                    Some(self.instructions.push_global(name.clone()))
                } else {
                    None
                };

                // Create a new struct definition
                let mut struct_def = crate::structs::StructDefinition::new(name.clone());

                // Process struct members (methods)
                for member in members {
                    let member_kind = member.kind().to_string();
                    match member.into_kind() {
                        NodeKind::StructFunctionDefinition(method_name, args, body) => {
                            // Create a child emitter for the method with struct context
                            let mut emitter = self.new_child();
                            emitter.current_struct = Some(name.clone());
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
            _ => unimplemented!("{}", kind),
        }

        Ok(())
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
}
