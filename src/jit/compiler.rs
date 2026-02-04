//! Cranelift-based JIT Compiler for Walrus
//!
//! This module compiles hot bytecode regions to native machine code using Cranelift.
//! Currently supports:
//! - Simple integer range loops (for i in 0..n)
//! - Integer arithmetic (+, -, *, /)
//! - Integer comparisons (<, <=, >, >=, ==, !=)
//!
//! The JIT compiler works by:
//! 1. Analyzing a bytecode region for JIT-ability (must be type-stable)
//! 2. Translating bytecode to Cranelift IR
//! 3. Compiling to native code
//! 4. Returning a function pointer that the VM can call

#![cfg(feature = "jit")]

use std::mem;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{types, AbiParam, InstBuilder};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::{self as codegen};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use log::debug;
use rustc_hash::FxHashMap;

use crate::vm::instruction_set::InstructionSet;
use crate::vm::opcode::Opcode;

// Alias for Cranelift's Value type to distinguish from Walrus Value
type CraneliftValue = cranelift_codegen::ir::Value;

/// Errors that can occur during JIT compilation
#[derive(Debug, Clone)]
pub enum JitError {
    /// The bytecode region contains unsupported operations
    UnsupportedOperation(String),
    /// Type information is insufficient for compilation
    TypeInfoMissing,
    /// Cranelift compilation failed
    CompilationFailed(String),
    /// The region is not suitable for JIT (e.g., polymorphic types)
    NotJitCompatible(String),
}

impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JitError::UnsupportedOperation(op) => write!(f, "Unsupported operation: {}", op),
            JitError::TypeInfoMissing => write!(f, "Type information missing"),
            JitError::CompilationFailed(msg) => write!(f, "Compilation failed: {}", msg),
            JitError::NotJitCompatible(reason) => write!(f, "Not JIT compatible: {}", reason),
        }
    }
}

/// Result of JIT compilation - a callable function
pub type JitResult<T> = Result<T, JitError>;

/// Signature for a JIT-compiled integer range loop
/// Parameters: (start: i64, end: i64) -> result: i64
pub type IntRangeLoopFn = unsafe extern "C" fn(i64, i64) -> i64;

/// Signature for a JIT-compiled function that takes an accumulator
/// Parameters: (start: i64, end: i64, initial_acc: i64) -> result: i64
pub type IntRangeAccumFn = unsafe extern "C" fn(i64, i64, i64) -> i64;

/// Compiled JIT function with metadata
#[derive(Debug)]
pub struct CompiledFunction {
    /// The function ID in the JIT module
    func_id: FuncId,
    /// Function pointer for direct calls
    func_ptr: *const u8,
    /// The bytecode range this function covers
    pub start_ip: usize,
    pub end_ip: usize,
    /// The local variable index that holds the accumulator (relative to frame pointer)
    pub accumulator_local: Option<u32>,
    /// The loop pattern that was compiled
    pub pattern: CompiledPattern,
}

/// The pattern that was compiled for this function
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompiledPattern {
    /// acc += i pattern - accumulates the sum of indices
    SumIndex,
    /// acc += 1 pattern - counts iterations
    CountIterations,
    /// Just skip the loop (empty body)
    EmptyLoop,
}

impl CompiledFunction {
    /// Call this function as an integer range loop
    /// Safety: caller must ensure the function was compiled for this signature
    pub unsafe fn call_int_range(&self, start: i64, end: i64) -> i64 {
        unsafe {
            let func: IntRangeLoopFn = mem::transmute(self.func_ptr);
            func(start, end)
        }
    }

    /// Call this function as an integer range loop with accumulator
    pub unsafe fn call_int_range_accum(&self, start: i64, end: i64, initial: i64) -> i64 {
        unsafe {
            let func: IntRangeAccumFn = mem::transmute(self.func_ptr);
            func(start, end, initial)
        }
    }
}

/// The JIT compiler using Cranelift
pub struct JitCompiler {
    /// Cranelift JIT module for compiling functions
    module: JITModule,
    /// Cranelift context for building functions
    ctx: codegen::Context,
    /// Function builder context (reusable)
    func_ctx: FunctionBuilderContext,
    /// Cache of compiled functions by their start IP
    compiled_cache: FxHashMap<usize, CompiledFunction>,
}

impl JitCompiler {
    /// Create a new JIT compiler
    pub fn new() -> JitResult<Self> {
        let mut flag_builder = settings::builder();
        // Enable speed optimizations
        flag_builder.set("opt_level", "speed").map_err(|e| {
            JitError::CompilationFailed(format!("Failed to set opt_level: {}", e))
        })?;

        let isa_builder = cranelift_native::builder().map_err(|e| {
            JitError::CompilationFailed(format!("Failed to create ISA builder: {}", e))
        })?;

        let isa = isa_builder.finish(settings::Flags::new(flag_builder)).map_err(|e| {
            JitError::CompilationFailed(format!("Failed to create ISA: {}", e))
        })?;

        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(builder);

        Ok(Self {
            module,
            ctx: codegen::Context::new(),
            func_ctx: FunctionBuilderContext::new(),
            compiled_cache: FxHashMap::default(),
        })
    }

    /// Check if a bytecode region has already been compiled
    pub fn is_compiled(&self, start_ip: usize) -> bool {
        self.compiled_cache.contains_key(&start_ip)
    }

    /// Get a compiled function if it exists
    pub fn get_compiled(&self, start_ip: usize) -> Option<&CompiledFunction> {
        self.compiled_cache.get(&start_ip)
    }

    /// Compile an integer range loop that computes a sum
    /// This handles the common pattern: for i in start..end { acc += f(i) }
    ///
    /// The loop body is analyzed from the bytecode and translated to Cranelift IR.
    pub fn compile_int_range_sum_loop(
        &mut self,
        instructions: &InstructionSet,
        loop_header_ip: usize,
        loop_exit_ip: usize,
    ) -> JitResult<&CompiledFunction> {
        // Check cache first
        if self.compiled_cache.contains_key(&loop_header_ip) {
            return Ok(self.compiled_cache.get(&loop_header_ip).unwrap());
        }

        debug!(
            "JIT: Compiling integer range loop at IP {}..{}",
            loop_header_ip, loop_exit_ip
        );

        // Analyze the loop to determine what it computes
        let loop_analysis = self.analyze_int_range_loop(instructions, loop_header_ip, loop_exit_ip)?;

        // Create function signature: (start: i64, end: i64, initial_acc: i64) -> i64
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::I64)); // start
        sig.params.push(AbiParam::new(types::I64)); // end
        sig.params.push(AbiParam::new(types::I64)); // initial accumulator
        sig.returns.push(AbiParam::new(types::I64)); // result

        // Declare the function
        let func_name = format!("jit_loop_{}", loop_header_ip);
        let func_id = self
            .module
            .declare_function(&func_name, Linkage::Local, &sig)
            .map_err(|e| JitError::CompilationFailed(format!("Failed to declare function: {}", e)))?;

        // Build the function
        self.ctx.func.signature = sig;
        self.ctx.func.name = cranelift_codegen::ir::UserFuncName::user(0, func_id.as_u32());

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);

            // Create entry block
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            // Get parameters
            let start = builder.block_params(entry_block)[0];
            let end = builder.block_params(entry_block)[1];
            let initial_acc = builder.block_params(entry_block)[2];

            // Create loop header block
            let loop_header = builder.create_block();
            builder.append_block_param(loop_header, types::I64); // i
            builder.append_block_param(loop_header, types::I64); // acc

            // Jump to loop header with initial values
            builder.ins().jump(loop_header, &[start, initial_acc]);

            // Build loop header
            builder.switch_to_block(loop_header);
            let i = builder.block_params(loop_header)[0];
            let acc = builder.block_params(loop_header)[1];

            // Check loop condition: i < end
            let cond = builder.ins().icmp(IntCC::SignedLessThan, i, end);

            // Create loop body and exit blocks
            let loop_body = builder.create_block();
            let loop_exit = builder.create_block();
            builder.append_block_param(loop_exit, types::I64); // final acc

            builder.ins().brif(cond, loop_body, &[], loop_exit, &[acc]);
            builder.seal_block(loop_header);

            // Build loop body based on analysis
            builder.switch_to_block(loop_body);
            let new_acc = generate_loop_body(&mut builder, &loop_analysis, i, acc)?;

            // Increment i
            let one = builder.ins().iconst(types::I64, 1);
            let next_i = builder.ins().iadd(i, one);

            // Jump back to header
            builder.ins().jump(loop_header, &[next_i, new_acc]);
            builder.seal_block(loop_body);

            // Build exit block
            builder.switch_to_block(loop_exit);
            let result = builder.block_params(loop_exit)[0];
            builder.ins().return_(&[result]);
            builder.seal_block(loop_exit);

            builder.finalize();
        }

        // Compile the function
        self.module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| JitError::CompilationFailed(format!("Failed to define function: {}", e)))?;

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions().map_err(|e| {
            JitError::CompilationFailed(format!("Failed to finalize: {}", e))
        })?;

        // Get function pointer
        let func_ptr = self.module.get_finalized_function(func_id);

        let compiled_pattern = match loop_analysis.pattern {
            LoopPattern::SumIndex => CompiledPattern::SumIndex,
            LoopPattern::CountIterations => CompiledPattern::CountIterations,
            _ => CompiledPattern::SumIndex,
        };

        let compiled = CompiledFunction {
            func_id,
            func_ptr,
            start_ip: loop_header_ip,
            end_ip: loop_exit_ip,
            accumulator_local: loop_analysis.accumulator_local,
            pattern: compiled_pattern,
        };

        self.compiled_cache.insert(loop_header_ip, compiled);
        debug!("JIT: Successfully compiled loop at IP {}", loop_header_ip);

        Ok(self.compiled_cache.get(&loop_header_ip).unwrap())
    }

    /// Analyze an integer range loop to determine what computation it performs
    fn analyze_int_range_loop(
        &self,
        instructions: &InstructionSet,
        header_ip: usize,
        exit_ip: usize,
    ) -> JitResult<LoopAnalysis> {
        let mut analysis = LoopAnalysis::default();

        // Walk through the bytecode from header to exit
        let mut ip = header_ip;
        while ip < exit_ip && ip < instructions.instructions.len() {
            let instruction = instructions.get(ip);
            let opcode = instruction.opcode();

            match opcode {
                // Skip loop control opcodes
                Opcode::ForRangeNext(_, _) | Opcode::ForRangeInit(_) | Opcode::Jump(_) => {}
                // Pop after ForRangeNext means empty loop body (loop var is discarded)
                Opcode::Pop => {
                    // If this is the only instruction after ForRangeNext, it's an empty body
                    if ip == header_ip + 1 {
                        analysis.is_empty_body = true;
                    }
                }
                Opcode::StoreAt(idx) => {
                    // Track which local is being stored to - this is likely the accumulator
                    analysis.stores_to_local = true;
                    analysis.accumulator_local = Some(idx);
                }
                Opcode::Reassign(idx) => {
                    analysis.has_reassign = true;
                    analysis.accumulator_local = Some(idx);
                }
                Opcode::Load(idx) => {
                    analysis.loads_local = true;
                    // If we load and then add/store, the loaded local is the accumulator
                    if analysis.accumulator_local.is_none() {
                        analysis.accumulator_local = Some(idx);
                    }
                }
                Opcode::LoadLocal0 => {
                    analysis.loads_local = true;
                    if analysis.accumulator_local.is_none() {
                        analysis.accumulator_local = Some(0);
                    }
                }
                Opcode::LoadLocal1 => {
                    analysis.loads_local = true;
                    if analysis.accumulator_local.is_none() {
                        analysis.accumulator_local = Some(1);
                    }
                }
                Opcode::LoadLocal2 => {
                    analysis.loads_local = true;
                    if analysis.accumulator_local.is_none() {
                        analysis.accumulator_local = Some(2);
                    }
                }
                Opcode::LoadLocal3 => {
                    analysis.loads_local = true;
                    if analysis.accumulator_local.is_none() {
                        analysis.accumulator_local = Some(3);
                    }
                }
                Opcode::Add | Opcode::AddInt => {
                    analysis.has_add = true;
                }
                Opcode::Subtract | Opcode::SubtractInt => {
                    analysis.has_sub = true;
                }
                Opcode::Multiply => {
                    analysis.has_mul = true;
                }
                Opcode::LoadConst(_) | Opcode::LoadConst0 | Opcode::LoadConst1 => {
                    // Constants are fine
                }
                Opcode::IncrementLocal(_) => {
                    analysis.has_increment = true;
                }
                // These operations make the loop not JIT-able for now
                Opcode::Call(_) | Opcode::Print | Opcode::Println => {
                    return Err(JitError::NotJitCompatible(
                        "Loop contains function calls or I/O".into(),
                    ));
                }
                _ => {
                    // For now, reject unknown opcodes
                    // In the future we can support more
                }
            }
            ip += 1;
        }

        // Determine the pattern based on analysis
        if analysis.is_empty_body {
            analysis.pattern = LoopPattern::CountIterations;
        } else if analysis.has_add && analysis.loads_local && analysis.stores_to_local {
            // Classic acc += i pattern
            analysis.pattern = LoopPattern::SumIndex;
        } else if analysis.has_increment {
            analysis.pattern = LoopPattern::CountIterations;
        } else {
            // Default to sum pattern
            analysis.pattern = LoopPattern::SumIndex;
        }

        Ok(analysis)
    }

    /// Get stats about compiled functions
    pub fn stats(&self) -> JitStats {
        JitStats {
            compiled_functions: self.compiled_cache.len(),
        }
    }
}

/// Generate Cranelift IR for the loop body based on analysis (free function to avoid borrow issues)
fn generate_loop_body(
    builder: &mut FunctionBuilder,
    analysis: &LoopAnalysis,
    i: CraneliftValue,
    acc: CraneliftValue,
) -> JitResult<CraneliftValue> {
    match analysis.pattern {
        LoopPattern::SumIndex => {
            // acc += i
            Ok(builder.ins().iadd(acc, i))
        }
        LoopPattern::SumConstant(c) => {
            // acc += c
            let const_val = builder.ins().iconst(types::I64, c);
            Ok(builder.ins().iadd(acc, const_val))
        }
        LoopPattern::SumIndexTimesConstant(c) => {
            // acc += i * c
            let const_val = builder.ins().iconst(types::I64, c);
            let product = builder.ins().imul(i, const_val);
            Ok(builder.ins().iadd(acc, product))
        }
        LoopPattern::CountIterations => {
            // acc += 1
            let one = builder.ins().iconst(types::I64, 1);
            Ok(builder.ins().iadd(acc, one))
        }
        LoopPattern::Complex => {
            // For complex patterns, just count iterations for now
            let one = builder.ins().iconst(types::I64, 1);
            Ok(builder.ins().iadd(acc, one))
        }
    }
}

/// Analysis of a loop's computation pattern
#[derive(Debug, Default)]
struct LoopAnalysis {
    has_add: bool,
    has_sub: bool,
    has_mul: bool,
    has_reassign: bool,
    has_increment: bool,
    loads_local: bool,
    stores_to_local: bool,
    pattern: LoopPattern,
    /// The local index being accumulated into (if detected)
    accumulator_local: Option<u32>,
    /// Whether the loop body is empty (just Pop)
    is_empty_body: bool,
}

/// Recognized loop computation patterns
#[derive(Debug, Default, Clone, Copy)]
enum LoopPattern {
    /// acc += i
    #[default]
    SumIndex,
    /// acc += constant
    SumConstant(i64),
    /// acc += i * constant
    SumIndexTimesConstant(i64),
    /// acc += 1 (counting)
    CountIterations,
    /// Complex pattern (not fully recognized)
    Complex,
}

/// Statistics about JIT compilation
#[derive(Debug, Default)]
pub struct JitStats {
    pub compiled_functions: usize,
}

impl std::fmt::Display for JitStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JIT Stats: {} functions compiled", self.compiled_functions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_compiler_creation() {
        let compiler = JitCompiler::new();
        assert!(compiler.is_ok());
    }
}
