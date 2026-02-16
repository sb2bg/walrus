//! Full-featured debugger for the Walrus VM.
//!
//! Provides step-by-step execution, breakpoints, variable inspection,
//! call stack display, and source context viewing.

use std::collections::HashSet;
use std::io::{self, Write};

/// Step mode for the debugger
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepMode {
    /// Run until next breakpoint
    Continue,
    /// Execute one instruction (step into)
    Step,
    /// Step over function calls
    Next,
    /// Run until current function returns
    Finish,
}

/// Information about a breakpoint
#[derive(Debug, Clone)]
pub struct BreakpointInfo {
    /// Source line number (1-indexed)
    pub line: usize,
    /// Whether this breakpoint is enabled
    pub enabled: bool,
}

/// The debugger state
#[derive(Debug)]
pub struct Debugger {
    /// Set of IPs where we should break
    breakpoint_ips: HashSet<usize>,
    /// Breakpoints by line number (for display)
    breakpoints: Vec<BreakpointInfo>,
    /// Current step mode
    step_mode: StepMode,
    /// Call stack depth when "next" was issued (for step-over)
    step_over_depth: Option<usize>,
    /// Call stack depth when "finish" was issued (for finish)
    finish_depth: Option<usize>,
    /// Whether to show output on next check
    pub should_prompt: bool,
}

impl Default for Debugger {
    fn default() -> Self {
        Self::new()
    }
}

impl Debugger {
    pub fn new() -> Self {
        Self {
            breakpoint_ips: HashSet::new(),
            breakpoints: Vec::new(),
            step_mode: StepMode::Step, // Start paused
            step_over_depth: None,
            finish_depth: None,
            should_prompt: true,
        }
    }

    /// Check if we should pause execution at the current IP
    pub fn should_break(&self, ip: usize, call_depth: usize) -> bool {
        // Always break on breakpoints
        if self.breakpoint_ips.contains(&ip) {
            return true;
        }

        match self.step_mode {
            StepMode::Continue => false,
            StepMode::Step => true,
            StepMode::Next => {
                // Break if we're at or above the depth where next was issued
                match self.step_over_depth {
                    Some(depth) => call_depth <= depth,
                    None => true,
                }
            }
            StepMode::Finish => {
                // Break when we return to a shallower call depth
                match self.finish_depth {
                    Some(depth) => call_depth < depth,
                    None => false,
                }
            }
        }
    }

    /// Set a line-based breakpoint. Returns the IPs that were set.
    pub fn set_breakpoint_at_line(
        &mut self,
        line: usize,
        line_table: &crate::vm::instruction_set::LineTable,
    ) -> Vec<usize> {
        let ips = line_table.get_ips_for_line(line);
        for &ip in &ips {
            self.breakpoint_ips.insert(ip);
        }
        if !ips.is_empty() {
            self.breakpoints.push(BreakpointInfo {
                line,
                enabled: true,
            });
        }
        ips
    }

    /// Remove a breakpoint at a line. Returns true if removed.
    pub fn remove_breakpoint_at_line(
        &mut self,
        line: usize,
        line_table: &crate::vm::instruction_set::LineTable,
    ) -> bool {
        let ips = line_table.get_ips_for_line(line);
        let mut removed = false;
        for ip in ips {
            if self.breakpoint_ips.remove(&ip) {
                removed = true;
            }
        }
        if removed {
            self.breakpoints.retain(|bp| bp.line != line);
        }
        removed
    }

    /// Set step mode to continue
    pub fn continue_execution(&mut self) {
        self.step_mode = StepMode::Continue;
        self.step_over_depth = None;
        self.finish_depth = None;
        self.should_prompt = false;
    }

    /// Set step mode to step (single instruction)
    pub fn step(&mut self) {
        self.step_mode = StepMode::Step;
        self.step_over_depth = None;
        self.finish_depth = None;
        self.should_prompt = false;
    }

    /// Set step mode to next (step over)
    pub fn next(&mut self, current_depth: usize) {
        self.step_mode = StepMode::Next;
        self.step_over_depth = Some(current_depth);
        self.finish_depth = None;
        self.should_prompt = false;
    }

    /// Set step mode to finish (run until return)
    pub fn finish(&mut self, current_depth: usize) {
        self.step_mode = StepMode::Finish;
        self.step_over_depth = None;
        self.finish_depth = Some(current_depth);
        self.should_prompt = false;
    }

    /// Get the current step mode
    pub fn step_mode(&self) -> StepMode {
        self.step_mode
    }

    /// Get all breakpoints
    pub fn breakpoints(&self) -> &[BreakpointInfo] {
        &self.breakpoints
    }

    /// Trigger prompt on next check
    pub fn trigger_prompt(&mut self) {
        self.should_prompt = true;
    }
}

/// Context passed to the debugger prompt for displaying state
pub struct DebugContext<'a> {
    pub ip: usize,
    pub stack: &'a [crate::value::Value],
    pub locals: &'a [crate::value::Value],
    pub globals: &'a [crate::value::Value],
    pub call_stack: &'a [DebugCallFrame],
    pub debug_info: Option<&'a crate::vm::instruction_set::DebugInfo>,
    pub instructions: &'a [crate::vm::opcode::Instruction],
    pub source: &'a str,
}

/// Simplified call frame info for debugger display
pub struct DebugCallFrame {
    pub function_name: String,
    pub return_ip: usize,
    pub frame_pointer: usize,
}

/// Run the interactive debugger prompt.
/// Returns the command to execute, or None to quit.
pub fn debug_prompt(debugger: &mut Debugger, ctx: &DebugContext) -> DebuggerCommand {
    // Show current location
    print_location(ctx);

    loop {
        print!("(walrus-debug) ");
        io::stdout().flush().expect("Failed to flush stdout");

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            return DebuggerCommand::Quit;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        let parts: Vec<&str> = input.split_whitespace().collect();
        let cmd = parts[0];
        let args = &parts[1..];

        match cmd {
            "s" | "step" => {
                debugger.step();
                return DebuggerCommand::Step;
            }
            "n" | "next" => {
                debugger.next(ctx.call_stack.len());
                return DebuggerCommand::Next;
            }
            "c" | "continue" => {
                debugger.continue_execution();
                return DebuggerCommand::Continue;
            }
            "finish" => {
                debugger.finish(ctx.call_stack.len());
                return DebuggerCommand::Finish;
            }
            "p" | "print" => {
                if args.is_empty() {
                    print_state(ctx);
                } else {
                    print_variable(ctx, args[0]);
                }
            }
            "locals" => {
                print_locals(ctx);
            }
            "bt" | "backtrace" => {
                print_backtrace(ctx);
            }
            "l" | "list" => {
                let line = args.first().and_then(|s| s.parse().ok());
                print_source_context(ctx, line);
            }
            "b" => {
                if args.is_empty() {
                    // List breakpoints
                    print_breakpoints(debugger);
                } else if let Ok(line) = args[0].parse::<usize>() {
                    // Set breakpoint at line
                    if let Some(debug_info) = ctx.debug_info {
                        let ips = debugger.set_breakpoint_at_line(line, &debug_info.line_table);
                        if ips.is_empty() {
                            println!("No code at line {}", line);
                        } else {
                            println!("Breakpoint set at line {} (IP: {:?})", line, ips);
                        }
                    } else {
                        println!("No debug info available for line breakpoints");
                    }
                } else {
                    println!("Usage: b <line>");
                }
            }
            "delete" => {
                if args.is_empty() {
                    println!("Usage: delete <line>");
                } else if let Ok(line) = args[0].parse::<usize>() {
                    if let Some(debug_info) = ctx.debug_info {
                        if debugger.remove_breakpoint_at_line(line, &debug_info.line_table) {
                            println!("Breakpoint at line {} removed", line);
                        } else {
                            println!("No breakpoint at line {}", line);
                        }
                    } else {
                        println!("No debug info available");
                    }
                } else {
                    println!("Usage: delete <line>");
                }
            }
            "stack" => {
                print_operand_stack(ctx);
            }
            "q" | "quit" => {
                return DebuggerCommand::Quit;
            }
            "h" | "help" => {
                print_help();
            }
            _ => {
                println!(
                    "Unknown command: {}. Type 'help' for available commands.",
                    cmd
                );
            }
        }
    }
}

/// Command returned by the debug prompt
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebuggerCommand {
    Step,
    Next,
    Continue,
    Finish,
    Quit,
}

fn print_location(ctx: &DebugContext) {
    let line = ctx.debug_info.and_then(|di| di.line_table.get_line(ctx.ip));

    if ctx.ip < ctx.instructions.len() {
        let instr = &ctx.instructions[ctx.ip];
        let opcode = instr.opcode();

        if let Some(line_num) = line {
            println!("-> Line {}, IP {}: {:?}", line_num, ctx.ip, opcode);
            // Show the source line
            if let Some(source_line) = ctx.source.lines().nth(line_num.saturating_sub(1)) {
                println!("   {}", source_line.trim());
            }
        } else {
            println!("-> IP {}: {:?}", ctx.ip, opcode);
        }
    }
}

fn print_state(ctx: &DebugContext) {
    println!("IP: {}", ctx.ip);
    println!("Call depth: {}", ctx.call_stack.len());
    println!("Stack size: {}", ctx.stack.len());
    println!("Locals count: {}", ctx.locals.len());
}

fn print_variable(ctx: &DebugContext, name: &str) {
    // Try to find the variable in locals
    if let Some(debug_info) = ctx.debug_info {
        for (i, local_name) in debug_info.local_names.iter().enumerate() {
            if local_name == name {
                if i < ctx.locals.len() {
                    println!("{} = {:?}", name, ctx.locals[i]);
                    return;
                }
            }
        }
        // Try globals
        for (i, global_name) in debug_info.global_names.iter().enumerate() {
            if global_name == name {
                if i < ctx.globals.len() {
                    println!("{} = {:?}", name, ctx.globals[i]);
                    return;
                }
            }
        }
        println!("Variable '{}' not found", name);
    } else {
        // No debug info, try numeric index
        if let Ok(idx) = name.parse::<usize>() {
            if idx < ctx.locals.len() {
                println!("local[{}] = {:?}", idx, ctx.locals[idx]);
                return;
            }
        }
        println!("No debug info available. Use numeric index.");
    }
}

fn print_locals(ctx: &DebugContext) {
    if ctx.locals.is_empty() {
        println!("No locals");
        return;
    }

    println!("Locals:");
    if let Some(debug_info) = ctx.debug_info {
        for (i, value) in ctx.locals.iter().enumerate() {
            let name = debug_info
                .local_names
                .get(i)
                .map(|s| s.as_str())
                .unwrap_or("?");
            println!("  [{}] {} = {:?}", i, name, value);
        }
    } else {
        for (i, value) in ctx.locals.iter().enumerate() {
            println!("  [{}] = {:?}", i, value);
        }
    }
}

fn print_backtrace(ctx: &DebugContext) {
    if ctx.call_stack.is_empty() {
        println!("Empty call stack");
        return;
    }

    println!("Call stack (most recent first):");
    for (i, frame) in ctx.call_stack.iter().rev().enumerate() {
        let line = ctx
            .debug_info
            .and_then(|di| di.line_table.get_line(frame.return_ip));
        let line_str = line.map(|l| format!(" at line {}", l)).unwrap_or_default();
        println!(
            "  #{}: {} (return IP: {}){}",
            i, frame.function_name, frame.return_ip, line_str
        );
    }
}

fn print_source_context(ctx: &DebugContext, target_line: Option<usize>) {
    let current_line = ctx
        .debug_info
        .and_then(|di| di.line_table.get_line(ctx.ip))
        .unwrap_or(1);

    let center_line = target_line.unwrap_or(current_line);
    let lines: Vec<&str> = ctx.source.lines().collect();

    if lines.is_empty() {
        println!("No source available");
        return;
    }

    let start = center_line.saturating_sub(4);
    let end = (center_line + 3).min(lines.len());

    println!("Source context:");
    for i in start..end {
        let line_num = i + 1;
        let marker = if line_num == current_line { "->" } else { "  " };
        if i < lines.len() {
            println!("{} {:4}: {}", marker, line_num, lines[i]);
        }
    }
}

fn print_breakpoints(debugger: &Debugger) {
    let bps = debugger.breakpoints();
    if bps.is_empty() {
        println!("No breakpoints set");
        return;
    }

    println!("Breakpoints:");
    for (i, bp) in bps.iter().enumerate() {
        let status = if bp.enabled { "enabled" } else { "disabled" };
        println!("  #{}: line {} ({})", i + 1, bp.line, status);
    }
}

fn print_operand_stack(ctx: &DebugContext) {
    if ctx.stack.is_empty() {
        println!("Operand stack is empty");
        return;
    }

    println!("Operand stack (top first):");
    for (i, value) in ctx.stack.iter().rev().enumerate() {
        println!("  [{}] {:?}", i, value);
    }
}

fn print_help() {
    println!(
        r#"Walrus Debugger Commands:
  s, step       - Execute one instruction (step into)
  n, next       - Step over function calls
  c, continue   - Resume execution until next breakpoint
  finish        - Run until current function returns

  p, print      - Print current state
  p <var>       - Print variable by name
  locals        - Show all local variables with names
  bt, backtrace - Show call stack
  l, list       - Show source context around current line
  l <line>      - Show source context around specified line

  b             - List all breakpoints
  b <line>      - Set breakpoint at line
  delete <line> - Remove breakpoint at line

  stack         - Show operand stack

  q, quit       - Exit debugger
  h, help       - Show this help
"#
    );
}
