use super::*;
use log::debug;

impl<'a> VM<'a> {
    /// Format the current call stack as a human-readable stack trace
    /// Truncates the middle if there are too many frames (like Python does)
    pub(super) fn format_stack_trace(&self) -> String {
        if self.call_stack.len() <= 1 {
            return String::new();
        }

        const MAX_FRAMES_TOP: usize = 5;
        const MAX_FRAMES_BOTTOM: usize = 5;

        let mut trace = String::from("\nStack trace (most recent call last):\n");
        let len = self.call_stack.len();

        if len <= MAX_FRAMES_TOP + MAX_FRAMES_BOTTOM {
            for (i, frame) in self.call_stack.iter().enumerate() {
                let name = if frame.function_name.is_empty() {
                    "<fn>"
                } else {
                    frame.function_name.as_str()
                };
                trace.push_str(&format!("  {}: {}\n", i, name));
            }
        } else {
            for (i, frame) in self.call_stack.iter().take(MAX_FRAMES_TOP).enumerate() {
                let name = if frame.function_name.is_empty() {
                    "<fn>"
                } else {
                    frame.function_name.as_str()
                };
                trace.push_str(&format!("  {}: {}\n", i, name));
            }

            let hidden = len - MAX_FRAMES_TOP - MAX_FRAMES_BOTTOM;
            trace.push_str(&format!("  ... {} more frames ...\n", hidden));

            for (i, frame) in self
                .call_stack
                .iter()
                .skip(len - MAX_FRAMES_BOTTOM)
                .enumerate()
            {
                let actual_index = len - MAX_FRAMES_BOTTOM + i;
                let name = if frame.function_name.is_empty() {
                    "<fn>"
                } else {
                    frame.function_name.as_str()
                };
                trace.push_str(&format!("  {}: {}\n", actual_index, name));
            }
        }

        trace
    }

    pub(super) fn stack_trace(&self) {
        for (i, frame) in self.stack.iter().enumerate() {
            debug!("| {} | {}: {}", self.function_name(), i, frame);
        }
    }

    /// Run the debugger prompt and return the command
    pub(super) fn run_debugger_prompt(
        &mut self,
        instructions: &InstructionSet,
    ) -> WalrusResult<debugger::DebuggerCommand> {
        let call_stack: Vec<debugger::DebugCallFrame> = self
            .call_stack
            .iter()
            .map(|f| debugger::DebugCallFrame {
                function_name: if f.function_name.is_empty() {
                    "<fn>".to_string()
                } else {
                    f.function_name.clone()
                },
                return_ip: f.return_ip,
                frame_pointer: f.frame_pointer,
            })
            .collect();

        let ctx = debugger::DebugContext {
            ip: self.ip,
            stack: &self.stack,
            locals: &self.locals,
            globals: &self.globals,
            call_stack: &call_stack,
            debug_info: instructions.debug_info.as_ref(),
            instructions: &instructions.instructions,
            source: self.source_ref.source(),
        };

        let cmd = if let Some(ref mut dbg) = self.debugger {
            debugger::debug_prompt(dbg, &ctx)
        } else {
            debugger::DebuggerCommand::Continue
        };

        Ok(cmd)
    }
}
