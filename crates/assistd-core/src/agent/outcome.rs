//! How an agent's run ended and what it wrote, kept by the loop as the run streams.

use std::mem;

use assistd_ipc::Access;
use assistd_llm::{LlmError, LlmEvent};

/// How an agent's run ended.
#[derive(Debug)]
pub enum AgentEnd {
    /// The model answered with its tools still on offer.
    Answered,
    /// The run ended after its tools were withdrawn, on what the model had by then.
    ToolsWithdrawn,
    /// The run was cancelled, or its events went unread, before the model answered.
    Interrupted,
    /// The backend failed.
    Failed(LlmError),
}

impl AgentEnd {
    /// The error of a run that failed.
    pub fn into_result(self) -> Result<(), LlmError> {
        match self {
            Self::Failed(error) => Err(error),
            Self::Answered | Self::ToolsWithdrawn | Self::Interrupted => Ok(()),
        }
    }
}

/// What an agent wrote: its last message, and before it the text of each earlier step
/// that streamed any, oldest first.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Written {
    pub notes: Vec<String>,
    /// Empty when no step streamed text.
    pub message: String,
}

/// How a run ended, what the agent wrote, how many steps asked for tools, and what its
/// tool calls touched.
#[derive(Debug)]
pub struct AgentOutcome {
    pub end: AgentEnd,
    /// A run that did not answer keeps the longest attempt at the step it stopped in, or
    /// else the text of the last step that streamed any.
    pub written: Written,
    pub tool_steps: u32,
    /// Paths are as the run's tools saw them.
    pub accesses: Vec<Access>,
}

/// What a run has streamed so far.
#[derive(Default)]
pub(super) struct RunRecord {
    /// The text of each earlier step that streamed any, oldest first.
    earlier: Vec<String>,
    current: String,
    /// The longest attempt at the current step that was cut off at the output-token limit.
    longest_cut_off: String,
    /// The current step asked for tools, so its text outlasts the start of the next step.
    called_tools: bool,
    tool_steps: u32,
    accesses: Vec<Access>,
}

impl RunRecord {
    pub(super) fn observe(&mut self, event: &LlmEvent) {
        match event {
            LlmEvent::StepStarted => {
                let text = mem::take(&mut self.current);
                if mem::take(&mut self.called_tools) && !text.is_empty() {
                    self.earlier.push(text);
                }
            }
            LlmEvent::StepDiscarded => {
                let attempt = mem::take(&mut self.current);
                if attempt.len() > self.longest_cut_off.len() {
                    self.longest_cut_off = attempt;
                }
            }
            LlmEvent::Delta { text } => self.current.push_str(text),
            LlmEvent::ToolCallsRequested { .. } => {
                self.called_tools = true;
                self.longest_cut_off.clear();
                self.tool_steps += 1;
            }
            LlmEvent::ToolResult { accesses, .. } => self.accesses.extend_from_slice(accesses),
            _ => {}
        }
    }

    /// The outcome of the run that streamed this and ended with `end`.
    pub(super) fn ended(mut self, end: AgentEnd) -> AgentOutcome {
        let (tool_steps, accesses) = (self.tool_steps, mem::take(&mut self.accesses));
        let written = match end {
            AgentEnd::Answered | AgentEnd::ToolsWithdrawn => Written {
                notes: self.earlier,
                message: self.current,
            },
            AgentEnd::Interrupted | AgentEnd::Failed(_) => self.stopped_early(),
        };
        AgentOutcome {
            end,
            written,
            tool_steps,
            accesses,
        }
    }

    fn stopped_early(mut self) -> Written {
        let attempt = if self.current.len() >= self.longest_cut_off.len() {
            self.current
        } else {
            self.longest_cut_off
        };
        let message = if attempt.is_empty() {
            self.earlier.pop().unwrap_or_default()
        } else {
            attempt
        };
        Written {
            notes: self.earlier,
            message,
        }
    }
}
