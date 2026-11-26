//! What an agent is, apart from the history it runs over.

use std::sync::Arc;

use assistd_llm::SlotAssignment;
use assistd_tools::ToolRegistry;

/// Tool-calling steps a turn may take while it keeps making different calls.
pub(super) const MAX_TOOL_STEPS: u32 = 200;

/// Consecutive identical tool calls after which the model is treated as stuck.
pub(super) const DUPLICATE_CALL_LIMIT: usize = 3;

/// What an agent is: the prompt its history opens with, the tools it is offered, the
/// limits its loop holds it to, and the server slot its requests use.
#[derive(Debug, Clone)]
pub struct AgentDefinition {
    pub system_prompt: String,
    pub tools: Arc<ToolRegistry>,
    /// Offered in place of `tools` while the turn plans.
    pub planning_tools: Option<Arc<ToolRegistry>>,
    pub limits: AgentLimits,
    pub slot: SlotAssignment,
}

impl AgentDefinition {
    /// An agent under `system_prompt` offered `tools` on `slot`, with the default limits
    /// and the same tools while it plans.
    pub fn new(system_prompt: String, tools: Arc<ToolRegistry>, slot: SlotAssignment) -> Self {
        Self {
            system_prompt,
            tools,
            planning_tools: None,
            limits: AgentLimits::default(),
            slot,
        }
    }
}

/// When an agent's loop withdraws its tools, leaving it to answer from what it has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgentLimits {
    /// Tool-calling steps one turn may take.
    pub tool_steps: u32,
    /// Identical tool calls in a row that count as stuck.
    pub duplicate_calls: usize,
}

impl Default for AgentLimits {
    fn default() -> Self {
        Self {
            tool_steps: MAX_TOOL_STEPS,
            duplicate_calls: DUPLICATE_CALL_LIMIT,
        }
    }
}
