//! Sub-agent runs: the agent loop over a throwaway conversation on the shared
//! llama-server, returning only the model's final answer.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use thiserror::Error;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use assistd_config::defaults::nz32;
use assistd_llm::{LlmBackend, LlmError, LlmEvent, LlmHealthProbe};
use assistd_tools::ToolRegistry;

use crate::agent::{Agent, StepCeiling};

/// Tool-calling steps a sub-agent may take before its run fails.
const SUBAGENT_MAX_TOOL_STEPS: u32 = 30;

/// History budget of a sub-agent conversation, capped at the main one's.
const SUBAGENT_MAX_HISTORY_TOKENS: NonZeroU32 = nz32(16_384);

const EVENT_BUFFER: usize = 32;

/// Why a sub-agent run ended without an answer.
#[derive(Debug, Error)]
pub enum SubAgentError {
    /// The run was cancelled before the model answered.
    #[error("sub-agent cancelled before it answered")]
    Cancelled,
    /// The run failed, including [`LlmError::StepLimit`] and
    /// [`LlmError::ContextLimit`] when it outgrew its budgets.
    #[error("sub-agent failed: {0}")]
    Llm(#[from] LlmError),
}

/// Runs sub-agents in conversations of their own beside `parent`'s, on the
/// same server and model.
#[derive(Debug)]
pub struct SubAgentRunner {
    parent: Arc<dyn LlmBackend>,
    health: Option<Arc<dyn LlmHealthProbe>>,
    tool_deadline: Duration,
}

impl SubAgentRunner {
    /// A runner beside `parent`; `health` and `tool_deadline` act as they do for [`Agent`].
    pub fn new(
        parent: Arc<dyn LlmBackend>,
        health: Option<Arc<dyn LlmHealthProbe>>,
        tool_deadline: Duration,
    ) -> Self {
        Self {
            parent,
            health,
            tool_deadline,
        }
    }

    /// Run `task` in an empty conversation with `system_prompt` and `tools` until the model
    /// answers, returning that answer's text. Nothing from the run enters the parent's
    /// conversation, and the conversation is dropped when the run ends or `cancel` fires.
    pub async fn run(
        &self,
        system_prompt: String,
        task: String,
        tools: Arc<ToolRegistry>,
        cancel: CancellationToken,
    ) -> Result<String, SubAgentError> {
        let backend = self
            .parent
            .isolated_conversation(system_prompt, SUBAGENT_MAX_HISTORY_TOKENS)?;
        let agent = Agent::new(backend, tools, self.health.clone(), self.tool_deadline)
            .with_step_ceiling(StepCeiling::Fail(SUBAGENT_MAX_TOOL_STEPS));
        let (tx, rx) = mpsc::channel(EVENT_BUFFER);
        let (outcome, answer) = tokio::join!(
            agent.run_turn(task, Vec::new(), tx, cancel),
            final_answer(rx)
        );
        outcome?;
        answer.ok_or(SubAgentError::Cancelled)
    }
}

/// The text of the step that ended the turn, or `None` when it never
/// finished. The backend reports its context size as each step starts, so
/// that report discards text streamed by the steps before it.
async fn final_answer(mut rx: mpsc::Receiver<LlmEvent>) -> Option<String> {
    let mut text = String::new();
    let mut done = false;
    while let Some(event) = rx.recv().await {
        match event {
            LlmEvent::ContextSize { .. } => text.clear(),
            LlmEvent::Delta { text: delta } => text.push_str(&delta),
            LlmEvent::Done => done = true,
            _ => {}
        }
    }
    done.then_some(text)
}

#[cfg(test)]
mod tests;
