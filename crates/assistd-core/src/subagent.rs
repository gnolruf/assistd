//! Sub-agent runs: the agent loop over a throwaway conversation on the shared
//! llama-server, returning only the model's final answer as a [`RelayReport`].

use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Map, Value};
use thiserror::Error;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use assistd_config::defaults::nz32;
use assistd_llm::{LlmBackend, LlmError, LlmEvent, LlmHealthProbe};
use assistd_tools::ToolRegistry;

use crate::agent::{Agent, StepCeiling};

mod report;

pub use report::RelayReport;

/// Tool-calling steps a sub-agent may take before its run fails.
const SUBAGENT_MAX_TOOL_STEPS: u32 = 30;

/// History budget of a sub-agent conversation, capped at the main one's.
const SUBAGENT_MAX_HISTORY_TOKENS: NonZeroU32 = nz32(16_384);

const EVENT_BUFFER: usize = 32;

/// Asks for the answer again in the requested shape once the run has answered.
const SCHEMA_INSTRUCTION: &str = "Give your final answer again as a single JSON value \
     matching the required schema, and nothing else.";

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
    /// The schema-constrained answer was not JSON, as when it was cut off.
    #[error("sub-agent's structured answer is not valid JSON: {0}")]
    MalformedAnswer(#[source] serde_json::Error),
}

/// The form a sub-agent's final answer must take.
#[derive(Debug, Clone, PartialEq)]
pub enum AnswerShape {
    /// Free text.
    Text,
    /// A JSON value matching this JSON schema, enforced by grammar.
    Schema(Map<String, Value>),
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
    /// answers in `shape`, returning only that answer as a report. Nothing else from the run
    /// leaves it, and the conversation is dropped when the run ends or `cancel` fires.
    pub async fn run(
        &self,
        system_prompt: String,
        task: String,
        tools: Arc<ToolRegistry>,
        shape: AnswerShape,
        cancel: CancellationToken,
    ) -> Result<RelayReport, SubAgentError> {
        let backend = self
            .parent
            .isolated_conversation(system_prompt, SUBAGENT_MAX_HISTORY_TOKENS)?;
        let agent = Agent::new(
            Arc::clone(&backend),
            tools,
            self.health.clone(),
            self.tool_deadline,
        )
        .with_step_ceiling(StepCeiling::Fail(SUBAGENT_MAX_TOOL_STEPS));
        let (tx, rx) = mpsc::channel(EVENT_BUFFER);
        let (outcome, answer) = tokio::join!(
            agent.run_turn(task, Vec::new(), tx, cancel.clone()),
            final_answer(rx)
        );
        outcome?;
        let answer = answer.ok_or(SubAgentError::Cancelled)?;
        let answer = match shape {
            AnswerShape::Text => answer,
            AnswerShape::Schema(schema) => {
                answer_in_schema(backend.as_ref(), schema, &cancel).await?
            }
        };
        Ok(RelayReport::new(&answer))
    }
}

/// Ask `backend` to restate its answer as JSON held to `schema`, returning
/// the parsed value re-serialised compactly.
async fn answer_in_schema(
    backend: &dyn LlmBackend,
    schema: Map<String, Value>,
    cancel: &CancellationToken,
) -> Result<String, SubAgentError> {
    let reply = tokio::select! {
        biased;
        () = cancel.cancelled() => return Err(SubAgentError::Cancelled),
        reply = backend.complete_with_schema(SCHEMA_INSTRUCTION.into(), schema) => reply?,
    };
    let value: Value = serde_json::from_str(&reply).map_err(SubAgentError::MalformedAnswer)?;
    Ok(value.to_string())
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
