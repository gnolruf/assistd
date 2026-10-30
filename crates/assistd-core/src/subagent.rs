//! Sub-agent runs: the agent loop over a throwaway conversation on the shared
//! llama-server, returning only the model's final answer as a [`RelayReport`].

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Map, Value};
use thiserror::Error;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use assistd_ipc::{RelayEvent, StatusKind};
use assistd_llm::{LlmBackend, LlmError, LlmEvent, LlmHealthProbe};
use assistd_tools::{ToolOutput, ToolRegistry};

use crate::agent::{Agent, TOOL_CALL_EVENTS};

mod report;

pub(crate) use report::REPORT_MAX_CHARS;
pub use report::RelayReport;

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
    /// The run failed.
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

/// Where a relay reports its progress: the turn whose tool call started it,
/// or nowhere outside a turn.
#[derive(Debug, Clone, Default)]
pub struct RelayProgress {
    turn: Option<mpsc::Sender<LlmEvent>>,
}

/// What a sub-agent run streamed: its last step's text, `None` when it never
/// finished, and how many tool-calling steps it took.
struct FinalAnswer {
    text: Option<String>,
    tool_steps: u32,
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
    /// answers in `shape`, returning only that answer as a report. Only the report reaches the
    /// model; `progress` is shown its tool calls, and the conversation is dropped when the
    /// run ends or `cancel` fires.
    pub async fn run(
        &self,
        system_prompt: String,
        task: String,
        tools: Arc<ToolRegistry>,
        shape: AnswerShape,
        cancel: CancellationToken,
        progress: &RelayProgress,
    ) -> Result<RelayReport, SubAgentError> {
        let backend = self.parent.isolated_conversation(system_prompt)?;
        let agent = Agent::new(
            Arc::clone(&backend),
            tools,
            self.health.clone(),
            self.tool_deadline,
        );
        let (tx, rx) = mpsc::channel(EVENT_BUFFER);
        let (outcome, finished) = tokio::join!(
            agent.run_turn(task, Vec::new(), tx, cancel.clone()),
            final_answer(rx, progress)
        );
        outcome?;
        let answer = finished.text.ok_or(SubAgentError::Cancelled)?;
        let answer = match shape {
            AnswerShape::Text => answer,
            AnswerShape::Schema(schema) => {
                answer_in_schema(backend.as_ref(), schema, &cancel).await?
            }
        };
        let report = RelayReport::new(&answer);
        progress
            .report(RelayEvent::Answered {
                answer_chars: u32::try_from(report.answer_chars()).unwrap_or(u32::MAX),
                tool_steps: finished.tool_steps,
                truncated: report.truncated(),
            })
            .await;
        Ok(report)
    }
}

impl RelayProgress {
    /// Reports to the turn whose tool call is being dispatched, if any.
    pub(crate) fn of_current_tool_call() -> Self {
        Self {
            turn: TOOL_CALL_EVENTS.try_with(Clone::clone).ok(),
        }
    }

    /// Send `event` to the turn; a turn that has ended drops it.
    pub(crate) async fn report(&self, event: RelayEvent) {
        if let Some(turn) = &self.turn {
            let _ = turn.send(LlmEvent::Relay(event)).await;
        }
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

/// Collect the run's answer while showing `progress` its tool calls and
/// compactions. The backend reports its context size as each step starts, so
/// that report discards text streamed by the steps before it.
async fn final_answer(mut rx: mpsc::Receiver<LlmEvent>, progress: &RelayProgress) -> FinalAnswer {
    let mut text = String::new();
    let mut done = false;
    let mut tool_steps = 0;
    let mut compacting_since = None;
    while let Some(event) = rx.recv().await {
        match event {
            LlmEvent::ContextSize { .. } => text.clear(),
            LlmEvent::Delta { text: delta } => text.push_str(&delta),
            LlmEvent::ToolCallsRequested { .. } => tool_steps += 1,
            LlmEvent::Done => done = true,
            event => {
                if let Some(shown) = relay_event_of(event, &mut compacting_since) {
                    progress.report(shown).await;
                }
            }
        }
    }
    FinalAnswer {
        text: done.then_some(text),
        tool_steps,
    }
}

/// What the user is shown of `event`, timing a compaction from the
/// announcement that starts it to the one that ends it.
fn relay_event_of(event: LlmEvent, compacting_since: &mut Option<Instant>) -> Option<RelayEvent> {
    match event {
        LlmEvent::ToolCall {
            name, arguments, ..
        } => Some(RelayEvent::ToolCall {
            name,
            args: arguments,
        }),
        LlmEvent::ToolResult { name, result, .. } => Some(RelayEvent::ToolResult {
            name,
            result: shown_result(&result),
        }),
        LlmEvent::Status {
            event: StatusKind::CompactingHistory,
            ..
        } => {
            *compacting_since = Some(Instant::now());
            Some(RelayEvent::Compacting)
        }
        LlmEvent::Status {
            event: StatusKind::HistoryCompacted,
            ..
        } => compacting_since.take().map(|since| RelayEvent::Compacted {
            duration_ms: u64::try_from(since.elapsed().as_millis()).unwrap_or(u64::MAX),
        }),
        _ => None,
    }
}

/// `result` as the user is shown it, without its attachments.
fn shown_result(result: &ToolOutput) -> Value {
    let mut shown = result.to_json();
    if let Some(fields) = shown.as_object_mut() {
        fields.remove("attachments");
    }
    shown
}

#[cfg(test)]
mod tests;
