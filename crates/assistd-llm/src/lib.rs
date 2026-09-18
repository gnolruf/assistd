//! LLM backend trait, the llama-server chat client that implements it,
//! and the child-process supervisor that keeps llama-server alive.

pub mod chat;
pub mod llama_server;

pub use chat::conversation::ToolCallRecord;
pub use chat::{ChatClientError, LlamaChatClient};
pub use llama_server::{
    LlamaServerControl, LlamaServerError, LlamaService, ReadyState, VisionState,
    detect_vision_support, probe_capabilities, probe_capabilities_routed,
};

use assistd_tools::Attachment;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::{Mutex, mpsc};

/// Reason an [`LlmHealthProbe::wait_for_ready`] call ended without
/// observing `ReadyState::Ready`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthWaitError {
    /// The wait budget elapsed before the supervisor reported Ready.
    Timeout,
    /// The supervisor entered `Degraded` (gave up after consecutive
    /// failures); waiting longer will not help.
    Degraded,
    /// No managed service is currently attached. The daemon is
    /// `Sleeping`, or the probe was constructed against a presence
    /// manager that hasn't woken yet.
    NoService,
}

impl std::fmt::Display for HealthWaitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthWaitError::Timeout => f.write_str("LLM did not become ready before timeout"),
            HealthWaitError::Degraded => {
                f.write_str("LLM supervisor entered Degraded; restart abandoned")
            }
            HealthWaitError::NoService => {
                f.write_str("no llama-server is currently managed (presence asleep?)")
            }
        }
    }
}

impl std::error::Error for HealthWaitError {}

/// Health/restart probe the chat client consults to classify an HTTP
/// failure as crash-induced (worth replaying) or transport-level
/// (propagated as an error).
#[async_trait]
pub trait LlmHealthProbe: Send + Sync {
    /// Current PID of the managed llama-server child, or `None` if
    /// none is alive (sleeping, or in mid-restart with the child not
    /// yet spawned).
    fn pid(&self) -> Option<u32>;

    /// Snapshot of the supervisor's readiness state. Returns `None`
    /// when no service is attached (presence asleep / not yet woken).
    fn state(&self) -> Option<ReadyState>;

    /// Block until the supervisor reports `ReadyState::Ready` or
    /// `timeout` elapses. Maps `Degraded` to `HealthWaitError::Degraded`
    /// immediately so the caller can give up rather than wait the full
    /// budget on a hopeless restart.
    async fn wait_for_ready(&self, timeout: Duration) -> Result<(), HealthWaitError>;
}

/// Errors surfaced by the [`LlmBackend`] trait.
#[derive(Debug, Error)]
pub enum LlmError {
    /// HTTP, SSE, timeout, or JSON failure inside the chat client.
    #[error(transparent)]
    Chat(#[from] ChatClientError),

    /// The model emitted a streamed tool call that could not be
    /// reassembled into a name plus JSON arguments.
    #[error("tool-call parse error: {0}")]
    ToolCallParse(String),

    /// The backend never came up and every call fails with this reason.
    #[error("LLM backend unavailable: {0}")]
    Unavailable(String),

    /// The llama-server child crashed mid-request and its supervisor is
    /// restarting it. Callers may wait for readiness and replay once.
    #[error("llama-server is restarting: {0}")]
    ServerRestarting(String),
}

pub type LlmResult<T> = std::result::Result<T, LlmError>;

/// Whether a request lets a reasoning model produce its `<think>`
/// block. Reasoning is discarded by one-shot callers, so a model that
/// spends the whole token budget thinking returns nothing at all;
/// [`Thinking::Disabled`] asks the chat template to skip it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Thinking {
    Enabled,
    Disabled,
}

/// Events streamed from a backend to the caller during generation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LlmEvent {
    /// A streamed chunk of model output.
    Delta { text: String },
    /// A streamed chunk of the model's reasoning, kept apart from
    /// `Delta` so it is neither persisted as reply text nor spoken.
    ReasoningDelta { text: String },
    /// The model asked to invoke a tool; emitted before the tool runs.
    ToolCall {
        id: String,
        name: String,
        arguments: Value,
    },
    /// A tool finished. `result` is the raw JSON the tool returned.
    ToolResult {
        id: String,
        name: String,
        result: Value,
    },
    /// Non-terminal recovery or status update.
    Status {
        severity: String,
        component: String,
        event: String,
        message: String,
    },
    /// The model has finished generating.
    Done,
}

/// One tool call the model requested during a `step`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Server-assigned call ID, echoed back on the matching result.
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// Result of executing one [`ToolCall`], fed back to the model through
/// [`LlmBackend::push_tool_results`].
#[derive(Debug, Clone)]
pub struct ToolResultPayload {
    /// The `id` of the [`ToolCall`] that produced this result.
    pub call_id: String,
    pub name: String,
    /// The LLM-facing body of the result.
    pub content: String,
    /// Images the tool produced, shown to the model on its next turn.
    pub attachments: Vec<Attachment>,
}

/// Outcome of a single [`LlmBackend::step`] call.
#[derive(Debug)]
pub enum StepOutcome {
    /// The model emitted plain text; the turn is complete. Any deltas
    /// were already streamed via the `tx` channel.
    Final,
    /// The model requested one or more tool calls. The caller must
    /// dispatch them and feed the results back via
    /// [`LlmBackend::push_tool_results`] before the next `step`.
    ToolCalls(Vec<ToolCall>),
}

/// Role of a persisted message being replayed into a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HistoryRole {
    System,
    User,
    Assistant,
    Tool,
}

/// One persisted message reconstructed for replay into a backend's
/// in-memory conversation. `tool_calls_json` is the stored JSON
/// verbatim; the backend parses it into its own shape.
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    pub role: HistoryRole,
    pub content: String,
    pub tool_calls_json: Option<Value>,
    pub tool_call_id: Option<String>,
    pub tool_name: Option<String>,
}

#[async_trait]
pub trait LlmBackend: Send + Sync + 'static {
    /// Generate a single-turn response to `prompt`, streaming tokens
    /// through `tx` and ending with [`LlmEvent::Done`]. A failed `send`
    /// means the consumer is gone; stop generating and return `Ok(())`.
    async fn generate(&self, prompt: String, tx: mpsc::Sender<LlmEvent>) -> LlmResult<()>;

    /// Append a user message without invoking the model. Backends
    /// without vision support ignore `attachments` rather than error.
    async fn push_user(&self, text: String, attachments: Vec<Attachment>) -> LlmResult<()>;

    /// Append the results of the most recent tool calls so the model
    /// sees them on its next [`Self::step`].
    async fn push_tool_results(&self, results: Vec<ToolResultPayload>) -> LlmResult<()>;

    /// Run one model invocation over the current conversation with the
    /// given tool schemas, streaming text deltas through `tx`. Backends
    /// without tool support ignore `tools` and return
    /// [`StepOutcome::Final`].
    async fn step(&self, tools: Vec<Value>, tx: mpsc::Sender<LlmEvent>) -> LlmResult<StepOutcome>;

    /// Stash a one-shot system message rendered by the next
    /// [`Self::step`] or [`Self::generate`] alongside the static system
    /// prompt, then discarded once that request commits.
    async fn set_transient_context(&self, _text: String) -> LlmResult<()> {
        Ok(())
    }

    /// Replace the in-memory conversation with `entries`, preserving the
    /// static system prompt.
    async fn replace_history(&self, _entries: Vec<HistoryEntry>) -> LlmResult<()> {
        Ok(())
    }

    /// Drop the most recent user message and everything after it, where
    /// tool results are not counted as user messages. Returns how many
    /// messages were removed.
    async fn truncate_to_last_real_user(&self) -> LlmResult<usize> {
        Ok(0)
    }

    /// Answer `prompt` outside the conversation, returning the model's
    /// final text with reasoning discarded. Answers are budgeted from
    /// the summary token allowance, so callers that want an answer
    /// rather than a train of thought pass [`Thinking::Disabled`].
    async fn complete_oneshot(&self, _prompt: String, _thinking: Thinking) -> LlmResult<String> {
        Err(LlmError::Unavailable(
            "complete_oneshot not supported by this backend".into(),
        ))
    }
}

/// Trivial backend that echoes the user's most recent push as a single
/// delta on the next `step`. Tool calls are never emitted.
pub struct EchoBackend {
    last_user: Mutex<String>,
}

impl Default for EchoBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl EchoBackend {
    pub fn new() -> Self {
        Self {
            last_user: Mutex::new(String::new()),
        }
    }
}

#[async_trait]
impl LlmBackend for EchoBackend {
    async fn generate(&self, prompt: String, tx: mpsc::Sender<LlmEvent>) -> LlmResult<()> {
        let _ = tx.send(LlmEvent::Delta { text: prompt }).await;
        let _ = tx.send(LlmEvent::Done).await;
        Ok(())
    }

    async fn push_user(&self, text: String, _attachments: Vec<Attachment>) -> LlmResult<()> {
        *self.last_user.lock().await = text;
        Ok(())
    }

    async fn push_tool_results(&self, _results: Vec<ToolResultPayload>) -> LlmResult<()> {
        Ok(())
    }

    async fn step(&self, _tools: Vec<Value>, tx: mpsc::Sender<LlmEvent>) -> LlmResult<StepOutcome> {
        let text = std::mem::take(&mut *self.last_user.lock().await);
        if !text.is_empty() {
            let _ = tx.send(LlmEvent::Delta { text }).await;
        }
        Ok(StepOutcome::Final)
    }
}

/// Backend whose every call fails with [`LlmError::Unavailable`] and a
/// fixed reason.
pub struct FailedBackend {
    reason: String,
}

impl FailedBackend {
    pub fn new(reason: String) -> Self {
        Self { reason }
    }
}

#[async_trait]
impl LlmBackend for FailedBackend {
    async fn generate(&self, _prompt: String, _tx: mpsc::Sender<LlmEvent>) -> LlmResult<()> {
        Err(LlmError::Unavailable(self.reason.clone()))
    }

    async fn push_user(&self, _text: String, _attachments: Vec<Attachment>) -> LlmResult<()> {
        Err(LlmError::Unavailable(self.reason.clone()))
    }

    async fn push_tool_results(&self, _results: Vec<ToolResultPayload>) -> LlmResult<()> {
        Err(LlmError::Unavailable(self.reason.clone()))
    }

    async fn step(
        &self,
        _tools: Vec<Value>,
        _tx: mpsc::Sender<LlmEvent>,
    ) -> LlmResult<StepOutcome> {
        Err(LlmError::Unavailable(self.reason.clone()))
    }
}

/// Crate version.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn echo_backend_streams_delta_then_done() {
        let backend = EchoBackend::new();
        let (tx, mut rx) = mpsc::channel(8);
        backend.generate("hello".into(), tx).await.unwrap();
        let first = rx.recv().await.unwrap();
        let second = rx.recv().await.unwrap();
        assert_eq!(
            first,
            LlmEvent::Delta {
                text: "hello".into()
            }
        );
        assert_eq!(second, LlmEvent::Done);
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn echo_backend_step_echoes_last_pushed_user() {
        let backend = EchoBackend::new();
        backend
            .push_user("what is 2+2?".into(), Vec::new())
            .await
            .unwrap();
        let (tx, mut rx) = mpsc::channel(8);
        let outcome = backend.step(Vec::new(), tx).await.unwrap();
        assert!(matches!(outcome, StepOutcome::Final));
        match rx.recv().await.unwrap() {
            LlmEvent::Delta { text } => assert_eq!(text, "what is 2+2?"),
            other => panic!("expected Delta, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn failed_backend_returns_error() {
        let backend = FailedBackend::new("server exploded".into());
        let (tx, _rx) = mpsc::channel(8);
        let err = backend.generate("hello".into(), tx).await.unwrap_err();
        assert!(err.to_string().contains("server exploded"));
    }

    #[tokio::test]
    async fn failed_backend_step_returns_error() {
        let backend = FailedBackend::new("down".into());
        let (tx, _rx) = mpsc::channel(8);
        let err = backend.step(Vec::new(), tx).await.unwrap_err();
        assert!(err.to_string().contains("down"));
    }

    #[test]
    fn version_is_not_empty() {
        assert!(!version().is_empty());
    }

    #[test]
    fn llm_event_tool_call_roundtrips() {
        let evt = LlmEvent::ToolCall {
            id: "c-1".into(),
            name: "run".into(),
            arguments: serde_json::json!({"command": "ls"}),
        };
        let json = serde_json::to_string(&evt).unwrap();
        let parsed: LlmEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, evt);
    }
}
