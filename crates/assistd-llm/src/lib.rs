#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::print_stdout,
        clippy::print_stderr
    )
)]

//! LLM backend trait and stub implementations.
//!
//! Concrete backends (`llama.cpp`, `ollama`, etc.) live in their own
//! modules/submodules and plug into the daemon via
//! [`LlmBackend`]. The daemon holds one as `Arc<dyn LlmBackend>` inside
//! `AppState` and streams response events back to connected clients.

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

/// Health/restart probe used by the chat client to classify HTTP
/// failures as crash-induced (worth replaying) vs. transport-level
/// (propagate as error).
///
/// Implemented in `assistd-core` over `Arc<PresenceManager>`. The
/// trait lives in this crate so [`LlamaChatClient`] can depend on it
/// without taking a circular dep on `assistd-core`.
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
///
/// The trait was previously returning `anyhow::Result<...>` everywhere,
/// which made it impossible for callers (the agent loop in particular)
/// to distinguish a transient HTTP hiccup from a backend that's
/// permanently unavailable. The variants below correspond to the
/// failure categories the daemon actually wants to react to:
///
/// - [`Self::Chat`] wraps a real chat-client failure (HTTP, SSE parse,
///   timeout, JSON error). Source chain preserved through `#[from]`.
/// - [`Self::ToolCallParse`] is a structural failure inside the
///   backend when finalising a streamed tool call: the model emitted
///   something we can't reassemble. Surfaced to the agent loop so it
///   can report a synthetic error and self-correct on the next step.
/// - [`Self::Unavailable`] is the `FailedBackend` shape: the daemon
///   started but llama-server never came up. Distinguished from
///   transient `Chat` errors so future code can choose to bail out
///   rather than retry.
#[derive(Debug, Error)]
pub enum LlmError {
    #[error(transparent)]
    Chat(#[from] ChatClientError),

    #[error("tool-call parse error: {0}")]
    ToolCallParse(String),

    #[error("LLM backend unavailable: {0}")]
    Unavailable(String),

    /// The llama-server child crashed mid-request and the chat client
    /// detected the supervisor is restarting it. The agent loop sees
    /// this and handles replay (emit a Status event, wait for ready,
    /// retry once); it never reaches the IPC layer as an error.
    #[error("llama-server is restarting: {0}")]
    ServerRestarting(String),
}

/// Convenience result alias. Library code should prefer `LlmError`
/// over `anyhow::Error` so callers can pattern-match.
pub type LlmResult<T> = std::result::Result<T, LlmError>;

/// Events streamed from a backend to the caller during generation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LlmEvent {
    /// A streamed chunk of model output.
    Delta { text: String },
    /// A streamed chunk of the model's chain-of-thought / reasoning
    /// content. Surfaced separately from `Delta` so the agent loop and
    /// downstream translators can render it as an expandable
    /// "Thinking…" block rather than mainline reply text, and so it
    /// doesn't get persisted into conversation history or read aloud
    /// by TTS. Backends emit this when the model has a separated
    /// reasoning channel (`reasoning_content`) or when the SSE handler
    /// extracts content between `<think>...</think>` tags.
    ReasoningDelta { text: String },
    /// The model asked to invoke a tool. Emitted by the agent loop just
    /// before the tool executes so clients can surface the call in-flight.
    ToolCall {
        id: String,
        name: String,
        arguments: Value,
    },
    /// The tool finished. Emitted by the agent loop after the tool's
    /// result has been passed back to the model. `result` is the raw
    /// JSON the tool returned (LLM-facing shape, matches `RunTool`'s
    /// output).
    ToolResult {
        id: String,
        name: String,
        result: Value,
    },
    /// Non-terminal recovery / status update emitted by the agent loop
    /// or chat backend. Mirrors the wire `Event::Status` shape so the
    /// state-layer translator can map 1:1 without re-deriving fields.
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
    /// Server-assigned call ID. Used on the replayed wire payload to
    /// match results back to calls.
    pub id: String,
    pub name: String,
    /// Arguments parsed as a JSON value. Tools see this directly via
    /// their `invoke(args)` implementation.
    pub arguments: Value,
}

/// Result of executing one [`ToolCall`]. The agent loop builds these
/// from tool-invoke output and hands them to
/// [`LlmBackend::push_tool_results`] so the model sees them on its next
/// `step`.
#[derive(Debug, Clone)]
pub struct ToolResultPayload {
    /// The matching `id` from the [`ToolCall`] that produced this result.
    pub call_id: String,
    pub name: String,
    /// The LLM-facing body: typically `RunTool`'s `output` field, which
    /// already carries the `[exit:N | Xms]` footer and any navigational
    /// error hints.
    pub content: String,
    /// Images the tool produced (e.g. from `see`). Carried forward into
    /// the model's next turn as a multimodal user message.
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

/// Persistence-shaped role for [`HistoryEntry`]. Mirrors
/// `assistd_memory::PersistedRole` so the daemon can translate row →
/// entry without pulling `assistd-memory` into this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HistoryRole {
    System,
    User,
    Assistant,
    Tool,
}

/// One persisted message reconstructed for replay into a backend's
/// in-memory conversation (used by branch /switch and daemon-startup
/// resume). `tool_calls_json` is verbatim JSON straight from the DB;
/// the concrete backend parses it back into its native shape so the
/// trait stays free of `ToolCallRecord` references.
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
    /// Generate a response to `prompt`, streaming tokens through `tx`.
    ///
    /// Implementations must send a terminal [`LlmEvent::Done`] (or return
    /// an error) when generation completes. The channel is a bounded
    /// `mpsc`; if `send` fails it means the consumer has disconnected and
    /// the implementation should stop generating and return `Ok(())`.
    ///
    /// Single-turn API used by code paths that don't want the agent
    /// loop's tool-dispatch machinery.
    async fn generate(&self, prompt: String, tx: mpsc::Sender<LlmEvent>) -> LlmResult<()>;

    /// Append a user message to the conversation. Does not invoke the
    /// model. Used by the agent loop at the start of each turn.
    ///
    /// `attachments` carries any images the user wants the model to see
    /// on its next turn. Pass an empty `Vec` for plain-text turns.
    /// Backends without vision support are expected to ignore the
    /// attachments rather than error.
    async fn push_user(&self, text: String, attachments: Vec<Attachment>) -> LlmResult<()>;

    /// Append the results of the most recent `tool_calls` request. The
    /// backend should commit these as messages the model will see on its
    /// next `step` (our implementation renders them as synthetic user
    /// messages with a `[tool:<name>]\n` prefix, plus attachments).
    async fn push_tool_results(&self, results: Vec<ToolResultPayload>) -> LlmResult<()>;

    /// Run one model invocation with the current conversation + the given
    /// tool schemas. Streams any text deltas through `tx`. Returns
    /// [`StepOutcome::Final`] when the model emitted text, or
    /// [`StepOutcome::ToolCalls`] when the model wants to invoke tools.
    ///
    /// Implementations that do not support tool use may return
    /// `StepOutcome::Final` and ignore `tools`.
    async fn step(&self, tools: Vec<Value>, tx: mpsc::Sender<LlmEvent>) -> LlmResult<StepOutcome>;

    /// Stash a one-shot system message that the next [`Self::step`] (or
    /// [`Self::generate`]) renders alongside the static system prompt.
    /// The implementation must consume the slot once the request
    /// commits, so a follow-up turn re-runs retrieval. Used by the
    /// daemon's auto-injection path to inject "Relevant past context: …"
    /// from semantic retrieval.
    ///
    /// Default implementation is a no-op so existing test backends and
    /// future stubs don't need to track per-turn context.
    async fn set_transient_context(&self, _text: String) -> LlmResult<()> {
        Ok(())
    }

    /// Replace the backend's in-memory conversation with `entries`,
    /// preserving the static system prompt. Used by `/switch` and by
    /// daemon-startup resume to align the LLM's view of history with
    /// the persisted DB state of a different branch.
    ///
    /// Default no-op so test/echo backends don't need to track history.
    async fn replace_history(&self, _entries: Vec<HistoryEntry>) -> LlmResult<()> {
        Ok(())
    }

    /// Drop the most recent "real" user message and everything after
    /// it from the in-memory conversation. A "real" user message is
    /// `Role::User` whose content does NOT start with the
    /// tool-result prefix. Returns the count of messages removed.
    ///
    /// Default no-op so test/echo backends are unaffected.
    async fn truncate_to_last_real_user(&self) -> LlmResult<usize> {
        Ok(0)
    }

    /// Used by the daemon's title-generation hook so the
    /// summarisation prompt cannot leak into the user's chat history.
    /// Returns the model's final text concatenated.
    async fn complete_oneshot(&self, _prompt: String) -> LlmResult<String> {
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
    /// Creates a new `EchoBackend` with an empty conversation state.
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

/// Backend that always fails with a fixed reason.
///
/// Used when llama-server fails to start so the TUI can still launch
/// and display the error rather than crashing before it appears.
pub struct FailedBackend {
    reason: String,
}

impl FailedBackend {
    /// Creates a `FailedBackend` that always returns [`LlmError::Unavailable`] with `reason`.
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

/// Returns the crate version string from `Cargo.toml`.
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
