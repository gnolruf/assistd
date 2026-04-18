//! LLM backend trait and stub implementations.
//!
//! Concrete backends (`llama.cpp`, `ollama`, etc.) live in their own
//! modules/submodules and plug into the daemon via
//! [`LlmBackend`]. The daemon holds one as `Arc<dyn LlmBackend>` inside
//! `AppState` and streams response events back to connected clients.

pub mod chat;
pub mod llama_server;

pub use chat::{ChatClientError, LlamaChatClient};
pub use llama_server::{LlamaServerControl, LlamaServerError, LlamaService, ReadyState};

use anyhow::Result;
use assistd_tools::Attachment;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{Mutex, mpsc};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LlmEvent {
    /// A streamed chunk of model output.
    Delta { text: String },
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

#[async_trait]
pub trait LlmBackend: Send + Sync + 'static {
    /// Generate a response to `prompt`, streaming tokens through `tx`.
    ///
    /// Implementations must send a terminal [`LlmEvent::Done`] (or return
    /// an error) when generation completes. The channel is a bounded
    /// `mpsc`; if `send` fails it means the consumer has disconnected and
    /// the implementation should stop generating and return `Ok(())`.
    ///
    /// This is the legacy single-turn API used by code paths that don't
    /// want the agent loop's tool-dispatch machinery.
    async fn generate(&self, prompt: String, tx: mpsc::Sender<LlmEvent>) -> Result<()>;

    /// Append a user message to the conversation. Does not invoke the
    /// model. Used by the agent loop at the start of each turn.
    async fn push_user(&self, text: String) -> Result<()>;

    /// Append the results of the most recent `tool_calls` request. The
    /// backend should commit these as messages the model will see on its
    /// next `step` (our implementation renders them as synthetic user
    /// messages with a `[tool:<name>]\n` prefix, plus attachments).
    async fn push_tool_results(&self, results: Vec<ToolResultPayload>) -> Result<()>;

    /// Run one model invocation with the current conversation + the given
    /// tool schemas. Streams any text deltas through `tx`. Returns
    /// [`StepOutcome::Final`] when the model emitted text, or
    /// [`StepOutcome::ToolCalls`] when the model wants to invoke tools.
    ///
    /// Implementations that do not support tool use may return
    /// `StepOutcome::Final` and ignore `tools`.
    async fn step(&self, tools: Vec<Value>, tx: mpsc::Sender<LlmEvent>) -> Result<StepOutcome>;
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
    async fn generate(&self, prompt: String, tx: mpsc::Sender<LlmEvent>) -> Result<()> {
        let _ = tx.send(LlmEvent::Delta { text: prompt }).await;
        let _ = tx.send(LlmEvent::Done).await;
        Ok(())
    }

    async fn push_user(&self, text: String) -> Result<()> {
        *self.last_user.lock().await = text;
        Ok(())
    }

    async fn push_tool_results(&self, _results: Vec<ToolResultPayload>) -> Result<()> {
        Ok(())
    }

    async fn step(&self, _tools: Vec<Value>, tx: mpsc::Sender<LlmEvent>) -> Result<StepOutcome> {
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
    pub fn new(reason: String) -> Self {
        Self { reason }
    }
}

#[async_trait]
impl LlmBackend for FailedBackend {
    async fn generate(&self, _prompt: String, _tx: mpsc::Sender<LlmEvent>) -> Result<()> {
        anyhow::bail!("{}", self.reason)
    }

    async fn push_user(&self, _text: String) -> Result<()> {
        anyhow::bail!("{}", self.reason)
    }

    async fn push_tool_results(&self, _results: Vec<ToolResultPayload>) -> Result<()> {
        anyhow::bail!("{}", self.reason)
    }

    async fn step(&self, _tools: Vec<Value>, _tx: mpsc::Sender<LlmEvent>) -> Result<StepOutcome> {
        anyhow::bail!("{}", self.reason)
    }
}

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
        backend.push_user("what is 2+2?".into()).await.unwrap();
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
