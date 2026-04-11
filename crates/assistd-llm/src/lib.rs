//! LLM backend trait and stub implementations.
//!
//! Concrete backends (`llama.cpp`, `ollama`, etc.) live in their own
//! modules/submodules and plug into the daemon via
//! [`LlmBackend`]. The daemon holds one as `Arc<dyn LlmBackend>` inside
//! `AppState` and streams response events back to connected clients.

pub mod llama_server;

pub use llama_server::{LlamaServerError, LlamaService, ModelSpec, ReadyState, ServerSpec};

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LlmEvent {
    /// A streamed chunk of model output.
    Delta { text: String },
    /// The model has finished generating.
    Done,
}

#[async_trait]
pub trait LlmBackend: Send + Sync + 'static {
    /// Generate a response to `prompt`, streaming tokens through `tx`.
    ///
    /// Implementations must send a terminal [`LlmEvent::Done`] (or return
    /// an error) when generation completes. The channel is a bounded
    /// `mpsc`; if `send` fails it means the consumer has disconnected and
    /// the implementation should stop generating and return `Ok(())`.
    async fn generate(&self, prompt: String, tx: mpsc::Sender<LlmEvent>) -> Result<()>;
}

/// Trivial backend that echoes the prompt back as a single delta.
///
/// Used by milestone-1 scaffolding and by tests. Replaced with a real
/// `llama.cpp` backend in the LLM-integration milestone.
pub struct EchoBackend;

#[async_trait]
impl LlmBackend for EchoBackend {
    async fn generate(&self, prompt: String, tx: mpsc::Sender<LlmEvent>) -> Result<()> {
        let _ = tx.send(LlmEvent::Delta { text: prompt }).await;
        let _ = tx.send(LlmEvent::Done).await;
        Ok(())
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
        let backend = EchoBackend;
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

    #[test]
    fn version_is_not_empty() {
        assert!(!version().is_empty());
    }
}
