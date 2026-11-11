//! A backend that answers each turn with the user's own message.

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::{Mutex, mpsc};

use assistd_tools::Attachment;

use crate::{LlmBackend, LlmEvent, LlmResult, StepOutcome, ToolResultPayload};

/// Trivial backend that echoes the user's most recent push as a single
/// delta on the next `step`. Tool calls are never emitted.
#[derive(Debug)]
pub struct EchoBackend {
    last_user: Mutex<String>,
}

impl Default for EchoBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl EchoBackend {
    /// Create a backend with no pending user message.
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
