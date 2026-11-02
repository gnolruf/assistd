//! The safety judge's model: a backend's server, and the user's side of its conversation.

use std::error::Error;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value};

use assistd_tools::JudgeModel;

use crate::{LlmBackend, OneShotReply};

/// A [`JudgeModel`] over `backend`, which should hold the main conversation so sub-agent
/// commands are judged against what the user asked for.
#[derive(Debug)]
pub struct BackendJudgeModel {
    backend: Arc<dyn LlmBackend>,
}

impl BackendJudgeModel {
    pub fn new(backend: Arc<dyn LlmBackend>) -> Self {
        Self { backend }
    }
}

#[async_trait]
impl JudgeModel for BackendJudgeModel {
    async fn user_messages(&self, limit: usize) -> Vec<String> {
        self.backend.recent_user_messages(limit).await
    }

    async fn reply(
        &self,
        prompt: String,
        schema: Map<String, Value>,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        Ok(self
            .backend
            .complete_oneshot(prompt, OneShotReply::Json(schema))
            .await?)
    }
}
