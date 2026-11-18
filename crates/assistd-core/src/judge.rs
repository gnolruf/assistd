//! The safety judge's model: a backend's server, and what the user asked of the agent it judges.

use std::error::Error;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value};

use assistd_config::Config;
use assistd_llm::{LlmBackend, OneShotReply};
use assistd_tools::{JUDGE_TIMEOUT, JudgeModel, SafetyJudge};

/// A [`JudgeModel`] answered by a backend's server, judging against either that backend's
/// history or one fixed prompt.
#[derive(Debug)]
pub(crate) struct BackendJudgeModel {
    backend: Arc<dyn LlmBackend>,
    asked: Asked,
}

#[derive(Debug)]
enum Asked {
    InBackend,
    Prompt(String),
}

impl BackendJudgeModel {
    /// Judges against the newest messages the user wrote in `backend`'s history.
    pub(crate) fn new(backend: Arc<dyn LlmBackend>) -> Self {
        Self {
            backend,
            asked: Asked::InBackend,
        }
    }

    /// Judges against `prompt` alone as what the user asked for; nothing in `backend`'s
    /// history reaches the judge.
    pub(crate) fn for_prompt(backend: Arc<dyn LlmBackend>, prompt: String) -> Self {
        Self {
            backend,
            asked: Asked::Prompt(prompt),
        }
    }
}

#[async_trait]
impl JudgeModel for BackendJudgeModel {
    async fn user_messages(&self, limit: usize) -> Vec<String> {
        match &self.asked {
            Asked::InBackend => self.backend.recent_user_messages(limit).await,
            Asked::Prompt(prompt) => vec![prompt.clone()],
        }
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

/// A judge asking `model`, when `tools.allow_auto_mode` allows Auto mode.
pub(crate) fn safety_judge(config: &Config, model: BackendJudgeModel) -> Option<SafetyJudge> {
    config
        .tools
        .allow_auto_mode
        .then(|| SafetyJudge::new(Arc::new(model), JUDGE_TIMEOUT))
}
