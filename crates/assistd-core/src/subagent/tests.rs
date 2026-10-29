use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Notify;

use assistd_llm::{LlmResult, StepOutcome, ToolResultPayload};
use assistd_tools::Attachment;

use super::*;

/// A parent whose isolated conversations block in `step` until dropped.
#[derive(Debug, Default)]
struct HangingParent {
    stepping: Arc<Notify>,
}

#[derive(Debug)]
struct HangingConversation {
    stepping: Arc<Notify>,
}

#[async_trait]
impl LlmBackend for HangingParent {
    async fn generate(&self, _prompt: String, _tx: mpsc::Sender<LlmEvent>) -> LlmResult<()> {
        unimplemented!("parent is never driven")
    }

    async fn push_user(&self, _text: String, _attachments: Vec<Attachment>) -> LlmResult<()> {
        unimplemented!("parent is never driven")
    }

    async fn push_tool_results(&self, _results: Vec<ToolResultPayload>) -> LlmResult<()> {
        unimplemented!("parent is never driven")
    }

    async fn step(
        &self,
        _tools: Vec<Value>,
        _tx: mpsc::Sender<LlmEvent>,
    ) -> LlmResult<StepOutcome> {
        unimplemented!("parent is never driven")
    }

    fn isolated_conversation(
        &self,
        _system_prompt: String,
        _max_history_tokens: NonZeroU32,
    ) -> LlmResult<Arc<dyn LlmBackend>> {
        Ok(Arc::new(HangingConversation {
            stepping: Arc::clone(&self.stepping),
        }))
    }
}

#[async_trait]
impl LlmBackend for HangingConversation {
    async fn generate(&self, _prompt: String, _tx: mpsc::Sender<LlmEvent>) -> LlmResult<()> {
        unimplemented!("sub-agent uses the step path only")
    }

    async fn push_user(&self, _text: String, _attachments: Vec<Attachment>) -> LlmResult<()> {
        Ok(())
    }

    async fn push_tool_results(&self, _results: Vec<ToolResultPayload>) -> LlmResult<()> {
        Ok(())
    }

    async fn step(
        &self,
        _tools: Vec<Value>,
        _tx: mpsc::Sender<LlmEvent>,
    ) -> LlmResult<StepOutcome> {
        self.stepping.notify_one();
        std::future::pending().await
    }
}

#[tokio::test]
async fn cancelling_the_parent_turn_stops_the_run() {
    let parent = Arc::new(HangingParent::default());
    let stepping = Arc::clone(&parent.stepping);
    let runner = SubAgentRunner::new(parent, None, Duration::from_secs(60));
    let parent_turn = CancellationToken::new();
    let run = runner.run(
        "sub".into(),
        "task".into(),
        Arc::new(ToolRegistry::new()),
        parent_turn.child_token(),
    );
    let cancel_mid_step = async {
        stepping.notified().await;
        parent_turn.cancel();
    };

    let (result, ()) = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::join!(run, cancel_mid_step)
    })
    .await
    .expect("cancellation stops the run");

    assert!(
        matches!(result, Err(SubAgentError::Cancelled)),
        "{result:?}"
    );
}
