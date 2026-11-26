//! A backend that plays back scripted model steps and records what it was asked.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::{Map, Value};
use tokio::sync::{Notify, mpsc};

use assistd_ipc::Attachment;

use crate::{
    LlmBackend, LlmError, LlmEvent, LlmResult, OneShotReply, ReplyLimits, SlotAssignment,
    StepOutcome, ToolCall, ToolResultPayload, TurnBackend,
};

/// A point where a scripted step waits until [`Gate::release`]; one never released stalls
/// the step for good.
#[derive(Debug, Default)]
pub struct Gate {
    reached: Notify,
    released: Notify,
}

impl Gate {
    /// A gate no step has reached.
    pub fn new() -> Arc<Self> {
        Arc::default()
    }

    /// Wait until a step is waiting here.
    pub async fn reached(&self) {
        self.reached.notified().await;
    }

    /// Let the step waiting here, or the next to arrive, go on.
    pub fn release(&self) {
        self.released.notify_one();
    }

    async fn hold(&self) {
        self.reached.notify_one();
        self.released.notified().await;
    }
}

/// What a scripted step does before it ends.
#[derive(Debug)]
enum Beat {
    Say(String),
    Pause(Duration),
    Wait(Arc<Gate>),
    Flood(String),
}

impl Beat {
    async fn play(self, tx: &mpsc::Sender<LlmEvent>) {
        match self {
            Self::Say(text) => {
                let _ = tx.send(LlmEvent::Delta { text }).await;
            }
            Self::Pause(duration) => tokio::time::sleep(duration).await,
            Self::Wait(gate) => gate.hold().await,
            Self::Flood(text) => {
                while tx
                    .send(LlmEvent::Delta { text: text.clone() })
                    .await
                    .is_ok()
                {}
            }
        }
    }
}

/// One scripted model step: what it streams and waits for, in the order added, then how it
/// ends. The default streams nothing and answers.
#[derive(Debug)]
pub struct Step {
    beats: Vec<Beat>,
    end: LlmResult<StepOutcome>,
}

impl Default for Step {
    fn default() -> Self {
        Self::ending(Ok(StepOutcome::Final))
    }
}

impl Step {
    fn ending(end: LlmResult<StepOutcome>) -> Self {
        Self {
            beats: Vec::new(),
            end,
        }
    }

    /// A step that answers with `text`, streamed as one delta unless empty.
    pub fn reply(text: impl Into<String>) -> Self {
        Self::default().saying(text)
    }

    /// A step that asks for `calls`.
    pub fn tool_calls(calls: impl IntoIterator<Item = ToolCall>) -> Self {
        Self::ending(Ok(StepOutcome::ToolCalls(calls.into_iter().collect())))
    }

    /// A step cut off at the output-token limit.
    pub fn truncated() -> Self {
        Self::ending(Ok(StepOutcome::Truncated))
    }

    /// A step that fails with `error`.
    pub fn error(error: LlmError) -> Self {
        Self::ending(Err(error))
    }

    /// Stream `text` as one delta; an empty `text` streams nothing.
    pub fn saying(mut self, text: impl Into<String>) -> Self {
        let text = text.into();
        if !text.is_empty() {
            self.beats.push(Beat::Say(text));
        }
        self
    }

    /// Sleep for `duration`.
    pub fn pausing(mut self, duration: Duration) -> Self {
        self.beats.push(Beat::Pause(duration));
        self
    }

    /// Wait at `gate` until it is released.
    pub fn waiting_at(mut self, gate: &Arc<Gate>) -> Self {
        self.beats.push(Beat::Wait(Arc::clone(gate)));
        self
    }

    /// Stream `text` over and over until nobody is reading the step.
    pub fn flooding(mut self, text: impl Into<String>) -> Self {
        self.beats.push(Beat::Flood(text.into()));
        self
    }
}

/// One step asked of a backend that journals: the backend's name and the user message it
/// was answering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JournaledStep {
    pub backend: String,
    pub prompt: String,
}

/// The steps asked of every backend journaling here, in the order they began.
#[derive(Debug, Clone, Default)]
pub struct StepJournal(Arc<Mutex<Vec<JournaledStep>>>);

impl StepJournal {
    /// The steps journaled so far.
    pub fn steps(&self) -> Vec<JournaledStep> {
        self.0.lock().clone()
    }
}

/// What a [`ScriptedBackend`] was asked, each list in order of arrival.
#[derive(Debug, Clone, Default)]
pub struct Seen {
    pub users: Vec<String>,
    pub tool_results: Vec<Vec<ToolResultPayload>>,
    /// The number of tool schemas offered to each step begun, finished or not.
    pub offered_tools: Vec<usize>,
    pub transient_notes: Vec<String>,
    pub interrupted_replies: Vec<String>,
    /// Each [`LlmBackend::complete_oneshot`] prompt and the reply it asked for.
    pub oneshots: Vec<(String, OneShotReply)>,
    /// The schema of each [`LlmBackend::complete_with_schema`].
    pub schemas: Vec<Map<String, Value>>,
    /// How many isolated backends were asked for, handed out or not.
    pub isolated_requests: usize,
}

/// A backend whose steps play a script in order, then all answer with the fallback
/// reply, empty unless set. One-shot and schema requests fail until given an answer.
#[derive(Debug, Default)]
pub struct ScriptedBackend {
    script: Mutex<VecDeque<Step>>,
    fallback_reply: String,
    oneshot_answer: Option<String>,
    schema_answer: Option<String>,
    isolated: Mutex<VecDeque<Arc<dyn LlmBackend>>>,
    journal: Option<(String, StepJournal)>,
    seen: Mutex<Seen>,
}

impl ScriptedBackend {
    /// A backend whose first steps are `steps`.
    pub fn new(steps: impl IntoIterator<Item = Step>) -> Self {
        Self {
            script: Mutex::new(steps.into_iter().collect()),
            ..Self::default()
        }
    }

    /// Answer every step past the script with `text`.
    pub fn then_replying(mut self, text: impl Into<String>) -> Self {
        self.fallback_reply = text.into();
        self
    }

    /// Answer every [`LlmBackend::complete_oneshot`] with `text`.
    pub fn answering_oneshots(mut self, text: impl Into<String>) -> Self {
        self.oneshot_answer = Some(text.into());
        self
    }

    /// Answer every [`LlmBackend::complete_with_schema`] with `text`.
    pub fn answering_schemas(mut self, text: impl Into<String>) -> Self {
        self.schema_answer = Some(text.into());
        self
    }

    /// Hand out `backends`, in order, as this backend's isolated backends; once they run
    /// out, asking for one fails.
    pub fn handing_out(self, backends: impl IntoIterator<Item = Arc<dyn LlmBackend>>) -> Self {
        self.isolated.lock().extend(backends);
        self
    }

    /// Record each step in `journal` under `name` as it begins.
    pub fn journaling(mut self, name: impl Into<String>, journal: &StepJournal) -> Self {
        self.journal = Some((name.into(), journal.clone()));
        self
    }

    /// Everything asked of this backend so far.
    pub fn seen(&self) -> Seen {
        self.seen.lock().clone()
    }

    fn begin_step(&self, offered_tools: usize) -> Step {
        let mut seen = self.seen.lock();
        seen.offered_tools.push(offered_tools);
        if let Some((name, journal)) = &self.journal {
            journal.0.lock().push(JournaledStep {
                backend: name.clone(),
                prompt: seen.users.last().cloned().unwrap_or_default(),
            });
        }
        self.script
            .lock()
            .pop_front()
            .unwrap_or_else(|| Step::reply(self.fallback_reply.as_str()))
    }
}

#[async_trait]
impl TurnBackend for ScriptedBackend {
    async fn push_user(&self, text: String, _attachments: Vec<Attachment>) -> LlmResult<()> {
        self.seen.lock().users.push(text);
        Ok(())
    }

    async fn push_tool_results(&self, results: Vec<ToolResultPayload>) -> LlmResult<()> {
        self.seen.lock().tool_results.push(results);
        Ok(())
    }

    async fn step(&self, tools: Vec<Value>, tx: mpsc::Sender<LlmEvent>) -> LlmResult<StepOutcome> {
        let step = self.begin_step(tools.len());
        for beat in step.beats {
            beat.play(&tx).await;
        }
        step.end
    }

    async fn push_interrupted_reply(&self, reply: String) -> LlmResult<()> {
        self.seen.lock().interrupted_replies.push(reply);
        Ok(())
    }

    async fn set_transient_note(&self, text: String) -> LlmResult<()> {
        self.seen.lock().transient_notes.push(text);
        Ok(())
    }
}

#[async_trait]
impl LlmBackend for ScriptedBackend {
    async fn complete_oneshot(&self, prompt: String, reply: OneShotReply) -> LlmResult<String> {
        self.seen.lock().oneshots.push((prompt, reply));
        self.oneshot_answer
            .clone()
            .ok_or_else(|| LlmError::Unavailable("no scripted one-shot answer".into()))
    }

    async fn recent_user_messages(&self, limit: usize) -> Vec<String> {
        let seen = self.seen.lock();
        seen.users[seen.users.len().saturating_sub(limit)..].to_vec()
    }

    async fn complete_with_schema(
        &self,
        _instruction: String,
        schema: Map<String, Value>,
    ) -> LlmResult<String> {
        self.seen.lock().schemas.push(schema);
        self.schema_answer
            .clone()
            .ok_or_else(|| LlmError::Unavailable("no scripted schema answer".into()))
    }

    fn isolated_backend(
        &self,
        _system_prompt: String,
        _slot: SlotAssignment,
        _limits: ReplyLimits,
    ) -> LlmResult<Arc<dyn LlmBackend>> {
        self.seen.lock().isolated_requests += 1;
        self.isolated
            .lock()
            .pop_front()
            .ok_or_else(|| LlmError::Unavailable("no scripted isolated backend left".into()))
    }
}
