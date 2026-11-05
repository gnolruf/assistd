//! `RunLane`: the model lane a task run holds while its turn may decode, given up
//! while it waits on the user or yields to the user's conversation, and the backend
//! that makes each model step yield.

use std::future::pending;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value};
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use assistd_config::TaskInterruption;
use assistd_ipc::{Event, StatusKind, TaskRunId};
use assistd_llm::{
    ConversationKey, HistoryReplay, LlmBackend, LlmError, LlmEvent, LlmResult, OneShotReply,
    SlotAssignment, StepOutcome, ToolResultPayload,
};
use assistd_tools::Attachment;

use super::AppState;
use super::lanes::LaneLease;
use crate::presence::RequestGuard;
use crate::recovery::{Component, StatusSeverity};

/// Why a yielded run's step ended without running.
const CANCELLED_WHILE_YIELDED: &str = "the task run was cancelled while it waited for the model";

tokio::task_local! {
    /// The lane of the task run whose turn is in scope.
    pub(crate) static RUN_LANE: Arc<RunLane>;
}

/// A task run's hold on a model lane and on the daemon staying awake.
#[derive(Debug)]
pub(crate) struct RunLane {
    pub(in crate::state) state: Arc<AppState>,
    pub(in crate::state) run: TaskRunId,
    /// Tags the run's events.
    pub(in crate::state) request_id: String,
    pub(in crate::state) held: LaneHold,
    /// The run's conversation, moved to the slot of each lane it regains.
    pub(in crate::state) backend: Arc<dyn LlmBackend>,
    pub(in crate::state) events: mpsc::Sender<Event>,
    pub(in crate::state) cancel: CancellationToken,
}

/// The lane and wake a [`RunLane`] holds now, if any.
#[derive(Debug)]
pub(in crate::state) struct LaneHold(watch::Sender<Option<Held>>);

#[derive(Debug)]
struct Held {
    lease: LaneLease,
    /// `None` when waking the model failed, which the run's next request reports.
    _awake: Option<RequestGuard>,
}

impl LaneHold {
    /// Holding `lease`'s lane with the daemon kept awake by `awake`.
    pub(in crate::state) fn new(lease: LaneLease, awake: RequestGuard) -> Self {
        Self(watch::Sender::new(Some(Held {
            lease,
            _awake: Some(awake),
        })))
    }
}

impl RunLane {
    /// Whether the run holds the main lane now.
    pub(in crate::state) fn on_main_lane(&self) -> bool {
        self.held
            .0
            .borrow()
            .as_ref()
            .is_some_and(|held| held.lease.on_main_lane())
    }

    pub(in crate::state) fn release(&self) {
        self.held.0.send_replace(None);
    }

    /// Wait first in the queue for a lane, move the conversation to its slot, and wake the
    /// model; `false` when the run was cancelled first.
    pub(in crate::state) async fn regain(&self) -> bool {
        let Some(lane) = self.state.runtime.task_queue.resume_first(self.run) else {
            return false;
        };
        let lease = tokio::select! {
            biased;
            () = self.cancel.cancelled() => return false,
            lease = lane => match lease {
                Ok(lease) => lease,
                Err(_) => return false,
            },
        };
        self.backend
            .move_to_slot(lease.slot(ConversationKey::TaskRun(self.run)));
        let awake = match self
            .state
            .wake_for_task_run(&self.request_id, &self.events, &self.cancel)
            .await
        {
            Ok(Some(awake)) => Some(awake),
            Ok(None) => return false,
            Err(e) => {
                warn!(target: "assistd::tasks", error = %e, "waking the model to resume the run failed");
                None
            }
        };
        self.held.0.send_replace(Some(Held {
            lease,
            _awake: awake,
        }));
        true
    }

    /// Wait until the user's conversation wants the main lane this run holds. Cancel-safe.
    pub(crate) async fn yield_requested(&self) {
        let mut conversation = self.state.runtime.conversation.subscribe();
        let mut held = self.held.0.subscribe();
        while !self.must_yield() {
            tokio::select! {
                changed = conversation.changed() => {
                    if changed.is_err() {
                        pending::<()>().await;
                    }
                }
                _ = held.changed() => {}
            }
        }
    }

    /// Give the main lane to the user's conversation if it wants it, then wait for a lane
    /// again; `false` when the run was cancelled first.
    pub(crate) async fn give_way(&self) -> bool {
        if !self.must_yield() {
            return true;
        }
        info!(target: "assistd::tasks", run = %self.run, "task run yields to the user's conversation");
        self.release();
        let regained = self.regain().await;
        if regained {
            info!(target: "assistd::tasks", run = %self.run, "task run resumed after the conversation");
        }
        regained
    }

    fn must_yield(&self) -> bool {
        let state = &self.state;
        state.config.tasks.interruption == TaskInterruption::Interrupt
            && state.runtime.lanes.is_single()
            && state.runtime.conversation.is_open()
            && self.on_main_lane()
    }
}

/// A task run's conversation whose model steps give way to the user's conversation:
/// before each step, and mid-step by discarding what streamed so far.
#[derive(Debug)]
pub(in crate::state) struct YieldingBackend {
    pub(in crate::state) inner: Arc<dyn LlmBackend>,
    pub(in crate::state) lane: Arc<RunLane>,
}

#[async_trait]
impl LlmBackend for YieldingBackend {
    async fn generate(&self, prompt: String, tx: mpsc::Sender<LlmEvent>) -> LlmResult<()> {
        self.inner.generate(prompt, tx).await
    }

    async fn push_user(&self, text: String, attachments: Vec<Attachment>) -> LlmResult<()> {
        self.inner.push_user(text, attachments).await
    }

    async fn push_tool_results(&self, results: Vec<ToolResultPayload>) -> LlmResult<()> {
        self.inner.push_tool_results(results).await
    }

    async fn step(&self, tools: Vec<Value>, tx: mpsc::Sender<LlmEvent>) -> LlmResult<StepOutcome> {
        loop {
            if !self.lane.give_way().await {
                return Err(LlmError::Unavailable(CANCELLED_WHILE_YIELDED.into()));
            }
            tokio::select! {
                biased;
                outcome = self.inner.step(tools.clone(), tx.clone()) => return outcome,
                () = self.lane.yield_requested() => announce_discarded_step(&tx).await,
            }
        }
    }

    async fn set_transient_context(&self, text: String) -> LlmResult<()> {
        self.inner.set_transient_context(text).await
    }

    async fn set_transient_note(&self, text: String) -> LlmResult<()> {
        self.inner.set_transient_note(text).await
    }

    async fn replace_history(&self, replay: HistoryReplay) -> LlmResult<()> {
        self.inner.replace_history(replay).await
    }

    async fn truncate_to_last_real_user(&self) -> LlmResult<usize> {
        self.inner.truncate_to_last_real_user().await
    }

    async fn complete_oneshot(&self, prompt: String, reply: OneShotReply) -> LlmResult<String> {
        self.inner.complete_oneshot(prompt, reply).await
    }

    async fn recent_user_messages(&self, limit: usize) -> Vec<String> {
        self.inner.recent_user_messages(limit).await
    }

    async fn complete_with_schema(
        &self,
        instruction: String,
        schema: Map<String, Value>,
    ) -> LlmResult<String> {
        self.inner.complete_with_schema(instruction, schema).await
    }

    fn isolated_conversation(
        &self,
        system_prompt: String,
        slot: SlotAssignment,
    ) -> LlmResult<Arc<dyn LlmBackend>> {
        self.inner.isolated_conversation(system_prompt, slot)
    }

    fn move_to_slot(&self, slot: SlotAssignment) {
        self.inner.move_to_slot(slot);
    }
}

async fn announce_discarded_step(tx: &mpsc::Sender<LlmEvent>) {
    let _ = tx
        .send(LlmEvent::Status {
            severity: StatusSeverity::Info,
            component: Component::Llm,
            event: StatusKind::YieldedToUser,
            message: "paused for the user's conversation; this step reruns when the task resumes"
                .into(),
        })
        .await;
}
