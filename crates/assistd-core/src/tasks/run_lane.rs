//! `RunLane`: the model lane a task run holds while its turn may decode, given up
//! while it waits on the user or yields to the user's interaction, and the backend
//! that makes each model step yield.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use assistd_ipc::{Attachment, StatusKind, TaskId, TaskRunId};
use assistd_llm::{
    LlmBackend, LlmError, LlmEvent, LlmResult, SlotHolder, StepOutcome, ToolResultPayload,
    TurnBackend,
};
use assistd_tools::YieldableLane;

use super::TaskService;
use crate::admission::activity::Activity;
use crate::admission::lanes::LaneLease;
use crate::presence::RequestGuard;
use crate::recovery::{Component, StatusSeverity};

/// Why a yielded run's step ended without running.
const CANCELLED_WHILE_YIELDED: &str = "the task run was cancelled while it waited for the model";

/// A task run's hold on a model lane and on the daemon staying awake.
#[derive(Debug)]
pub(super) struct RunLane {
    pub(super) tasks: Arc<TaskService>,
    pub(super) task: TaskId,
    pub(super) run: TaskRunId,
    pub(super) held: LaneHold,
    /// The run's backend, moved to the slot of each lane it regains.
    pub(super) backend: Arc<dyn LlmBackend>,
    pub(super) cancel: CancellationToken,
}

/// The lane and wake a [`RunLane`] holds now, if any.
#[derive(Debug)]
pub(super) struct LaneHold(watch::Sender<Option<Held>>);

#[derive(Debug)]
struct Held {
    lease: LaneLease,
    /// `None` when waking the model failed, which the run's next request reports.
    _awake: Option<RequestGuard>,
}

impl LaneHold {
    /// Holding `lease`'s lane with the daemon kept awake by `awake`.
    pub(super) fn new(lease: LaneLease, awake: RequestGuard) -> Self {
        Self(watch::Sender::new(Some(Held {
            lease,
            _awake: Some(awake),
        })))
    }
}

impl RunLane {
    /// Whether the run holds the main lane now.
    pub(super) fn on_main_lane(&self) -> bool {
        self.held
            .0
            .borrow()
            .as_ref()
            .is_some_and(|held| held.lease.on_main_lane())
    }

    pub(super) fn release(&self) {
        self.held.0.send_replace(None);
    }

    /// Wait first in the queue for a lane, move the backend to its slot, and wake the
    /// model; `false` when the run was cancelled first.
    pub(super) async fn regain(&self) -> bool {
        let Some(lane) = self.tasks.deps.admission.task_queue.resume_first(self.run) else {
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
            .move_to_slot(lease.slot(SlotHolder::TaskRun(self.run)));
        let awake = match self.tasks.wake_for_task_run(None, &self.cancel).await {
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

    fn must_yield(&self, activity: &Activity) -> bool {
        activity.run_must_yield(self.run, self.tasks.deps.config.tasks.interruption)
    }
}

#[async_trait]
impl YieldableLane for RunLane {
    async fn yield_requested(&self) {
        let mut activity = self.tasks.deps.admission.watch_activity();
        while !self.must_yield(&activity.current()) {
            activity.changed().await;
        }
    }

    async fn give_way(&self) -> bool {
        if !self.must_yield(&self.tasks.deps.admission.activity()) {
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
}

/// A task run's backend whose model steps give way to the user's interaction:
/// before each step, and mid-step by discarding what streamed so far.
#[derive(Debug)]
pub(super) struct YieldingBackend {
    pub(super) inner: Arc<dyn TurnBackend>,
    pub(super) lane: Arc<RunLane>,
}

#[async_trait]
impl TurnBackend for YieldingBackend {
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

    async fn push_interrupted_reply(&self, reply: String) -> LlmResult<()> {
        self.inner.push_interrupted_reply(reply).await
    }

    async fn set_transient_note(&self, text: String) -> LlmResult<()> {
        self.inner.set_transient_note(text).await
    }
}

async fn announce_discarded_step(tx: &mpsc::Sender<LlmEvent>) {
    let _ = tx.send(LlmEvent::StepDiscarded).await;
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
