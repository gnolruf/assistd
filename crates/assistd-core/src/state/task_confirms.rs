//! Confirmations paused task runs wait on: the registry `TaskConfirm` answers, and the
//! parking that pauses a run, gives up its lane, and resumes it first in the queue.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex as StdMutex;
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use assistd_ipc::{Event, TaskGrant, TaskId, TaskPrompt, TaskRunId};
use assistd_llm::{ConversationKey, LlmBackend};
use assistd_memory::MemoryError;
use assistd_tools::{Approval, ConfirmationRequest, PromptParking};

use super::lanes::LaneLease;
use super::{AppState, send_error};
use crate::presence::RequestGuard;

/// Why a `TaskConfirm` was refused.
#[derive(Debug, Error)]
pub(in crate::state) enum TaskConfirmError {
    #[error("task run {0} is not waiting on confirmation {1}")]
    NotWaiting(TaskRunId, String),

    #[error("no grant can cover this confirmation, so none can be added to the task")]
    NotGrantable,

    #[error(transparent)]
    Memory(#[from] MemoryError),
}

/// The confirmation each paused task run waits on.
#[derive(Debug, Default)]
pub(in crate::state) struct TaskConfirms {
    pending: StdMutex<HashMap<TaskRunId, PendingConfirm>>,
}

impl TaskConfirms {
    /// Record `pending` as what `run` waits on until the returned guard drops.
    fn register(&self, run: TaskRunId, pending: PendingConfirm) -> RegisteredConfirm<'_> {
        let confirm_id = pending.confirm_id.clone();
        self.pending.lock().insert(run, pending);
        RegisteredConfirm {
            confirms: self,
            run,
            confirm_id,
        }
    }

    /// The task, prompt and grants of confirmation `confirm_id`, which `run` must wait on.
    fn waiting(
        &self,
        run: TaskRunId,
        confirm_id: &str,
    ) -> Result<(TaskId, TaskPrompt, Vec<TaskGrant>), TaskConfirmError> {
        let pending = self.pending.lock();
        let waiting = pending
            .get(&run)
            .filter(|pending| pending.confirm_id == confirm_id)
            .ok_or_else(|| TaskConfirmError::NotWaiting(run, confirm_id.to_string()))?;
        Ok((waiting.task, waiting.prompt.clone(), waiting.grants.clone()))
    }

    /// Answer confirmation `confirm_id`, which `run` must wait on, with `approval`.
    fn answer(
        &self,
        run: TaskRunId,
        confirm_id: &str,
        approval: Approval,
    ) -> Result<(), TaskConfirmError> {
        let mut pending = self.pending.lock();
        if pending
            .get(&run)
            .is_none_or(|pending| pending.confirm_id != confirm_id)
        {
            return Err(TaskConfirmError::NotWaiting(run, confirm_id.to_string()));
        }
        if let Some(answered) = pending.remove(&run) {
            let _ = answered.answer.send(approval);
        }
        Ok(())
    }
}

/// A confirmation a paused run waits on, and where its answer goes.
#[derive(Debug)]
struct PendingConfirm {
    confirm_id: String,
    task: TaskId,
    /// The prompt the run was started with, which a grant added for it must still match.
    prompt: TaskPrompt,
    grants: Vec<TaskGrant>,
    answer: oneshot::Sender<Approval>,
}

/// A confirmation registered in [`TaskConfirms`], forgotten when dropped.
struct RegisteredConfirm<'confirms> {
    confirms: &'confirms TaskConfirms,
    run: TaskRunId,
    confirm_id: String,
}

impl Drop for RegisteredConfirm<'_> {
    fn drop(&mut self) {
        let mut pending = self.confirms.pending.lock();
        if pending
            .get(&self.run)
            .is_some_and(|pending| pending.confirm_id == self.confirm_id)
        {
            pending.remove(&self.run);
        }
    }
}

/// What a task run holds while its turn may decode: a model lane, and the daemon kept
/// awake. Both are given up while the run waits on the user.
#[derive(Debug)]
pub(in crate::state) struct RunHold {
    held: StdMutex<Option<Held>>,
}

#[derive(Debug)]
struct Held {
    lease: LaneLease,
    /// `None` when waking the model failed, which the run's next request reports.
    _awake: Option<RequestGuard>,
}

impl RunHold {
    pub(in crate::state) fn new(lease: LaneLease, awake: RequestGuard) -> Self {
        Self {
            held: StdMutex::new(Some(Held {
                lease,
                _awake: Some(awake),
            })),
        }
    }

    /// Whether the run holds the main lane now.
    pub(in crate::state) fn on_main_lane(&self) -> bool {
        self.held
            .lock()
            .as_ref()
            .is_some_and(|held| held.lease.on_main_lane())
    }

    fn release(&self) {
        self.held.lock().take();
    }

    fn take(&self, lease: LaneLease, awake: Option<RequestGuard>) {
        *self.held.lock() = Some(Held {
            lease,
            _awake: awake,
        });
    }
}

/// Parks a task run's ungranted prompts: the run shows as needing input and gives up its
/// hold until `TaskConfirm` answers, then waits first in the queue for a lane.
#[derive(Debug)]
pub(in crate::state) struct TaskRunParking {
    pub(in crate::state) state: Arc<AppState>,
    pub(in crate::state) task: TaskId,
    pub(in crate::state) prompt: TaskPrompt,
    pub(in crate::state) run: TaskRunId,
    /// Tags the run's events.
    pub(in crate::state) request_id: String,
    pub(in crate::state) hold: Arc<RunHold>,
    pub(in crate::state) backend: Arc<dyn LlmBackend>,
    pub(in crate::state) events: mpsc::Sender<Event>,
    pub(in crate::state) cancel: CancellationToken,
}

#[async_trait]
impl PromptParking for TaskRunParking {
    async fn park(&self, req: ConfirmationRequest) -> Approval {
        let (answer, answered) = oneshot::channel();
        let confirm_id = uuid::Uuid::new_v4().to_string();
        let grants = req.grantable.grants();
        let _registered = self.state.runtime.task_confirms.register(
            self.run,
            PendingConfirm {
                confirm_id: confirm_id.clone(),
                task: self.task,
                prompt: self.prompt.clone(),
                grants: grants.clone(),
                answer,
            },
        );
        self.pause().await;
        self.announce(req, confirm_id, grants).await;
        let approval = tokio::select! {
            biased;
            () = self.cancel.cancelled() => return Approval::Deny,
            answer = answered => answer.unwrap_or(Approval::Deny),
        };
        if self.resume().await {
            approval
        } else {
            Approval::Deny
        }
    }
}

impl TaskRunParking {
    async fn announce(&self, req: ConfirmationRequest, confirm_id: String, grants: Vec<TaskGrant>) {
        info!(
            target: "assistd::tasks",
            tool = %req.tool,
            pattern = %req.matched_pattern,
            "task run paused for input"
        );
        let event = Event::ConfirmRequest {
            id: self.request_id.clone(),
            confirm_id,
            tool: req.tool,
            script: req.script,
            matched_pattern: req.matched_pattern,
            always_allow: Vec::new(),
            grants,
            source: req.source,
        };
        let _ = self.events.send(event).await;
    }

    async fn pause(&self) {
        self.state.runtime.task_queue.mark_needs_input(self.run);
        self.hold.release();
        if let Err(e) = self
            .state
            .memory
            .tasks
            .set_run_needs_input(self.run, true)
            .await
        {
            warn!(target: "assistd::tasks", error = %e, "recording that the run needs input failed");
        }
    }

    /// Wait first in the queue for a lane, move the conversation to its slot, and wake the
    /// model; `false` when the run was cancelled first.
    async fn resume(&self) -> bool {
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
        self.hold.take(lease, awake);
        if let Err(e) = self
            .state
            .memory
            .tasks
            .set_run_needs_input(self.run, false)
            .await
        {
            warn!(target: "assistd::tasks", error = %e, "recording that the run resumed failed");
        }
        info!(target: "assistd::tasks", "task run resumed");
        true
    }
}

impl AppState {
    pub(super) async fn handle_task_confirm(
        self: Arc<Self>,
        id: String,
        run: TaskRunId,
        confirm_id: String,
        approval: Approval,
        tx: mpsc::Sender<Event>,
    ) {
        match self.answer_task_confirm(run, &confirm_id, approval).await {
            Ok(()) => {
                let _ = tx.send(Event::Done { id }).await;
            }
            Err(e) => send_error(&tx, id, e.to_string()).await,
        }
    }

    /// Answer `run`'s confirmation with `approval`, first saving its grants on the task when
    /// it is [`Approval::Always`] and the task's prompt is still the run's.
    async fn answer_task_confirm(
        &self,
        run: TaskRunId,
        confirm_id: &str,
        approval: Approval,
    ) -> Result<(), TaskConfirmError> {
        let confirms = &self.runtime.task_confirms;
        if approval == Approval::Always {
            let (task, prompt, grants) = confirms.waiting(run, confirm_id)?;
            if grants.is_empty() {
                return Err(TaskConfirmError::NotGrantable);
            }
            if !self.memory.tasks.add_grants(task, prompt, grants).await? {
                info!(
                    target: "assistd::tasks",
                    %run,
                    "the task's prompt changed since the run started; the grant holds for this run only"
                );
            }
        }
        confirms.answer(run, confirm_id, approval)
    }
}
