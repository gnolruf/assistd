//! Confirmations paused task runs wait on: the registry `TaskConfirm` answers, the
//! parking that pauses a run, gives up its lane, and resumes it first in the queue, and the
//! "always" answer that adds a web host to the task's allowed hosts.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex as StdMutex;
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};
use tracing::{info, warn};

use assistd_ipc::{Event, HostName, PausedPrompt, TaskId, TaskRunId};
use assistd_memory::TaskRecord;
use assistd_tools::{Approval, ConfirmationRequest, PromptParking, Unattended};

use super::run_lane::RunLane;
use super::{AppState, DispatchError, send_error};

/// Why a `TaskConfirm` was refused.
#[derive(Debug, Error)]
pub(in crate::state) enum TaskConfirmError {
    #[error("task run {0} is not waiting on confirmation {1}")]
    NotWaiting(TaskRunId, String),

    #[error("confirmation {0} offers no host to allow for the task")]
    NoHost(String),
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

    /// Answer confirmation `confirm_id`, which `run` must wait on, with `approval`; an
    /// [`Approval::Always`] returns the host it allows, and is refused when none is offered.
    fn answer(
        &self,
        run: TaskRunId,
        confirm_id: &str,
        approval: Approval,
    ) -> Result<Option<AllowedHost>, TaskConfirmError> {
        let mut pending = self.pending.lock();
        let Entry::Occupied(waiting) = pending.entry(run) else {
            return Err(TaskConfirmError::NotWaiting(run, confirm_id.to_string()));
        };
        let confirm = waiting.get();
        if confirm.confirm_id != confirm_id {
            return Err(TaskConfirmError::NotWaiting(run, confirm_id.to_string()));
        }
        let allowed = match (approval, &confirm.host) {
            (Approval::Always, Some(host)) => Some(AllowedHost {
                task: confirm.task,
                host: host.clone(),
            }),
            (Approval::Always, None) => {
                return Err(TaskConfirmError::NoHost(confirm_id.to_string()));
            }
            (Approval::Once | Approval::Deny, _) => None,
        };
        let _ = waiting.remove().answer.send(approval);
        Ok(allowed)
    }
}

/// A confirmation a paused run of `task` waits on, and where its answer goes.
#[derive(Debug)]
struct PendingConfirm {
    confirm_id: String,
    task: TaskId,
    /// The web host an "always" answer allows for the task.
    host: Option<HostName>,
    answer: oneshot::Sender<Approval>,
}

/// A web host an "always" answer allows for the runs of `task`.
#[derive(Debug)]
struct AllowedHost {
    task: TaskId,
    host: HostName,
}

/// A confirmation registered in [`TaskConfirms`], forgotten when dropped.
struct RegisteredConfirm<'confirms> {
    confirms: &'confirms TaskConfirms,
    run: TaskRunId,
    confirm_id: String,
}

impl Drop for RegisteredConfirm<'_> {
    fn drop(&mut self) {
        if let Entry::Occupied(waiting) = self.confirms.pending.lock().entry(self.run)
            && waiting.get().confirm_id == self.confirm_id
        {
            waiting.remove();
        }
    }
}

/// Parks the task run prompts its mode and hosts leave open: the run shows as needing input
/// and gives up its hold until `TaskConfirm` answers, then waits first in the queue for a lane.
#[derive(Debug)]
pub(in crate::state) struct TaskRunParking {
    pub(in crate::state) lane: Arc<RunLane>,
}

#[async_trait]
impl PromptParking for TaskRunParking {
    async fn park(&self, req: ConfirmationRequest) -> Approval {
        let (answer, answered) = oneshot::channel();
        let confirm_id = uuid::Uuid::new_v4().to_string();
        let host = match &req.unattended {
            Unattended::WebHost(host) => HostName::try_from(host.clone()).ok(),
            _ => None,
        };
        let lane = &self.lane;
        let _registered = lane.state.runtime.task_confirms.register(
            lane.run,
            PendingConfirm {
                confirm_id: confirm_id.clone(),
                task: lane.task,
                host: host.clone(),
                answer,
            },
        );
        self.pause(PausedPrompt {
            confirm_id: confirm_id.clone(),
            tool: req.tool.clone(),
            script: req.script.clone(),
            matched_pattern: req.matched_pattern.clone(),
            host,
            source: req.source.clone(),
        })
        .await;
        self.announce(req, confirm_id).await;
        let approval = tokio::select! {
            biased;
            () = lane.cancel.cancelled() => return Approval::Deny,
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
    async fn announce(&self, req: ConfirmationRequest, confirm_id: String) {
        info!(
            target: "assistd::tasks",
            tool = %req.tool,
            pattern = %req.matched_pattern,
            "task run paused for input"
        );
        let event = Event::ConfirmRequest {
            id: self.lane.request_id.clone(),
            confirm_id,
            tool: req.tool,
            script: req.script,
            matched_pattern: req.matched_pattern,
            always_allow: Vec::new(),
            source: req.source,
        };
        let _ = self.lane.events.send(event).await;
    }

    async fn pause(&self, prompt: PausedPrompt) {
        let lane = &self.lane;
        lane.state
            .runtime
            .task_queue
            .mark_needs_input(lane.run, prompt);
        lane.release();
        if let Err(e) = lane
            .state
            .memory
            .tasks
            .set_run_needs_input(lane.run, true)
            .await
        {
            warn!(target: "assistd::tasks", error = %e, "recording that the run needs input failed");
        }
    }

    /// Wait first in the queue for a lane and take it again; `false` when the run was
    /// cancelled first.
    async fn resume(&self) -> bool {
        let lane = &self.lane;
        if !lane.regain().await {
            return false;
        }
        if let Err(e) = lane
            .state
            .memory
            .tasks
            .set_run_needs_input(lane.run, false)
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
    ) -> Result<(), DispatchError> {
        match self
            .runtime
            .task_confirms
            .answer(run, &confirm_id, approval)
        {
            Ok(None) => {
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Ok(Some(allowed)) => {
                let record = self.allow_task_host(allowed).await.map(|r| vec![r]);
                self.runtime.task_scheduler.reschedule();
                self.send_tasks(id, "allowing the host for the task", record, &tx)
                    .await
            }
            Err(e) => {
                send_error(&tx, id, e.to_string()).await;
                Ok(())
            }
        }
    }

    /// Add `allowed.host` to its task's allowed hosts, unless already there.
    async fn allow_task_host(&self, allowed: AllowedHost) -> Result<TaskRecord, DispatchError> {
        let _pass = self.runtime.task_scheduler.hold().await;
        let mut record = self.memory.tasks.get(allowed.task).await?;
        if record.spec.allowed_hosts.contains(&allowed.host) {
            return Ok(record);
        }
        record.spec.allowed_hosts.push(allowed.host);
        Ok(self.memory.tasks.save(Some(record.id), record.spec).await?)
    }
}

/// The approval a `TaskConfirm`'s `allow` and `always` give; `always` alone denies.
pub(super) fn approval(allow: bool, always: bool) -> Approval {
    match (allow, always) {
        (false, _) => Approval::Deny,
        (true, false) => Approval::Once,
        (true, true) => Approval::Always,
    }
}
