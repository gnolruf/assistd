//! The task queue's dispatcher, which starts queued runs as model lanes free
//! up, and the `TaskQueue` handler.

use std::future::pending;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use assistd_ipc::{Event, SuspendReason, TaskQueueOp, TaskQueueSnapshot, TaskRunId};
use assistd_memory::{TaskRunEnd, WaitingRun};

use super::lanes::{LaneLease, ModelLanes, changed_or_pending};
use super::task_queue::{Cancelled, QueueOpError};
use super::task_runs::DAEMON_STOPPED;
use super::{AppState, send_error};

/// The `id` broadcast `TaskQueue` events carry.
const TASK_QUEUE_EVENT_ID: &str = "task_queue";

impl AppState {
    /// Reload the runs left waiting when the daemon last stopped, failing those it
    /// stopped mid-run, then start queued runs as lanes free up and queue runs as time
    /// triggers fire, until the task runner stops.
    pub async fn start_task_queue(self: &Arc<Self>) {
        self.reload_task_queue().await;
        let dispatcher = Arc::clone(self);
        self.runtime
            .task_runner
            .spawn_service(move |stopping| dispatcher.dispatch_queued_runs(stopping));
        let scheduler = Arc::clone(self);
        self.runtime
            .task_runner
            .spawn_service(move |stopping| scheduler.schedule_task_fires(stopping));
    }

    /// The task queue and the model lanes as they stand.
    pub(super) fn task_queue_snapshot(&self) -> TaskQueueSnapshot {
        self.runtime
            .task_queue
            .snapshot(self.runtime.lanes.holders())
    }

    pub(super) async fn handle_task_queue(
        self: Arc<Self>,
        id: String,
        op: TaskQueueOp,
        tx: mpsc::Sender<Event>,
    ) {
        match self.apply_task_queue_op(op).await {
            Ok(()) => {
                let _ = tx.send(Event::Done { id }).await;
            }
            Err(e) => send_error(&tx, id, e.to_string()).await,
        }
    }

    /// Drop waiting run `run`, recording it as cancelled, or stop it if it is running.
    pub(super) async fn cancel_task_run(&self, run: TaskRunId) -> Result<(), QueueOpError> {
        let runner = &self.runtime.task_runner;
        match self.runtime.task_queue.cancel(run, || runner.cancel(run)) {
            Cancelled::Waiting => {
                self.memory
                    .tasks
                    .finish_run(run, TaskRunEnd::Cancelled)
                    .await?;
                Ok(())
            }
            Cancelled::Running => Ok(()),
            Cancelled::NotFound => Err(QueueOpError::NotQueued(run)),
        }
    }

    async fn apply_task_queue_op(&self, op: TaskQueueOp) -> Result<(), QueueOpError> {
        let queue = &self.runtime.task_queue;
        match op {
            TaskQueueOp::Move { run, to } => {
                let _order = queue.lock_order().await;
                let order = queue.move_to(run, to)?;
                self.memory.tasks.reorder_queue(order).await?;
            }
            TaskQueueOp::Cancel { run } => self.cancel_task_run(run).await?,
            TaskQueueOp::Pause => self.set_task_queue_paused(true).await?,
            TaskQueueOp::Resume => self.set_task_queue_paused(false).await?,
        }
        Ok(())
    }

    /// Record that the queue is paused, or not, then hold or release its runs.
    async fn set_task_queue_paused(&self, paused: bool) -> Result<(), QueueOpError> {
        self.memory.tasks.set_queue_paused(paused).await?;
        self.runtime.task_queue.set_paused(paused);
        Ok(())
    }

    async fn reload_task_queue(&self) {
        let recovered = match self.memory.tasks.recover_queue(DAEMON_STOPPED.into()).await {
            Ok(recovered) => recovered,
            Err(e) => {
                warn!(target: "assistd::tasks", error = %e, "reloading the task queue failed");
                return;
            }
        };
        self.runtime.task_queue.set_paused(recovered.paused);
        for WaitingRun { run, task } in recovered.waiting {
            match self.memory.tasks.get(task).await {
                Ok(record) => self
                    .runtime
                    .task_queue
                    .push(run, record, unrequested_run_id(run)),
                Err(e) => warn!(
                    target: "assistd::tasks",
                    %run,
                    error = %e,
                    "dropping a queued run whose task could not be read"
                ),
            }
        }
    }

    /// Start the next queued run whenever a lane can take it, and broadcast the queue
    /// after every change, until `stopping` fires.
    async fn dispatch_queued_runs(self: Arc<Self>, stopping: CancellationToken) {
        let mut queue_changes = self.runtime.task_queue.subscribe();
        let mut lane_changes = self.runtime.lanes.watch();
        let mut gpu_hold = self.subsystems.presence.gpu_hold();
        let mut published = None;
        loop {
            let held = *gpu_hold.borrow_and_update();
            self.suspend_for_gpu(held).await;
            queue_changes.mark_unchanged();
            self.publish_task_queue(&mut published);
            let next = self.runtime.task_queue.next_to_start();
            tokio::select! {
                biased;
                () = stopping.cancelled() => return,
                _ = queue_changes.changed() => {}
                () = lane_changes.changed() => {}
                () = changed_or_pending(&mut gpu_hold) => {}
                (run, lease) = admit(&self.runtime.lanes, next) => self.start_admitted(run, lease),
            }
        }
    }

    /// Start waiting run `run` on `lease`'s lane, unless it stopped waiting meanwhile.
    fn start_admitted(self: &Arc<Self>, run: TaskRunId, lease: LaneLease) {
        let started = self
            .runtime
            .task_queue
            .start(run, |queued| self.spawn_task_run(queued, lease));
        if !started {
            debug!(target: "assistd::tasks", %run, "admitted run left the queue first");
        }
    }

    /// Suspend every waiting run while the GPU monitor holds the daemon asleep, and
    /// queue them again once it lets go, recording each change.
    async fn suspend_for_gpu(&self, held: bool) {
        let reason = held.then_some(SuspendReason::GpuBusy);
        for run in self.runtime.task_queue.suspend_waiting(reason) {
            if let Err(e) = self.memory.tasks.set_run_waiting(run, reason).await {
                warn!(
                    target: "assistd::tasks",
                    %run,
                    error = %e,
                    "recording the run's suspension failed"
                );
            }
        }
    }

    /// Broadcast the queue if it differs from `published`, the last broadcast.
    fn publish_task_queue(&self, published: &mut Option<TaskQueueSnapshot>) {
        let snapshot = self.task_queue_snapshot();
        if published.as_ref() == Some(&snapshot) {
            return;
        }
        self.runtime.publish(&Event::TaskQueue {
            id: TASK_QUEUE_EVENT_ID.to_string(),
            snapshot: snapshot.clone(),
        });
        *published = Some(snapshot);
    }
}

/// The request id tagging the events of run `run`, which no client request queued.
pub(super) fn unrequested_run_id(run: TaskRunId) -> String {
    format!("task-run-{run}")
}

/// A lane for `next`; never returns without a run to admit.
async fn admit(lanes: &Arc<ModelLanes>, next: Option<TaskRunId>) -> (TaskRunId, LaneLease) {
    let Some(run) = next else {
        return pending().await;
    };
    (run, lanes.admit(run).await)
}
