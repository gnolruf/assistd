//! The task queue's dispatcher, which grants lanes as who is using the model changes,
//! starts the runs granted one and marks the runs held, and the `TaskQueue` handler.

use std::future::pending;
use std::sync::Arc;

use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::warn;

use assistd_ipc::{Event, EventBody, SuspendReason, TaskQueueOp, TaskQueueSnapshot, TaskRunId};
use assistd_memory::{TaskRunEnd, WaitingRun};

use super::TaskService;
use super::runs::DAEMON_STOPPED;
use crate::admission::queue::{Cancelled, QueueOpError};

/// The `id` broadcast `TaskQueue` events carry.
const TASK_QUEUE_EVENT_ID: &str = "task_queue";

impl TaskService {
    /// Reload the runs left waiting when the daemon last stopped, failing those it
    /// stopped mid-run, prune old finished runs, then start queued runs as lanes free up and queue runs as time
    /// and event triggers fire, until the task runner stops.
    pub async fn start_task_queue(self: &Arc<Self>) {
        self.reload_task_queue().await;
        self.prune_task_runs().await;
        let dispatcher = Arc::clone(self);
        self.runner
            .spawn_service(move |stopping| dispatcher.dispatch_queued_runs(stopping));
        let scheduler = Arc::clone(self);
        self.runner
            .spawn_service(move |stopping| scheduler.schedule_task_fires(stopping));
        let watcher = Arc::clone(self);
        self.runner
            .spawn_service(move |stopping| watcher.watch_task_events(stopping));
        let startup = Arc::clone(self);
        self.runner
            .spawn_service(move |stopping| startup.fire_on_daemon_started(stopping));
    }

    /// Drop waiting run `run`, recording it as cancelled, or stop it if it is running.
    pub(super) async fn cancel_task_run(&self, run: TaskRunId) -> Result<(), QueueOpError> {
        let runner = &self.runner;
        match self
            .deps
            .admission
            .task_queue
            .cancel(run, || runner.cancel(run))
        {
            Cancelled::Waiting => {
                self.deps
                    .store
                    .finish_run(run, TaskRunEnd::Cancelled)
                    .await?;
                Ok(())
            }
            Cancelled::Running => Ok(()),
            Cancelled::NotFound => Err(QueueOpError::NotQueued(run)),
        }
    }

    pub(super) async fn apply_task_queue_op(&self, op: TaskQueueOp) -> Result<(), QueueOpError> {
        let queue = &self.deps.admission.task_queue;
        match op {
            TaskQueueOp::Move { run, to } => {
                let _order = queue.lock_order().await;
                let order = queue.move_to(run, to)?;
                self.deps.store.reorder_queue(order).await?;
            }
            TaskQueueOp::Cancel { run } => self.cancel_task_run(run).await?,
            TaskQueueOp::CancelTurn { id } => queue.cancel_turn(&id)?,
            TaskQueueOp::Pause => self.set_task_queue_paused(true).await?,
            TaskQueueOp::Resume => {
                self.set_task_queue_paused(false).await?;
                self.deps.admission.interaction.end();
            }
        }
        Ok(())
    }

    /// Record that the queue is paused, or not, then hold or release its runs.
    async fn set_task_queue_paused(&self, paused: bool) -> Result<(), QueueOpError> {
        self.deps.store.set_queue_paused(paused).await?;
        self.deps.admission.task_queue.set_paused(paused);
        Ok(())
    }

    async fn reload_task_queue(&self) {
        let recovered = match self.deps.store.recover_queue(DAEMON_STOPPED.into()).await {
            Ok(recovered) => recovered,
            Err(e) => {
                warn!(target: "assistd::tasks", error = %e, "reloading the task queue failed");
                return;
            }
        };
        self.deps.admission.task_queue.set_paused(recovered.paused);
        for WaitingRun { run, task } in recovered.waiting {
            match self.deps.store.get(task).await {
                Ok(record) => {
                    self.deps
                        .admission
                        .task_queue
                        .push(run, record, unrequested_run_id(run));
                }
                Err(e) => warn!(
                    target: "assistd::tasks",
                    %run,
                    error = %e,
                    "dropping a queued run whose task could not be read"
                ),
            }
        }
    }

    /// After every change to the queue or to who is using the model: grant lanes, start
    /// each run granted its first, mark the waiting runs held, end an interaction quiet
    /// long enough and broadcast the queue, until `stopping` fires.
    async fn dispatch_queued_runs(self: Arc<Self>, stopping: CancellationToken) {
        let queue = &self.deps.admission.task_queue;
        let mut queue_changes = queue.subscribe();
        let mut activity_changes = self.deps.admission.watch_activity();
        let mut published = None;
        loop {
            queue.grant_lanes();
            let activity = activity_changes.current();
            self.suspend_waiting_runs(activity.waiting_runs_held())
                .await;
            queue_changes.mark_unchanged();
            self.publish_task_queue(&mut published);
            tokio::select! {
                biased;
                () = stopping.cancelled() => return,
                start = queue.granted_run() => {
                    queue.start_run(start, |queued, lease| self.spawn_task_run(queued, lease));
                }
                _ = queue_changes.changed() => {}
                () = activity_changes.changed() => {}
                () = quiet_ends(activity.quiet_until()) => {
                    self.deps.admission.interaction.end_if_quiet_by(Instant::now());
                }
            }
        }
    }

    /// Mark every waiting run suspended for `reason`, or queued again when `None`,
    /// recording each change.
    async fn suspend_waiting_runs(&self, reason: Option<SuspendReason>) {
        for run in self.deps.admission.task_queue.suspend_waiting(reason) {
            if let Err(e) = self.deps.store.set_run_waiting(run, reason).await {
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
        let snapshot = self.deps.admission.queue_snapshot();
        if published.as_ref() == Some(&snapshot) {
            return;
        }
        self.deps.bus.publish(&Event::new(
            TASK_QUEUE_EVENT_ID.to_string(),
            EventBody::TaskQueue {
                snapshot: snapshot.clone(),
            },
        ));
        *published = Some(snapshot);
    }
}

/// The request id tagging the events of run `run`, which no client request queued.
pub(super) fn unrequested_run_id(run: TaskRunId) -> String {
    format!("task-run-{run}")
}

/// When a quiet interaction's quiet time runs out; never for any other.
async fn quiet_ends(quiet_until: Option<Instant>) {
    match quiet_until {
        Some(until) => tokio::time::sleep_until(until).await,
        None => pending().await,
    }
}
