//! The task queue's dispatcher, which starts queued runs and user turns as model
//! lanes free up and holds runs for the user's interaction, and the `TaskQueue` handler.

use std::future::pending;
use std::sync::Arc;

use tokio::sync::OwnedMutexGuard;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use assistd_ipc::{Event, EventBody, SuspendReason, TaskQueueOp, TaskQueueSnapshot, TaskRunId};
use assistd_memory::{TaskRunEnd, WaitingRun};

use super::TaskService;
use super::runs::DAEMON_STOPPED;
use crate::admission::interaction::InteractionState;
use crate::admission::lanes::{LaneLease, ModelLanes, changed_or_pending};
use crate::admission::queue::{Cancelled, NextStart, QueueOpError, TurnStart};

/// The `id` broadcast `TaskQueue` events carry.
const TASK_QUEUE_EVENT_ID: &str = "task_queue";

/// A waiting item [`admit`] found a lane for.
enum Admitted {
    Run(TaskRunId, LaneLease),
    /// The user turn of the query with this request id, and the main lane.
    Turn(String, OwnedMutexGuard<()>),
}

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

    /// Start the next queued item whenever a lane can take it, hold waiting runs while the
    /// GPU monitor or the user's interaction does, end that interaction once it has been
    /// quiet long enough, and broadcast the queue after every change, until `stopping` fires.
    async fn dispatch_queued_runs(self: Arc<Self>, stopping: CancellationToken) {
        let mut queue_changes = self.deps.admission.task_queue.subscribe();
        let mut lane_changes = self.deps.admission.lanes.watch();
        let mut gpu_hold = self.deps.presence.gpu_hold();
        let mut interaction = self.deps.admission.interaction.subscribe();
        let mut published = None;
        loop {
            let gpu_held = *gpu_hold.borrow_and_update();
            let interaction_now = *interaction.borrow_and_update();
            self.suspend_waiting_runs(gpu_held).await;
            queue_changes.mark_unchanged();
            self.publish_task_queue(&mut published);
            let next = self.deps.admission.task_queue.next_to_start();
            tokio::select! {
                biased;
                () = stopping.cancelled() => return,
                _ = queue_changes.changed() => {}
                () = lane_changes.changed() => {}
                () = changed_or_pending(&mut gpu_hold) => {}
                () = changed_or_pending(&mut interaction) => {}
                () = quiet_ends(interaction_now) => {
                    self.deps.admission.interaction.end_if_quiet_by(Instant::now());
                }
                admitted = admit(&self.deps.admission.lanes, next) => self.start_admitted(admitted),
            }
        }
    }

    /// Start the waiting item a lane was found for, unless it stopped waiting meanwhile or
    /// the user's interaction took the main lane first.
    fn start_admitted(self: &Arc<Self>, admitted: Admitted) {
        match admitted {
            Admitted::Run(_, lease)
                if lease.on_main_lane() && self.deps.admission.interaction_holds_queue() =>
            {
                debug!(target: "assistd::tasks", "the user's conversation took the lane first");
            }
            Admitted::Run(run, lease) => {
                let started = self
                    .deps
                    .admission
                    .task_queue
                    .start(run, lease, |queued, lease| {
                        self.spawn_task_run(queued, lease);
                    });
                if !started {
                    debug!(target: "assistd::tasks", %run, "admitted run left the queue first");
                }
            }
            Admitted::Turn(id, lane) => {
                let turn = self
                    .deps
                    .admission
                    .interaction
                    .begin_turn(self.deps.admission.interaction_quiet_time());
                let start = TurnStart { lane, turn };
                if let Err(unstarted) = self.deps.admission.task_queue.start_turn(&id, start) {
                    debug!(target: "assistd::tasks", %id, "admitted user turn left the queue first");
                    unstarted.turn.abandon();
                }
            }
        }
    }

    /// Suspend every waiting run while the GPU monitor holds the daemon asleep or the
    /// user's interaction holds the only lane, and queue them again once both let go,
    /// recording each change.
    async fn suspend_waiting_runs(&self, gpu_held: bool) {
        let reason = if gpu_held {
            Some(SuspendReason::GpuBusy)
        } else {
            self.deps
                .admission
                .interaction_holds_queue()
                .then_some(SuspendReason::UserInteraction)
        };
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

/// A lane for `next`; never returns without an item to admit.
async fn admit(lanes: &Arc<ModelLanes>, next: Option<NextStart>) -> Admitted {
    match next {
        None => pending().await,
        Some(NextStart::Run(run)) => Admitted::Run(run, lanes.admit(run).await),
        Some(NextStart::Turn(id)) => Admitted::Turn(id, lanes.admit_user_turn().await),
    }
}

/// When a quiet interaction's quiet time runs out; never for any other.
async fn quiet_ends(interaction: InteractionState) {
    match interaction {
        InteractionState::Quiet { until, .. } => tokio::time::sleep_until(until).await,
        InteractionState::Idle | InteractionState::Talking { .. } => pending().await,
    }
}
