//! The task queue's dispatcher, which starts queued runs and user turns as model
//! lanes free up and holds runs for the user's conversation, and the `TaskQueue` handler.

use std::future::pending;
use std::sync::Arc;

use tokio::sync::{OwnedMutexGuard, mpsc};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use assistd_ipc::{
    ConversationHold, Event, SuspendReason, TaskQueueOp, TaskQueueSnapshot, TaskRunId,
};
use assistd_memory::{TaskRunEnd, WaitingRun};

use super::conversation::ConversationState;
use super::lanes::{LaneLease, ModelLanes, changed_or_pending};
use super::task_queue::{Cancelled, NextStart, QueueOpError, TurnStart};
use super::task_runs::DAEMON_STOPPED;
use super::{AppState, send_error};

/// The `id` broadcast `TaskQueue` events carry.
const TASK_QUEUE_EVENT_ID: &str = "task_queue";

/// A waiting item [`admit`] found a lane for.
enum Admitted {
    Run(TaskRunId, LaneLease),
    /// The user turn of the query with this request id, and the main lane.
    Turn(String, OwnedMutexGuard<()>),
}

impl AppState {
    /// Reload the runs left waiting when the daemon last stopped, failing those it
    /// stopped mid-run, then start queued runs as lanes free up and queue runs as time
    /// and event triggers fire, until the task runner stops.
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
        let watcher = Arc::clone(self);
        self.runtime
            .task_runner
            .spawn_service(move |stopping| watcher.watch_task_events(stopping));
        let startup = Arc::clone(self);
        self.runtime
            .task_runner
            .spawn_service(move |stopping| startup.fire_on_daemon_started(stopping));
    }

    /// The task queue and the model lanes as they stand.
    pub(super) fn task_queue_snapshot(&self) -> TaskQueueSnapshot {
        self.runtime
            .task_queue
            .snapshot(self.runtime.lanes.holders(), self.conversation_hold())
    }

    /// Whether the user's conversation keeps task runs off the model's only lane.
    pub(super) fn conversation_holds_queue(&self) -> bool {
        self.runtime.lanes.is_single() && self.runtime.conversation.is_open()
    }

    fn conversation_hold(&self) -> Option<ConversationHold> {
        if !self.runtime.lanes.is_single() {
            return None;
        }
        match self.runtime.conversation.state() {
            ConversationState::Idle => None,
            ConversationState::Talking { .. } => Some(ConversationHold { resumes_at: None }),
            ConversationState::Quiet { resumes_at, .. } => Some(ConversationHold {
                resumes_at: Some(resumes_at),
            }),
        }
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
            TaskQueueOp::CancelTurn { id } => queue.cancel_turn(&id)?,
            TaskQueueOp::Pause => self.set_task_queue_paused(true).await?,
            TaskQueueOp::Resume => {
                self.set_task_queue_paused(false).await?;
                self.runtime.conversation.end();
            }
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

    /// Start the next queued item whenever a lane can take it, hold waiting runs while the
    /// GPU monitor or the user's conversation does, end the conversation once it has been
    /// quiet long enough, and broadcast the queue after every change, until `stopping` fires.
    async fn dispatch_queued_runs(self: Arc<Self>, stopping: CancellationToken) {
        let mut queue_changes = self.runtime.task_queue.subscribe();
        let mut lane_changes = self.runtime.lanes.watch();
        let mut gpu_hold = self.subsystems.presence.gpu_hold();
        let mut conversation = self.runtime.conversation.subscribe();
        let mut published = None;
        loop {
            let gpu_held = *gpu_hold.borrow_and_update();
            let talking = *conversation.borrow_and_update();
            self.suspend_waiting_runs(gpu_held).await;
            queue_changes.mark_unchanged();
            self.publish_task_queue(&mut published);
            let next = self.runtime.task_queue.next_to_start();
            tokio::select! {
                biased;
                () = stopping.cancelled() => return,
                _ = queue_changes.changed() => {}
                () = lane_changes.changed() => {}
                () = changed_or_pending(&mut gpu_hold) => {}
                () = changed_or_pending(&mut conversation) => {}
                () = quiet_ends(talking) => {
                    self.runtime.conversation.end_if_quiet_by(Instant::now());
                }
                admitted = admit(&self.runtime.lanes, next) => self.start_admitted(admitted),
            }
        }
    }

    /// Start the waiting item a lane was found for, unless it stopped waiting meanwhile or
    /// the user's conversation took the main lane first.
    fn start_admitted(self: &Arc<Self>, admitted: Admitted) {
        match admitted {
            Admitted::Run(_, lease) if lease.on_main_lane() && self.conversation_holds_queue() => {
                debug!(target: "assistd::tasks", "the user's conversation took the lane first");
            }
            Admitted::Run(run, lease) => {
                let started = self.runtime.task_queue.start(run, lease, |queued, lease| {
                    self.spawn_task_run(queued, lease);
                });
                if !started {
                    debug!(target: "assistd::tasks", %run, "admitted run left the queue first");
                }
            }
            Admitted::Turn(id, lane) => {
                let turn = self
                    .runtime
                    .conversation
                    .begin_turn(self.conversation_quiet_time());
                let start = TurnStart { lane, turn };
                if let Err(unstarted) = self.runtime.task_queue.start_turn(&id, start) {
                    debug!(target: "assistd::tasks", %id, "admitted user turn left the queue first");
                    unstarted.turn.abandon();
                }
            }
        }
    }

    /// Suspend every waiting run while the GPU monitor holds the daemon asleep or the
    /// user's conversation holds the only lane, and queue them again once both let go,
    /// recording each change.
    async fn suspend_waiting_runs(&self, gpu_held: bool) {
        let reason = if gpu_held {
            Some(SuspendReason::GpuBusy)
        } else {
            self.conversation_holds_queue()
                .then_some(SuspendReason::UserConversation)
        };
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

/// A lane for `next`; never returns without an item to admit.
async fn admit(lanes: &Arc<ModelLanes>, next: Option<NextStart>) -> Admitted {
    match next {
        None => pending().await,
        Some(NextStart::Run(run)) => Admitted::Run(run, lanes.admit(run).await),
        Some(NextStart::Turn(id)) => Admitted::Turn(id, lanes.admit_user_turn().await),
    }
}

/// When a quiet conversation's quiet time runs out; never for any other.
async fn quiet_ends(conversation: ConversationState) {
    match conversation {
        ConversationState::Quiet { until, .. } => tokio::time::sleep_until(until).await,
        ConversationState::Idle | ConversationState::Talking { .. } => pending().await,
    }
}
