//! The task queue's dispatcher, which starts queued runs and user turns as model
//! lanes free up and holds runs for the user's conversation, and the `TaskQueue` handler.

use std::future::pending;
use std::sync::Arc;

use tokio::sync::OwnedMutexGuard;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use assistd_ipc::{Event, EventBody, SuspendReason, TaskQueueOp, TaskQueueSnapshot, TaskRunId};
use assistd_memory::{TaskRunEnd, WaitingRun};

use super::conversation::ConversationState;
use super::lanes::{LaneLease, ModelLanes, changed_or_pending};
use super::send_error;
use super::task_queue::{Cancelled, NextStart, QueueOpError, TurnStart};
use super::task_runs::DAEMON_STOPPED;
use super::task_service::TaskService;
use crate::bus::EventSink;

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

    pub(super) async fn handle_task_queue(&self, id: String, op: TaskQueueOp, tx: EventSink) {
        match self.apply_task_queue_op(op).await {
            Ok(()) => {
                let _ = tx.send(Event::new(id, EventBody::Done)).await;
            }
            Err(e) => send_error(&tx, id, e.to_string()).await,
        }
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

    async fn apply_task_queue_op(&self, op: TaskQueueOp) -> Result<(), QueueOpError> {
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
                self.deps.admission.conversation.end();
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
    /// GPU monitor or the user's conversation does, end the conversation once it has been
    /// quiet long enough, and broadcast the queue after every change, until `stopping` fires.
    async fn dispatch_queued_runs(self: Arc<Self>, stopping: CancellationToken) {
        let mut queue_changes = self.deps.admission.task_queue.subscribe();
        let mut lane_changes = self.deps.admission.lanes.watch();
        let mut gpu_hold = self.deps.presence.gpu_hold();
        let mut conversation = self.deps.admission.conversation.subscribe();
        let mut published = None;
        loop {
            let gpu_held = *gpu_hold.borrow_and_update();
            let conversation_now = *conversation.borrow_and_update();
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
                () = changed_or_pending(&mut conversation) => {}
                () = quiet_ends(conversation_now) => {
                    self.deps.admission.conversation.end_if_quiet_by(Instant::now());
                }
                admitted = admit(&self.deps.admission.lanes, next) => self.start_admitted(admitted),
            }
        }
    }

    /// Start the waiting item a lane was found for, unless it stopped waiting meanwhile or
    /// the user's conversation took the main lane first.
    fn start_admitted(self: &Arc<Self>, admitted: Admitted) {
        match admitted {
            Admitted::Run(_, lease)
                if lease.on_main_lane() && self.deps.admission.conversation_holds_queue() =>
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
                    .conversation
                    .begin_turn(self.deps.admission.conversation_quiet_time());
                let start = TurnStart { lane, turn };
                if let Err(unstarted) = self.deps.admission.task_queue.start_turn(&id, start) {
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
            self.deps
                .admission
                .conversation_holds_queue()
                .then_some(SuspendReason::UserConversation)
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

/// When a quiet conversation's quiet time runs out; never for any other.
async fn quiet_ends(conversation: ConversationState) {
    match conversation {
        ConversationState::Quiet { until, .. } => tokio::time::sleep_until(until).await,
        ConversationState::Idle | ConversationState::Talking { .. } => pending().await,
    }
}
