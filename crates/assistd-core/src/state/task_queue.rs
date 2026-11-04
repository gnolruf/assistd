//! `TaskQueue`: the task runs waiting for a model lane, in the order they
//! start, and those running.

use thiserror::Error;
use tokio::sync::{Mutex, MutexGuard, watch};

use assistd_ipc::{
    LaneHolder, SuspendReason, TaskId, TaskQueueItem, TaskQueueSnapshot, TaskRunId, TaskRunState,
};
use assistd_memory::{MemoryError, TaskRecord};

/// Why a change to the task queue was refused.
#[derive(Debug, Error)]
pub(in crate::state) enum QueueOpError {
    #[error("no task run {0} is queued or running")]
    NotQueued(TaskRunId),

    #[error("task run {0} is already running")]
    Running(TaskRunId),

    #[error(transparent)]
    Memory(#[from] MemoryError),
}

/// Unfinished task runs: those running, then those waiting in the order they start.
#[derive(Debug, Default)]
pub(in crate::state) struct TaskQueue {
    state: watch::Sender<QueueState>,
    /// Held across a change to the waiting order and the write that persists it.
    order_writes: Mutex<()>,
}

impl TaskQueue {
    /// Watch the queue, which changes with every run added, started, moved or ended.
    pub(in crate::state) fn subscribe(&self) -> watch::Receiver<QueueState> {
        self.state.subscribe()
    }

    /// Add `run` of `task` last, its events tagged with `request_id`.
    pub(in crate::state) fn push(&self, run: TaskRunId, task: TaskRecord, request_id: String) {
        self.state.send_modify(|state| {
            state.waiting.push(QueuedRun {
                run,
                task,
                request_id,
                suspended: None,
            });
        });
    }

    /// The waiting run that starts next, `None` while paused or none may start.
    pub(in crate::state) fn next_to_start(&self) -> Option<TaskRunId> {
        let state = self.state.borrow();
        if state.paused {
            return None;
        }
        state
            .waiting
            .iter()
            .find(|queued| queued.suspended.is_none())
            .map(|queued| queued.run)
    }

    /// Move waiting run `run` to the running runs and hand it to `spawn` before any
    /// [`Self::cancel`] can see it; `false` when it is no longer waiting to start.
    pub(in crate::state) fn start(&self, run: TaskRunId, spawn: impl FnOnce(QueuedRun)) -> bool {
        self.state.send_if_modified(|state| {
            let Some(index) = state
                .waiting
                .iter()
                .position(|queued| queued.run == run && queued.suspended.is_none())
            else {
                return false;
            };
            let queued = state.waiting.remove(index);
            state.running.push(queued.clone());
            spawn(queued);
            true
        })
    }

    /// Take ended run `run` out of the running runs.
    pub(in crate::state) fn finish(&self, run: TaskRunId) {
        self.state.send_modify(|state| {
            state.running.retain(|queued| queued.run != run);
        });
    }

    /// Drop waiting run `run`, or stop running run `run` with `stop`, which says
    /// whether it reached the run.
    pub(in crate::state) fn cancel(
        &self,
        run: TaskRunId,
        stop: impl FnOnce() -> bool,
    ) -> Cancelled {
        let mut outcome = Cancelled::NotFound;
        self.state.send_if_modified(|state| {
            if let Some(index) = state.waiting.iter().position(|queued| queued.run == run) {
                state.waiting.remove(index);
                outcome = Cancelled::Waiting;
                return true;
            }
            if state.running.iter().any(|queued| queued.run == run) && stop() {
                outcome = Cancelled::Running;
            }
            false
        });
        outcome
    }

    /// Drop every waiting run of deleted task `task`, whose rows went with it.
    pub(in crate::state) fn forget_task(&self, task: TaskId) {
        self.state.send_if_modified(|state| {
            let before = state.waiting.len();
            state.waiting.retain(|queued| queued.task.id != task);
            state.waiting.len() != before
        });
    }

    /// Hold the waiting order until the returned guard drops, so a change to it and
    /// the write persisting it are not interleaved with another's.
    pub(in crate::state) async fn lock_order(&self) -> MutexGuard<'_, ()> {
        self.order_writes.lock().await
    }

    /// Place waiting run `run` at index `to` of the waiting runs, clamped to the
    /// last, and return the new waiting order.
    pub(in crate::state) fn move_to(
        &self,
        run: TaskRunId,
        to: u32,
    ) -> Result<Vec<TaskRunId>, QueueOpError> {
        let mut moved = Err(QueueOpError::NotQueued(run));
        self.state.send_if_modified(|state| {
            let Some(from) = state.waiting.iter().position(|queued| queued.run == run) else {
                if state.running.iter().any(|queued| queued.run == run) {
                    moved = Err(QueueOpError::Running(run));
                }
                return false;
            };
            let queued = state.waiting.remove(from);
            let to =
                usize::try_from(to).map_or(state.waiting.len(), |to| to.min(state.waiting.len()));
            state.waiting.insert(to, queued);
            moved = Ok(state.waiting.iter().map(|queued| queued.run).collect());
            from != to
        });
        moved
    }

    pub(in crate::state) fn set_paused(&self, paused: bool) {
        self.state.send_if_modified(|state| {
            let changed = state.paused != paused;
            state.paused = paused;
            changed
        });
    }

    /// Suspend every waiting run for `reason`, or let every suspended one start again
    /// when `None`, returning the runs whose state changed.
    pub(in crate::state) fn suspend_waiting(
        &self,
        reason: Option<SuspendReason>,
    ) -> Vec<TaskRunId> {
        let mut changed = Vec::new();
        self.state.send_if_modified(|state| {
            for queued in &mut state.waiting {
                if queued.suspended != reason {
                    queued.suspended = reason;
                    changed.push(queued.run);
                }
            }
            !changed.is_empty()
        });
        changed
    }

    /// The queue as the wire shows it, beside the lanes' `holders`.
    pub(in crate::state) fn snapshot(&self, holders: Vec<LaneHolder>) -> TaskQueueSnapshot {
        let state = self.state.borrow();
        let running = state
            .running
            .iter()
            .map(|queued| queued.item(TaskRunState::Running));
        let waiting = state.waiting.iter().map(|queued| {
            queued.item(match queued.suspended {
                Some(reason) => TaskRunState::Suspended { reason },
                None => TaskRunState::Queued,
            })
        });
        TaskQueueSnapshot {
            items: running.chain(waiting).collect(),
            lanes: holders,
            paused: state.paused,
        }
    }
}

/// One unfinished run, with the request id its events are tagged with.
#[derive(Debug, Clone)]
pub(in crate::state) struct QueuedRun {
    pub(in crate::state) run: TaskRunId,
    pub(in crate::state) task: TaskRecord,
    pub(in crate::state) request_id: String,
    /// Why a waiting run may not start yet; `None` once it may.
    suspended: Option<SuspendReason>,
}

impl QueuedRun {
    fn item(&self, state: TaskRunState) -> TaskQueueItem {
        TaskQueueItem {
            run: self.run,
            task: self.task.id,
            task_name: self.task.spec.name.clone(),
            state,
        }
    }
}

/// How [`TaskQueue::cancel`] dealt with a run.
#[derive(Debug)]
pub(in crate::state) enum Cancelled {
    /// It was waiting and is now out of the queue.
    Waiting,
    /// It was running and has been asked to stop.
    Running,
    NotFound,
}

/// The runs a [`TaskQueue`] holds and whether it is paused.
#[derive(Debug, Default)]
pub(in crate::state) struct QueueState {
    running: Vec<QueuedRun>,
    waiting: Vec<QueuedRun>,
    paused: bool,
}
