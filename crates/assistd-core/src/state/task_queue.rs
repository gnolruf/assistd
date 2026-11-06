//! `TaskQueue`: the task runs and queued user turns waiting for a model lane,
//! in the order they start, and the runs running.

use std::collections::HashMap;

use parking_lot::Mutex as StdMutex;
use thiserror::Error;
use tokio::sync::{Mutex, MutexGuard, OwnedMutexGuard, oneshot, watch};

use assistd_ipc::{
    ConversationHold, LaneHolder, PausedPrompt, QueueItem, SuspendReason, TaskId,
    TaskQueueSnapshot, TaskRunId, TaskRunState,
};
use assistd_memory::{MemoryError, TaskRecord};

use super::conversation::ConversationTurn;
use super::lanes::LaneLease;

/// Why a change to the task queue was refused.
#[derive(Debug, Error)]
pub(in crate::state) enum QueueOpError {
    #[error("no task run {0} is queued or running")]
    NotQueued(TaskRunId),

    #[error("task run {0} is already running")]
    Running(TaskRunId),

    #[error("no user turn {0} is queued")]
    TurnNotQueued(String),

    #[error(transparent)]
    Memory(#[from] MemoryError),
}

/// Unfinished task runs: those running, then those waiting in the order they start.
#[derive(Debug, Default)]
pub(in crate::state) struct TaskQueue {
    state: watch::Sender<QueueState>,
    /// Held across a change to the waiting order and the write that persists it.
    order_writes: Mutex<()>,
    /// Where each resuming run waits for its lane.
    handoffs: StdMutex<HashMap<TaskRunId, oneshot::Sender<LaneLease>>>,
}

impl TaskQueue {
    /// Watch the queue, which changes with every run added, started, moved or ended.
    pub(in crate::state) fn subscribe(&self) -> watch::Receiver<QueueState> {
        self.state.subscribe()
    }

    /// Add `run` of `task` last, its events tagged with `request_id`.
    pub(in crate::state) fn push(&self, run: TaskRunId, task: TaskRecord, request_id: String) {
        self.state.send_modify(|state| {
            state.waiting.push(Waiting::Run(QueuedRun {
                run,
                task,
                request_id,
                suspended: None,
                stage: RunStage::Active,
            }));
        });
    }

    /// Queue the user turn of query `id`, which `start` is handed once it may run: behind
    /// the user turns already waiting, else last. Returns how many items are ahead of it.
    pub(in crate::state) fn push_turn(
        &self,
        id: String,
        preview: String,
        start: oneshot::Sender<TurnStart>,
    ) -> u32 {
        let mut behind = 0;
        self.state.send_modify(|state| {
            let at = state
                .waiting
                .iter()
                .rposition(|waiting| matches!(waiting, Waiting::Turn(_)))
                .map_or(state.waiting.len(), |last| last + 1);
            state
                .waiting
                .insert(at, Waiting::Turn(QueuedTurn { id, preview, start }));
            behind = u32::try_from(state.running.len() + at).unwrap_or(u32::MAX);
        });
        behind
    }

    pub(in crate::state) fn has_waiting_turns(&self) -> bool {
        self.state
            .borrow()
            .waiting
            .iter()
            .any(|waiting| matches!(waiting, Waiting::Turn(_)))
    }

    /// The waiting item that starts next: the first user turn or the first run that may
    /// start, whichever comes first, a held run never holding up a turn behind it.
    pub(in crate::state) fn next_to_start(&self) -> Option<NextStart> {
        let state = self.state.borrow();
        state.waiting.iter().find_map(|waiting| match waiting {
            Waiting::Turn(turn) => Some(NextStart::Turn(turn.id.clone())),
            Waiting::Run(queued) if !state.paused && queued.suspended.is_none() => {
                Some(NextStart::Run(queued.run))
            }
            Waiting::Run(_) => None,
        })
    }

    /// Take waiting user turn `id` out of the queue and hand it `start`, or give `start`
    /// back when the turn is no longer waiting or its query has gone.
    pub(in crate::state) fn start_turn(&self, id: &str, start: TurnStart) -> Result<(), TurnStart> {
        let mut taken = None;
        self.state.send_if_modified(|state| {
            let index = state.waiting.iter().position(|waiting| waiting.is_turn(id));
            taken = index.map(|index| state.waiting.remove(index));
            taken.is_some()
        });
        match taken {
            Some(Waiting::Turn(turn)) => turn.start.send(start),
            _ => Err(start),
        }
    }

    /// Drop waiting user turn `id`, whose query then ends without running.
    pub(in crate::state) fn cancel_turn(&self, id: &str) -> Result<(), QueueOpError> {
        let removed = self.state.send_if_modified(|state| {
            let before = state.waiting.len();
            state.waiting.retain(|waiting| !waiting.is_turn(id));
            state.waiting.len() != before
        });
        if removed {
            Ok(())
        } else {
            Err(QueueOpError::TurnNotQueued(id.to_string()))
        }
    }

    /// Move waiting run `run` to the running runs with `lease`, before any [`Self::cancel`]
    /// can see it: a resuming run is handed the lease, and any other is handed to `spawn`
    /// with it. `false` when it is no longer waiting to start.
    pub(in crate::state) fn start(
        &self,
        run: TaskRunId,
        lease: LaneLease,
        spawn: impl FnOnce(QueuedRun, LaneLease),
    ) -> bool {
        self.state.send_if_modified(|state| {
            let Some(index) = state.waiting.iter().position(|waiting| {
                waiting
                    .run()
                    .is_some_and(|queued| queued.run == run && queued.suspended.is_none())
            }) else {
                return false;
            };
            let Waiting::Run(mut queued) = state.waiting.remove(index) else {
                unreachable!("position matched a run");
            };
            match queued.stage {
                RunStage::Active => {
                    state.running.push(queued.clone());
                    spawn(queued, lease);
                }
                RunStage::Resuming | RunStage::NeedsInput(_) => {
                    queued.stage = RunStage::Active;
                    state.running.push(queued);
                    if let Some(handoff) = self.handoffs.lock().remove(&run) {
                        let _ = handoff.send(lease);
                    }
                }
            }
            true
        })
    }

    /// Take ended run `run` out of the queue.
    pub(in crate::state) fn finish(&self, run: TaskRunId) {
        self.handoffs.lock().remove(&run);
        self.state.send_modify(|state| {
            state.running.retain(|queued| queued.run != run);
            state.waiting.retain(|waiting| !waiting.is_run(run));
        });
    }

    /// Show running run `run` as paused on `prompt`, which nobody has answered yet.
    pub(in crate::state) fn mark_needs_input(&self, run: TaskRunId, prompt: PausedPrompt) {
        self.state.send_if_modified(|state| {
            let Some(queued) = state.running.iter_mut().find(|queued| queued.run == run) else {
                return false;
            };
            queued.stage = RunStage::NeedsInput(Box::new(prompt));
            true
        });
    }

    /// Put running run `run`, which gave up its lane to wait on a confirmation or the user's
    /// conversation, first among the waiting items; the receiver gets its lane once it starts
    /// again. `None` when it is not running.
    pub(in crate::state) fn resume_first(
        &self,
        run: TaskRunId,
    ) -> Option<oneshot::Receiver<LaneLease>> {
        let (handoff, lane) = oneshot::channel();
        let moved = self.state.send_if_modified(|state| {
            let Some(index) = state.running.iter().position(|queued| queued.run == run) else {
                return false;
            };
            let mut queued = state.running.remove(index);
            queued.stage = RunStage::Resuming;
            state.waiting.insert(0, Waiting::Run(queued));
            self.handoffs.lock().insert(run, handoff);
            true
        });
        moved.then_some(lane)
    }

    /// Drop waiting run `run`, or stop running or resuming run `run` with `stop`, which
    /// says whether it reached the run.
    pub(in crate::state) fn cancel(
        &self,
        run: TaskRunId,
        stop: impl FnOnce() -> bool,
    ) -> Cancelled {
        let mut outcome = Cancelled::NotFound;
        self.state.send_if_modified(|state| {
            if let Some(index) = state.waiting.iter().position(|waiting| waiting.is_run(run)) {
                let Waiting::Run(queued) = state.waiting.remove(index) else {
                    unreachable!("position matched a run");
                };
                outcome = if queued.stage == RunStage::Active {
                    Cancelled::Waiting
                } else {
                    self.handoffs.lock().remove(&run);
                    if stop() {
                        Cancelled::Running
                    } else {
                        Cancelled::NotFound
                    }
                };
                return true;
            }
            if state.running.iter().any(|queued| queued.run == run) && stop() {
                outcome = Cancelled::Running;
            }
            false
        });
        outcome
    }

    /// Drop every run of deleted task `task` that is waiting to start, whose rows went
    /// with it.
    pub(in crate::state) fn forget_task(&self, task: TaskId) {
        self.state.send_if_modified(|state| {
            let before = state.waiting.len();
            state.waiting.retain(|waiting| {
                waiting
                    .run()
                    .is_none_or(|queued| queued.task.id != task || queued.stage != RunStage::Active)
            });
            state.waiting.len() != before
        });
    }

    /// Hold the waiting order until the returned guard drops, so a change to it and
    /// the write persisting it are not interleaved with another's.
    pub(in crate::state) async fn lock_order(&self) -> MutexGuard<'_, ()> {
        self.order_writes.lock().await
    }

    /// Place waiting run `run` at index `to` of the waiting items, clamped to the
    /// last, and return the new order of the waiting runs.
    pub(in crate::state) fn move_to(
        &self,
        run: TaskRunId,
        to: u32,
    ) -> Result<Vec<TaskRunId>, QueueOpError> {
        let mut moved = Err(QueueOpError::NotQueued(run));
        self.state.send_if_modified(|state| {
            let Some(from) = state.waiting.iter().position(|waiting| waiting.is_run(run)) else {
                if state.running.iter().any(|queued| queued.run == run) {
                    moved = Err(QueueOpError::Running(run));
                }
                return false;
            };
            let queued = state.waiting.remove(from);
            let to =
                usize::try_from(to).map_or(state.waiting.len(), |to| to.min(state.waiting.len()));
            state.waiting.insert(to, queued);
            moved = Ok(state
                .waiting
                .iter()
                .filter_map(Waiting::run)
                .map(|queued| queued.run)
                .collect());
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
    /// when `None`, returning the runs not yet started whose state changed.
    pub(in crate::state) fn suspend_waiting(
        &self,
        reason: Option<SuspendReason>,
    ) -> Vec<TaskRunId> {
        let mut changed = Vec::new();
        self.state.send_if_modified(|state| {
            let mut modified = false;
            for queued in state.waiting.iter_mut().filter_map(Waiting::run_mut) {
                if queued.suspended != reason {
                    queued.suspended = reason;
                    modified = true;
                    if queued.stage == RunStage::Active {
                        changed.push(queued.run);
                    }
                }
            }
            modified
        });
        changed
    }

    /// The queue as the wire shows it, beside the lanes' `holders` and the conversation's
    /// `held`.
    pub(in crate::state) fn snapshot(
        &self,
        holders: Vec<LaneHolder>,
        held: Option<ConversationHold>,
    ) -> TaskQueueSnapshot {
        let state = self.state.borrow();
        let running = state.running.iter().map(|queued| match &queued.stage {
            RunStage::NeedsInput(prompt) => {
                queued.item(TaskRunState::NeedsInput, Some(PausedPrompt::clone(prompt)))
            }
            RunStage::Active | RunStage::Resuming => queued.item(TaskRunState::Running, None),
        });
        let waiting = state.waiting.iter().map(|waiting| match waiting {
            Waiting::Run(queued) => {
                let state = match queued.suspended {
                    Some(reason) => TaskRunState::Suspended { reason },
                    None => TaskRunState::Queued,
                };
                queued.item(state, None)
            }
            Waiting::Turn(turn) => QueueItem::UserTurn {
                id: turn.id.clone(),
                preview: turn.preview.clone(),
            },
        });
        TaskQueueSnapshot {
            items: running.chain(waiting).collect(),
            lanes: holders,
            paused: state.paused,
            held,
        }
    }
}

/// What a queued user turn is handed when it may run: the main lane, and its place in
/// the conversation, which holds the queue from then on.
#[derive(Debug)]
pub(in crate::state) struct TurnStart {
    pub(in crate::state) lane: OwnedMutexGuard<()>,
    pub(in crate::state) turn: ConversationTurn,
}

/// The waiting item [`TaskQueue::next_to_start`] picks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::state) enum NextStart {
    Run(TaskRunId),
    /// The user turn of the query with this request id.
    Turn(String),
}

/// One waiting item.
#[derive(Debug)]
enum Waiting {
    Run(QueuedRun),
    Turn(QueuedTurn),
}

impl Waiting {
    fn run(&self) -> Option<&QueuedRun> {
        match self {
            Waiting::Run(queued) => Some(queued),
            Waiting::Turn(_) => None,
        }
    }

    fn run_mut(&mut self) -> Option<&mut QueuedRun> {
        match self {
            Waiting::Run(queued) => Some(queued),
            Waiting::Turn(_) => None,
        }
    }

    fn is_run(&self, run: TaskRunId) -> bool {
        self.run().is_some_and(|queued| queued.run == run)
    }

    fn is_turn(&self, id: &str) -> bool {
        matches!(self, Waiting::Turn(turn) if turn.id == id)
    }
}

/// A user message waiting behind the queue, and where its start is handed.
#[derive(Debug)]
struct QueuedTurn {
    id: String,
    preview: String,
    start: oneshot::Sender<TurnStart>,
}

/// One unfinished run, with the request id its events are tagged with.
#[derive(Debug, Clone)]
pub(in crate::state) struct QueuedRun {
    pub(in crate::state) run: TaskRunId,
    pub(in crate::state) task: TaskRecord,
    pub(in crate::state) request_id: String,
    /// Why a waiting run may not start yet; `None` once it may.
    suspended: Option<SuspendReason>,
    stage: RunStage,
}

/// How far a run has got through its turn, which it may pause partway.
#[derive(Debug, Clone, PartialEq, Eq)]
enum RunStage {
    /// Waiting to start, or running unpaused.
    Active,
    /// Paused on this confirmation, which nobody has answered, its lane given up.
    NeedsInput(Box<PausedPrompt>),
    /// Answered, and waiting for a lane to carry on its turn.
    Resuming,
}

impl QueuedRun {
    fn item(&self, state: TaskRunState, prompt: Option<PausedPrompt>) -> QueueItem {
        QueueItem::Run {
            run: self.run,
            task: self.task.id,
            task_name: self.task.spec.name.clone(),
            state,
            prompt,
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

/// The runs and user turns a [`TaskQueue`] holds and whether it is paused.
#[derive(Debug, Default)]
pub(in crate::state) struct QueueState {
    running: Vec<QueuedRun>,
    waiting: Vec<Waiting>,
    paused: bool,
}
