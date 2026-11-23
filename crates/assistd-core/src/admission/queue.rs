//! `TaskQueue`: what holds each model lane, everything waiting for one in the order
//! it starts, and the task runs running. One function says which waiting entries may
//! take a lane, and every change to the queue grants them theirs.

use std::collections::BTreeMap;
use std::future::pending;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use chrono::{DateTime, Utc};
use thiserror::Error;
use tokio::sync::{Mutex, MutexGuard, mpsc, oneshot, watch};

use assistd_config::TaskInterruption;
use assistd_ipc::{
    PausedPrompt, QueueItem, SuspendReason, TaskId, TaskQueueSnapshot, TaskRunId, TaskRunState,
};
use assistd_memory::{MemoryError, TaskRecord};

use super::activity::{Activity, ActivityWatch};
use super::interaction::InteractionState;
use super::lanes::{LaneLease, LaneUse, MAIN_LANE};
use crate::presence::SettledPresence;

/// Why a change to the task queue was refused.
#[derive(Debug, Error)]
pub enum QueueOpError {
    #[error("no task run {0} is queued or running")]
    NotQueued(TaskRunId),

    #[error("task run {0} is already running")]
    Running(TaskRunId),

    #[error("no user turn {0} is queued")]
    TurnNotQueued(String),

    #[error(transparent)]
    Memory(#[from] MemoryError),
}

/// The model's lanes and the queue for them: user turns, prompt refines, task runs and
/// session titles waiting for a lane, and the unfinished runs that hold or have held one.
#[derive(Debug)]
pub(crate) struct TaskQueue {
    state: watch::Sender<QueueState>,
    /// Held across a change to the waiting order and the write that persists it.
    order_writes: Mutex<()>,
    interaction: watch::Receiver<InteractionState>,
    slots: watch::Receiver<NonZeroU32>,
    presence: watch::Receiver<SettledPresence>,
    next_ticket: AtomicU64,
    granted_runs: mpsc::UnboundedSender<RunStart>,
    runs_to_start: Mutex<mpsc::UnboundedReceiver<RunStart>>,
}

impl TaskQueue {
    /// A queue for as many lanes as `slots` reports, read beside the user's
    /// `interaction` and presence's settled state `presence`.
    pub(super) fn new(
        interaction: watch::Receiver<InteractionState>,
        slots: watch::Receiver<NonZeroU32>,
        presence: watch::Receiver<SettledPresence>,
    ) -> Arc<Self> {
        let (granted_runs, runs_to_start) = mpsc::unbounded_channel();
        Arc::new(Self {
            state: watch::Sender::default(),
            order_writes: Mutex::default(),
            interaction,
            slots,
            presence,
            next_ticket: AtomicU64::default(),
            granted_runs,
            runs_to_start: Mutex::new(runs_to_start),
        })
    }

    /// Watch the queue, which changes with every entry added, started, moved or ended.
    pub(crate) fn subscribe(&self) -> watch::Receiver<QueueState> {
        self.state.subscribe()
    }

    pub(super) fn watch_activity(&self) -> ActivityWatch {
        ActivityWatch::new(
            self.interaction.clone(),
            self.subscribe(),
            self.slots.clone(),
            self.presence.clone(),
        )
    }

    /// Add `run` of `task`, which request `request_id` queued, last.
    pub(crate) fn push(self: &Arc<Self>, run: TaskRunId, task: TaskRecord, request_id: String) {
        self.state.send_modify(|state| {
            state.waiting.push(Waiting::Run {
                queued: QueuedRun {
                    run,
                    task,
                    request_id,
                    suspended: None,
                    paused_on: None,
                    started_at: None,
                },
                resume: None,
            });
        });
        self.grant_lanes();
    }

    /// Queue the user turn of query `id` where `interruption` puts it: ahead of the
    /// waiting runs, or behind them when it waits for the queue.
    pub(super) fn queue_turn(
        self: &Arc<Self>,
        id: String,
        preview: String,
        interruption: TaskInterruption,
    ) -> WaitingTurn {
        let (waiter, (place, behind)) = self.wait_for_lane(|state, activity, waiter| {
            let place = state.turn_place(activity, interruption);
            let at = match place {
                TurnPlace::Front => state.front(),
                TurnPlace::BehindQueue => state.behind_waiting_turns(),
            };
            let behind = state.ahead_for_main_lane(at, activity);
            state.waiting.insert(
                at,
                Waiting::Turn(QueuedTurn {
                    id,
                    preview,
                    waiter,
                }),
            );
            (place, behind)
        });
        WaitingTurn {
            waiter,
            place,
            behind,
        }
    }

    /// Queue a task prompt's refine behind the waiting user turns and refines, else first.
    pub(crate) fn queue_refine(self: &Arc<Self>) -> LaneWaiter {
        let (waiter, ()) = self.wait_for_lane(|state, _, waiter| {
            state.waiting.insert(state.front(), Waiting::Refine(waiter));
        });
        waiter
    }

    /// Queue a session title's generation last.
    pub(crate) fn queue_title(self: &Arc<Self>) -> LaneWaiter {
        let (waiter, ()) = self.wait_for_lane(|state, _, waiter| {
            state.waiting.push(Waiting::Title(waiter));
        });
        waiter
    }

    fn wait_for_lane<T: Default>(
        self: &Arc<Self>,
        insert: impl FnOnce(&mut QueueState, &Activity, Waiter) -> T,
    ) -> (LaneWaiter, T) {
        let ticket = self.next_ticket.fetch_add(1, Ordering::Relaxed);
        let (lease, granted) = oneshot::channel();
        let mut inserted = T::default();
        self.state.send_modify(|state| {
            let activity = self.activity_over(state);
            inserted = insert(state, &activity, Waiter { ticket, lease });
        });
        self.grant_lanes();
        let waiter = LaneWaiter {
            queue: Arc::clone(self),
            ticket,
            granted,
        };
        (waiter, inserted)
    }

    /// Give a lane to every waiting entry that may take one now.
    pub(crate) fn grant_lanes(self: &Arc<Self>) {
        let mut granted = Vec::new();
        self.state.send_if_modified(|state| {
            let activity = self.activity_over(state);
            for (index, lane) in state.startable(&activity).into_iter().rev() {
                granted.push(state.start(index, LaneLease::new(lane, Arc::clone(self))));
            }
            !granted.is_empty()
        });
        for grant in granted.into_iter().rev() {
            match grant {
                Granted::Waiter(waiter, lease) => drop(waiter.send(lease)),
                Granted::Run(start) => drop(self.granted_runs.send(start)),
            }
        }
    }

    pub(super) fn release(self: &Arc<Self>, lane: u32) {
        self.state.send_modify(|state| {
            state.lanes.remove(&lane);
        });
        self.grant_lanes();
    }

    fn forget_waiter(&self, ticket: u64) {
        self.state.send_if_modified(|state| {
            let before = state.waiting.len();
            state.waiting.retain(|waiting| {
                waiting
                    .waiter()
                    .is_none_or(|waiter| waiter.ticket != ticket)
            });
            state.waiting.len() != before
        });
    }

    fn activity_over(&self, state: &QueueState) -> Activity {
        Activity::new(
            *self.interaction.borrow(),
            state.lanes.clone(),
            *self.slots.borrow(),
            *self.presence.borrow(),
        )
    }

    /// The next run granted its first lane. Cancel-safe.
    pub(crate) async fn granted_run(&self) -> RunStart {
        match self.runs_to_start.lock().await.recv().await {
            Some(start) => start,
            None => pending().await,
        }
    }

    /// Hand `start` to `spawn`, before any [`Self::cancel`] can find it unstarted; a
    /// run cancelled since it was granted gives its lane back instead.
    pub(crate) fn start_run(&self, start: RunStart, spawn: impl FnOnce(QueuedRun, LaneLease)) {
        let cancelled = {
            let state = self.state.borrow();
            if state.is_running(start.queued.run) {
                spawn(start.queued, start.lease);
                None
            } else {
                Some(start)
            }
        };
        drop(cancelled);
    }

    /// Drop waiting user turn `id`, whose query then ends without running.
    pub(crate) fn cancel_turn(&self, id: &str) -> Result<(), QueueOpError> {
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

    /// Take ended run `run` out of the queue.
    pub(crate) fn finish(&self, run: TaskRunId) {
        self.state.send_modify(|state| {
            state.running.retain(|queued| queued.run != run);
            state.waiting.retain(|waiting| !waiting.is_run(run));
        });
    }

    /// Show running run `run` as paused on `prompt`, which nobody has answered yet.
    pub(crate) fn mark_needs_input(&self, run: TaskRunId, prompt: PausedPrompt) {
        self.state.send_if_modified(|state| {
            let Some(queued) = state.running.iter_mut().find(|queued| queued.run == run) else {
                return false;
            };
            queued.paused_on = Some(Box::new(prompt));
            true
        });
    }

    /// Put running run `run`, which gave up its lane to wait on a confirmation or the user's
    /// interaction, first among the waiting entries; the receiver gets its lane once it
    /// starts again. `None` when it is not running.
    pub(crate) fn resume_first(
        self: &Arc<Self>,
        run: TaskRunId,
    ) -> Option<oneshot::Receiver<LaneLease>> {
        let (resume, lane) = oneshot::channel();
        let mut resume = Some(resume);
        let moved = self.state.send_if_modified(|state| {
            let Some(index) = state.running.iter().position(|queued| queued.run == run) else {
                return false;
            };
            let mut queued = state.running.remove(index);
            queued.paused_on = None;
            let resume = resume.take();
            state.waiting.insert(0, Waiting::Run { queued, resume });
            true
        });
        self.grant_lanes();
        moved.then_some(lane)
    }

    /// Drop run `run` when it has not started, or stop it with `stop`, which says whether
    /// it reached the run, when it is running or resuming.
    pub(crate) fn cancel(&self, run: TaskRunId, stop: impl FnOnce() -> bool) -> Cancelled {
        let mut outcome = Cancelled::NotFound;
        self.state.send_if_modified(|state| {
            if let Some(resume) = state.take_waiting_run(run) {
                outcome = match resume {
                    None => Cancelled::Waiting,
                    Some(_) if stop() => Cancelled::Running,
                    Some(_) => Cancelled::NotFound,
                };
                return true;
            }
            let Some(index) = state.running.iter().position(|queued| queued.run == run) else {
                return false;
            };
            if stop() {
                outcome = Cancelled::Running;
                return false;
            }
            state.running.remove(index);
            outcome = Cancelled::Waiting;
            true
        });
        outcome
    }

    /// Drop every run of deleted task `task` that has never started, whose rows went
    /// with it.
    pub(crate) fn forget_task(&self, task: TaskId) {
        self.state.send_if_modified(|state| {
            let before = state.waiting.len();
            state.waiting.retain(|waiting| {
                waiting
                    .unstarted_run()
                    .is_none_or(|queued| queued.task.id != task)
            });
            state.waiting.len() != before
        });
    }

    /// Hold the waiting order until the returned guard drops, so a change to it and
    /// the write persisting it are not interleaved with another's.
    pub(crate) async fn lock_order(&self) -> MutexGuard<'_, ()> {
        self.order_writes.lock().await
    }

    /// Place waiting run `run` at index `to` of the listed waiting items, clamped to the
    /// last, and return the new order of the waiting runs.
    pub(crate) fn move_to(&self, run: TaskRunId, to: u32) -> Result<Vec<TaskRunId>, QueueOpError> {
        let mut moved = Err(QueueOpError::NotQueued(run));
        self.state.send_if_modified(|state| {
            let Some(from) = state.waiting.iter().position(|waiting| waiting.is_run(run)) else {
                if state.is_running(run) {
                    moved = Err(QueueOpError::Running(run));
                }
                return false;
            };
            let queued = state.waiting.remove(from);
            let to = state.listed_index(to);
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

    pub(crate) fn set_paused(self: &Arc<Self>, paused: bool) {
        self.state.send_if_modified(|state| {
            let changed = state.paused != paused;
            state.paused = paused;
            changed
        });
        self.grant_lanes();
    }

    /// Mark every waiting run suspended for `reason`, or not suspended when `None`,
    /// returning the runs never started whose mark changed.
    pub(crate) fn suspend_waiting(&self, reason: Option<SuspendReason>) -> Vec<TaskRunId> {
        let mut changed = Vec::new();
        self.state.send_if_modified(|state| {
            let mut modified = false;
            for waiting in &mut state.waiting {
                let Waiting::Run { queued, resume } = waiting else {
                    continue;
                };
                if queued.suspended != reason {
                    queued.suspended = reason;
                    modified = true;
                    if resume.is_none() {
                        changed.push(queued.run);
                    }
                }
            }
            modified
        });
        changed
    }

    /// The queue as the wire shows it, refines and titles left out, beside the lanes'
    /// holders and the interaction's hold in `activity`.
    pub(super) fn snapshot(&self, activity: &Activity) -> TaskQueueSnapshot {
        let state = self.state.borrow();
        let running = state.running.iter().map(|queued| match &queued.paused_on {
            Some(prompt) => {
                queued.item(TaskRunState::NeedsInput, Some(PausedPrompt::clone(prompt)))
            }
            None => queued.item(TaskRunState::Running, None),
        });
        let waiting = state.waiting.iter().filter_map(Waiting::item);
        TaskQueueSnapshot {
            items: running.chain(waiting).collect(),
            lanes: activity.lane_holders(),
            paused: state.paused,
            held: activity.interaction_hold(),
        }
    }
}

/// A place in the queue for a lane, given up when dropped.
#[derive(Debug)]
pub(crate) struct LaneWaiter {
    queue: Arc<TaskQueue>,
    ticket: u64,
    granted: oneshot::Receiver<LaneLease>,
}

impl LaneWaiter {
    /// The lane, once granted; `None` when the entry was taken out of the queue first.
    /// Cancel-safe.
    pub(crate) async fn granted(mut self) -> Option<LaneLease> {
        (&mut self.granted).await.ok()
    }
}

impl Drop for LaneWaiter {
    fn drop(&mut self) {
        self.queue.forget_waiter(self.ticket);
    }
}

/// A user turn in the queue.
#[derive(Debug)]
pub(super) struct WaitingTurn {
    pub(super) waiter: LaneWaiter,
    pub(super) place: TurnPlace,
    /// How many holders and waiting entries get the main lane before it.
    pub(super) behind: u32,
}

/// Where a user turn's entry went in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum TurnPlace {
    /// Ahead of the waiting runs; the interaction opens on arrival.
    #[default]
    Front,
    /// Behind the runs waiting when it arrived; the interaction opens once it is granted.
    BehindQueue,
}

/// A run granted its first lane, for the dispatcher to start.
#[derive(Debug)]
pub(crate) struct RunStart {
    queued: QueuedRun,
    lease: LaneLease,
}

/// A lane on its way to the entry granted it.
enum Granted {
    Waiter(oneshot::Sender<LaneLease>, LaneLease),
    Run(RunStart),
}

/// Where a waiting entry's lane is sent, and the ticket its [`LaneWaiter`] drops it by.
#[derive(Debug)]
struct Waiter {
    ticket: u64,
    lease: oneshot::Sender<LaneLease>,
}

/// One waiting entry.
#[derive(Debug)]
enum Waiting {
    Run {
        queued: QueuedRun,
        /// Where a run that gave up its lane mid-turn gets its next; `None` for a run
        /// that has never started.
        resume: Option<oneshot::Sender<LaneLease>>,
    },
    Turn(QueuedTurn),
    Refine(Waiter),
    Title(Waiter),
}

impl Waiting {
    fn run(&self) -> Option<&QueuedRun> {
        match self {
            Waiting::Run { queued, .. } => Some(queued),
            Waiting::Turn(_) | Waiting::Refine(_) | Waiting::Title(_) => None,
        }
    }

    fn unstarted_run(&self) -> Option<&QueuedRun> {
        match self {
            Waiting::Run {
                queued,
                resume: None,
            } => Some(queued),
            _ => None,
        }
    }

    fn waiter(&self) -> Option<&Waiter> {
        match self {
            Waiting::Turn(turn) => Some(&turn.waiter),
            Waiting::Refine(waiter) | Waiting::Title(waiter) => Some(waiter),
            Waiting::Run { .. } => None,
        }
    }

    fn is_run(&self, run: TaskRunId) -> bool {
        self.run().is_some_and(|queued| queued.run == run)
    }

    fn is_turn(&self, id: &str) -> bool {
        matches!(self, Waiting::Turn(turn) if turn.id == id)
    }

    fn lane_use(&self) -> LaneUse {
        match self {
            Waiting::Run { queued, .. } => LaneUse::Run(queued.run),
            Waiting::Turn(_) => LaneUse::UserTurn,
            Waiting::Refine(_) => LaneUse::Refine,
            Waiting::Title(_) => LaneUse::Title,
        }
    }

    /// Whether the entry competes for the main lane: a user turn, a refine, and a run
    /// when the main lane is the only one. A title only takes it when nothing else does.
    fn contends_for_main_lane(&self, activity: &Activity) -> bool {
        match self {
            Waiting::Turn(_) | Waiting::Refine(_) => true,
            Waiting::Run { .. } => activity.single_lane(),
            Waiting::Title(_) => false,
        }
    }

    /// The entry as the wire lists it; `None` for a refine or a title.
    fn item(&self) -> Option<QueueItem> {
        match self {
            Waiting::Run { queued, .. } => {
                let state = match queued.suspended {
                    Some(reason) => TaskRunState::Suspended { reason },
                    None => TaskRunState::Queued,
                };
                Some(queued.item(state, None))
            }
            Waiting::Turn(turn) => Some(QueueItem::UserTurn {
                id: turn.id.clone(),
                preview: turn.preview.clone(),
            }),
            Waiting::Refine(_) | Waiting::Title(_) => None,
        }
    }
}

/// A user message waiting for the main lane.
#[derive(Debug)]
struct QueuedTurn {
    id: String,
    preview: String,
    waiter: Waiter,
}

/// One unfinished run, with the id of the request that queued it.
#[derive(Debug, Clone)]
pub(crate) struct QueuedRun {
    pub(crate) run: TaskRunId,
    pub(crate) task: TaskRecord,
    pub(crate) request_id: String,
    /// Why a waiting run is shown as held; `None` once it may start.
    suspended: Option<SuspendReason>,
    /// The confirmation a running run paused on, its lane given up, until it is answered.
    paused_on: Option<Box<PausedPrompt>>,
    /// When it first took a lane; `None` while it has never run.
    started_at: Option<DateTime<Utc>>,
}

impl QueuedRun {
    fn item(&self, state: TaskRunState, prompt: Option<PausedPrompt>) -> QueueItem {
        QueueItem::Run {
            run: self.run,
            task: self.task.id,
            task_name: self.task.spec.name.clone(),
            state,
            prompt,
            started_at: self.started_at,
        }
    }
}

/// How [`TaskQueue::cancel`] dealt with a run.
#[derive(Debug)]
pub(crate) enum Cancelled {
    /// It had not started and is now out of the queue.
    Waiting,
    /// It was running and has been asked to stop.
    Running,
    NotFound,
}

/// What a [`TaskQueue`] holds: each held lane's holder, the runs running, the waiting
/// entries in the order they start, and whether its runs are paused.
#[derive(Debug, Default)]
pub(crate) struct QueueState {
    lanes: BTreeMap<u32, LaneUse>,
    running: Vec<QueuedRun>,
    waiting: Vec<Waiting>,
    paused: bool,
}

impl QueueState {
    pub(super) fn lane_holders(&self) -> &BTreeMap<u32, LaneUse> {
        &self.lanes
    }

    /// The index and lane of each waiting entry that may start now. Turns and refines take
    /// the main lane; runs take it when it is the only one, else another, unless held or
    /// paused; a title takes it when nothing else does while the daemon is `Active`.
    fn startable(&self, activity: &Activity) -> Vec<(usize, u32)> {
        let runs_held = self.paused || activity.waiting_runs_held().is_some();
        let mut free = activity.free_lanes();
        let mut starts = Vec::new();
        for (index, waiting) in self.waiting.iter().enumerate() {
            let lane = match waiting {
                Waiting::Run { .. } if runs_held => None,
                Waiting::Run { .. } if !activity.single_lane() => {
                    let task_lane = free.iter().copied().find(|&lane| lane != MAIN_LANE);
                    task_lane.and_then(|lane| free.take(&lane))
                }
                Waiting::Run { .. } | Waiting::Turn(_) | Waiting::Refine(_) => {
                    free.take(&MAIN_LANE)
                }
                Waiting::Title(_) => None,
            };
            starts.extend(lane.map(|lane| (index, lane)));
        }
        if free.contains(&MAIN_LANE) && activity.settled_active() {
            let title = self
                .waiting
                .iter()
                .position(|waiting| matches!(waiting, Waiting::Title(_)));
            starts.extend(title.map(|index| (index, MAIN_LANE)));
            starts.sort_unstable();
        }
        starts
    }

    /// Take the entry at `index` out of the waiting ones as the holder of `lease`'s lane.
    fn start(&mut self, index: usize, lease: LaneLease) -> Granted {
        let waiting = self.waiting.remove(index);
        self.lanes.insert(lease.lane(), waiting.lane_use());
        match waiting {
            Waiting::Turn(QueuedTurn { waiter, .. })
            | Waiting::Refine(waiter)
            | Waiting::Title(waiter) => Granted::Waiter(waiter.lease, lease),
            Waiting::Run { mut queued, resume } => {
                queued.started_at.get_or_insert_with(Utc::now);
                queued.suspended = None;
                self.running.push(queued.clone());
                match resume {
                    Some(resume) => Granted::Waiter(resume, lease),
                    None => Granted::Run(RunStart { queued, lease }),
                }
            }
        }
    }

    /// Behind the queue under `after_queue` when the interaction is closed and a run
    /// holds the main lane or user turns already wait; at the front otherwise.
    fn turn_place(&self, activity: &Activity, interruption: TaskInterruption) -> TurnPlace {
        let turns_wait = self
            .waiting
            .iter()
            .any(|waiting| matches!(waiting, Waiting::Turn(_)));
        if interruption == TaskInterruption::AfterQueue
            && !activity.interaction_open()
            && (activity.main_held_by_run() || turns_wait)
        {
            TurnPlace::BehindQueue
        } else {
            TurnPlace::Front
        }
    }

    /// The index behind the waiting user turns and refines, else the first.
    fn front(&self) -> usize {
        self.waiting
            .iter()
            .rposition(|waiting| matches!(waiting, Waiting::Turn(_) | Waiting::Refine(_)))
            .map_or(0, |last| last + 1)
    }

    /// The index behind the waiting user turns, else the last.
    fn behind_waiting_turns(&self) -> usize {
        self.waiting
            .iter()
            .rposition(|waiting| matches!(waiting, Waiting::Turn(_)))
            .map_or(self.waiting.len(), |last| last + 1)
    }

    /// How many get the main lane before an entry going in at index `at`: its holder,
    /// and the entries ahead that contend for it.
    fn ahead_for_main_lane(&self, at: usize, activity: &Activity) -> u32 {
        let waiting = self.waiting[..at]
            .iter()
            .filter(|waiting| waiting.contends_for_main_lane(activity))
            .count();
        let ahead = waiting + usize::from(!activity.main_lane_free());
        u32::try_from(ahead).unwrap_or(u32::MAX)
    }

    /// The index among all waiting entries of index `to` among the listed ones, clamped
    /// to the last.
    fn listed_index(&self, to: u32) -> usize {
        let listed = self
            .waiting
            .iter()
            .enumerate()
            .filter(|(_, waiting)| waiting.item().is_some())
            .map(|(index, _)| index);
        usize::try_from(to)
            .ok()
            .and_then(|to| listed.clone().nth(to))
            .unwrap_or(self.waiting.len())
    }

    fn is_running(&self, run: TaskRunId) -> bool {
        self.running.iter().any(|queued| queued.run == run)
    }

    /// Take waiting run `run` out, returning where it would have resumed; `None` when
    /// it is not waiting.
    fn take_waiting_run(&mut self, run: TaskRunId) -> Option<Option<oneshot::Sender<LaneLease>>> {
        let index = self
            .waiting
            .iter()
            .position(|waiting| waiting.is_run(run))?;
        match self.waiting.remove(index) {
            Waiting::Run { resume, .. } => Some(resume),
            Waiting::Turn(_) | Waiting::Refine(_) | Waiting::Title(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use futures_util::FutureExt;

    use assistd_ipc::{PermissionMode, TaskSpec};

    use super::super::interaction::UserInteraction;
    use super::*;
    use crate::presence::SleepCause;
    use Class::{Refine, Run, Title, Turn};

    #[derive(Debug, Clone, Copy)]
    enum Class {
        Turn,
        Refine,
        Run,
        Title,
    }

    const TALKING: InteractionState = InteractionState::Talking {
        turns: NonZeroU32::MIN,
        ends_with_turn: false,
    };

    fn task() -> TaskRecord {
        TaskRecord {
            id: TaskId(1),
            spec: TaskSpec {
                name: "digest".to_string().try_into().unwrap(),
                prompt: "summarize my inbox".to_string().try_into().unwrap(),
                triggers: Vec::new(),
                enabled: true,
                mode: PermissionMode::Manual,
                allowed_hosts: Vec::new(),
                announce: false,
                kept_runs: None,
            },
            next_fire: None,
        }
    }

    fn waiting(class: Class, ticket: u64) -> Waiting {
        let waiter = Waiter {
            ticket,
            lease: oneshot::channel().0,
        };
        match class {
            Turn => Waiting::Turn(QueuedTurn {
                id: ticket.to_string(),
                preview: String::new(),
                waiter,
            }),
            Refine => Waiting::Refine(waiter),
            Title => Waiting::Title(waiter),
            Run => Waiting::Run {
                queued: QueuedRun {
                    run: TaskRunId(ticket.cast_signed()),
                    task: task(),
                    request_id: String::new(),
                    suspended: None,
                    paused_on: None,
                    started_at: None,
                },
                resume: None,
            },
        }
    }

    /// One row of the table: who is using the model, the lanes, the queue, and the
    /// entries that start, each by its index and lane.
    struct Case {
        name: &'static str,
        lanes: u32,
        interaction: InteractionState,
        presence: SettledPresence,
        paused: bool,
        main_held: bool,
        waiting: &'static [Class],
        starts: &'static [(usize, u32)],
    }

    impl Case {
        const IDLE: Self = Self {
            name: "",
            lanes: 1,
            interaction: InteractionState::Idle,
            presence: SettledPresence::Active,
            paused: false,
            main_held: false,
            waiting: &[],
            starts: &[],
        };

        fn started(&self) -> Vec<(usize, u32)> {
            let holders: BTreeMap<u32, LaneUse> = self
                .main_held
                .then_some((MAIN_LANE, LaneUse::UserTurn))
                .into_iter()
                .collect();
            let state = QueueState {
                lanes: holders.clone(),
                running: Vec::new(),
                waiting: (0..)
                    .zip(self.waiting)
                    .map(|(n, &c)| waiting(c, n))
                    .collect(),
                paused: self.paused,
            };
            let lanes = NonZeroU32::new(self.lanes).unwrap();
            let activity = Activity::new(self.interaction, holders, lanes, self.presence);
            state.startable(&activity)
        }
    }

    #[test]
    fn turns_refines_and_runs_start_as_the_lanes_and_who_is_using_the_model_allow() {
        let gpu_sleep = SettledPresence::Sleeping(SleepCause::GpuBusy);
        let cases = [
            Case {
                name: "the first entry takes the only lane",
                waiting: &[Run, Turn],
                starts: &[(0, 0)],
                ..Case::IDLE
            },
            Case {
                name: "nothing starts while the only lane is held",
                main_held: true,
                waiting: &[Turn, Refine, Run, Title],
                ..Case::IDLE
            },
            Case {
                name: "a run the interaction holds lets the turn behind it start",
                interaction: TALKING,
                waiting: &[Run, Turn],
                starts: &[(1, 0)],
                ..Case::IDLE
            },
            Case {
                name: "a paused queue holds its runs, not a refine",
                paused: true,
                waiting: &[Run, Refine],
                starts: &[(1, 0)],
                ..Case::IDLE
            },
            Case {
                name: "a sleep for the GPU holds runs, not a user turn",
                presence: gpu_sleep,
                waiting: &[Run, Turn],
                starts: &[(1, 0)],
                ..Case::IDLE
            },
            Case {
                name: "runs take the lanes beside the main one, one each",
                lanes: 3,
                waiting: &[Run, Run, Run, Turn],
                starts: &[(0, 1), (1, 2), (3, 0)],
                ..Case::IDLE
            },
            Case {
                name: "the interaction holds no run beside a task lane",
                lanes: 3,
                interaction: TALKING,
                waiting: &[Run, Turn],
                starts: &[(0, 1), (1, 0)],
                ..Case::IDLE
            },
            Case {
                name: "a run leaves a held main lane alone when there are others",
                lanes: 2,
                main_held: true,
                waiting: &[Refine, Run],
                starts: &[(1, 1)],
                ..Case::IDLE
            },
            Case {
                name: "a paused queue with several lanes",
                lanes: 3,
                paused: true,
                waiting: &[Run, Turn],
                starts: &[(1, 0)],
                ..Case::IDLE
            },
        ];
        for case in cases {
            assert_eq!(case.started(), case.starts, "{}", case.name);
        }
    }

    #[test]
    fn a_title_starts_only_when_nothing_else_takes_the_main_lane_and_the_daemon_is_active() {
        let gpu_sleep = SettledPresence::Sleeping(SleepCause::GpuBusy);
        let cases = [
            Case {
                name: "a title alone takes the main lane",
                waiting: &[Title, Title],
                starts: &[(0, 0)],
                ..Case::IDLE
            },
            Case {
                name: "a title gives way to a turn behind it",
                waiting: &[Title, Turn],
                starts: &[(1, 0)],
                ..Case::IDLE
            },
            Case {
                name: "a title gives way to a run on the only lane",
                waiting: &[Title, Run],
                starts: &[(1, 0)],
                ..Case::IDLE
            },
            Case {
                name: "a title starts past a held run",
                interaction: TALKING,
                waiting: &[Run, Title],
                starts: &[(1, 0)],
                ..Case::IDLE
            },
            Case {
                name: "a title starts beside runs on their own lanes",
                lanes: 2,
                waiting: &[Title, Run],
                starts: &[(0, 0), (1, 1)],
                ..Case::IDLE
            },
            Case {
                name: "a title waits while the daemon is drowsy",
                presence: SettledPresence::Drowsy,
                waiting: &[Title],
                ..Case::IDLE
            },
            Case {
                name: "a title and a run both wait out a sleep for the GPU",
                lanes: 2,
                presence: gpu_sleep,
                waiting: &[Run, Title],
                ..Case::IDLE
            },
        ];
        for case in cases {
            assert_eq!(case.started(), case.starts, "{}", case.name);
        }
    }

    /// A queue for `slots` lanes while the daemon is `Active`, with the interaction it
    /// reads and the channels that keep it so.
    struct Lanes {
        queue: Arc<TaskQueue>,
        interaction: Arc<UserInteraction>,
        _slots: watch::Sender<NonZeroU32>,
        _presence: watch::Sender<SettledPresence>,
    }

    impl Lanes {
        fn new(slots: u32) -> Self {
            let interaction = Arc::<UserInteraction>::default();
            let (slots, slot_count) = watch::channel(NonZeroU32::new(slots).unwrap());
            let (presence, settled) = watch::channel(SettledPresence::Active);
            Self {
                queue: TaskQueue::new(interaction.subscribe(), slot_count, settled),
                interaction,
                _slots: slots,
                _presence: presence,
            }
        }

        /// Queue run `run` and return its start if it was granted a lane at once.
        fn push(&self, run: i64) -> Option<RunStart> {
            self.queue.push(TaskRunId(run), task(), String::new());
            self.granted_run()
        }

        fn granted_run(&self) -> Option<RunStart> {
            self.queue.granted_run().now_or_never()
        }

        fn turn(&self, id: &str, interruption: TaskInterruption) -> WaitingTurn {
            self.queue
                .queue_turn(id.to_string(), String::new(), interruption)
        }

        /// The waiting entries the wire lists: a run by its id, a user turn by its
        /// query's.
        fn listed(&self) -> Vec<String> {
            let activity = self.queue.watch_activity().current();
            let snapshot = self.queue.snapshot(&activity);
            let waiting = snapshot.items.into_iter().filter_map(|item| match item {
                QueueItem::Run {
                    state: TaskRunState::Running,
                    ..
                } => None,
                QueueItem::Run { run, .. } => Some(run.to_string()),
                QueueItem::UserTurn { id, .. } => Some(id),
            });
            waiting.collect()
        }
    }

    fn granted(waiter: LaneWaiter) -> Option<LaneLease> {
        waiter.granted().now_or_never().flatten()
    }

    #[tokio::test]
    async fn a_user_turn_goes_before_the_waiting_runs_unless_it_waits_for_the_queue() {
        for interruption in [TaskInterruption::Interrupt, TaskInterruption::AfterTask] {
            let lanes = Lanes::new(1);
            let running = lanes.push(1).expect("the lane was free");
            assert!(lanes.push(2).is_none());

            let first = lanes.turn("first", interruption);
            let second = lanes.turn("second", interruption);
            assert_eq!((first.place, first.behind), (TurnPlace::Front, 1));
            assert_eq!((second.place, second.behind), (TurnPlace::Front, 2));
            assert_eq!(lanes.listed(), ["first", "second", "2"], "{interruption:?}");

            drop(running);
            let lane = granted(first.waiter).expect("the first turn took the freed lane");
            assert!(lanes.granted_run().is_none());
            drop(lane);
            assert!(granted(second.waiter).is_some());
        }
    }

    #[tokio::test]
    async fn an_after_queue_turn_waits_behind_the_runs_queued_before_it() {
        let after_queue = TaskInterruption::AfterQueue;
        let lanes = Lanes::new(1);
        let running = lanes.push(1).expect("the lane was free");
        assert!(lanes.push(2).is_none());

        let first = lanes.turn("first", after_queue);
        let second = lanes.turn("second", after_queue);
        assert!(lanes.push(3).is_none());
        assert_eq!((first.place, first.behind), (TurnPlace::BehindQueue, 2));
        assert_eq!((second.place, second.behind), (TurnPlace::BehindQueue, 3));
        assert_eq!(lanes.listed(), ["2", "first", "second", "3"]);

        drop(running);
        let queued_before = lanes.granted_run().expect("the run ahead took the lane");
        let _interaction = lanes.interaction.begin_turn(Duration::ZERO);
        let joining = lanes.turn("joining", after_queue);
        assert_eq!((joining.place, joining.behind), (TurnPlace::Front, 3));
        assert_eq!(lanes.listed(), ["first", "second", "joining", "3"]);

        drop(queued_before);
        assert!(granted(first.waiter).is_some());
    }

    #[tokio::test]
    async fn an_entry_whose_waiter_is_dropped_leaves_the_queue_and_never_takes_a_lane() {
        let lanes = Lanes::new(1);
        let holder = granted(lanes.queue.queue_refine()).expect("the lane was free");
        let turn = lanes.turn("left", TaskInterruption::Interrupt);
        let refine = lanes.queue.queue_refine();
        let title = lanes.queue.queue_title();
        assert_eq!(lanes.listed(), ["left"]);

        drop((turn, refine, title));
        assert!(lanes.listed().is_empty());
        assert!(lanes.queue.state.borrow().waiting.is_empty());
        drop(holder);
        assert!(lanes.queue.state.borrow().lanes.is_empty());
    }

    #[tokio::test]
    async fn a_run_cancelled_between_its_grant_and_its_start_gives_the_lane_back() {
        let lanes = Lanes::new(1);
        let start = lanes.push(1).expect("the lane was free");

        let cancelled = lanes.queue.cancel(TaskRunId(1), || false);
        assert!(matches!(cancelled, Cancelled::Waiting), "{cancelled:?}");
        lanes.queue.start_run(start, |queued, _lease| {
            panic!("cancelled run {} was started", queued.run)
        });
        assert!(lanes.queue.state.borrow().lanes.is_empty());
    }
}
