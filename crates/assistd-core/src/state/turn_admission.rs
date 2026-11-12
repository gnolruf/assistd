//! How a user turn reaches the main model lane when a task run may hold it, as
//! `[tasks] interruption` says.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex as StdMutex;
use tokio::sync::{Mutex, OwnedMutexGuard, oneshot, watch};
use tokio_util::sync::CancellationToken;

use assistd_config::{Config, TaskInterruption};
use assistd_ipc::{ConversationHold, Event, EventBody, TaskQueueSnapshot};

use super::conversation::{ConversationState, ConversationTurn, UserConversation};
use super::lanes::ModelLanes;
use super::task_queue::{TaskQueue, TurnStart};
use crate::bus::EventSink;

/// Longest preview of a queued user turn's text, in characters.
const TURN_PREVIEW_CHARS: usize = 80;

/// What a user turn holds: the main lane, the main conversation against branch
/// operations, and its place in the user's conversation.
#[derive(Debug)]
pub(super) struct MainLaneTurn {
    pub(super) lane: OwnedMutexGuard<()>,
    pub(super) user_turn: OwnedMutexGuard<()>,
    pub(super) turn: ConversationTurn,
}

/// A user turn waiting in the task queue, taken out again if dropped before it starts.
struct QueuedTurnGuard<'queue> {
    queue: &'queue TaskQueue,
    id: String,
}

impl Drop for QueuedTurnGuard<'_> {
    fn drop(&mut self) {
        let _ = self.queue.cancel_turn(&self.id);
    }
}

/// Who may use the model next: the lanes, the locks user turns and branch operations
/// take, the user's conversation, the queue of waiting runs and turns, and the token
/// that interrupts turns.
#[derive(Debug)]
pub(super) struct TurnAdmission {
    config: Arc<Config>,
    /// The main model lane, held by user turns and by a task run on it.
    pub(super) agent_turn_lock: Arc<Mutex<()>>,
    /// Serialises user turns with the branch operations that edit the main conversation.
    pub(super) user_turn_lock: Arc<Mutex<()>>,
    pub(super) lanes: Arc<ModelLanes>,
    /// The user's conversation, which holds the task queue off a lone lane.
    pub(super) conversation: Arc<UserConversation>,
    pub(super) task_queue: TaskQueue,
    /// Parent of every turn's cancellation token; an interrupt cancels it
    /// and installs a fresh one.
    interrupt: StdMutex<CancellationToken>,
}

impl TurnAdmission {
    /// Admission over as many lanes as `slots`, the server's slot count, reports.
    pub(super) fn new(config: Arc<Config>, slots: watch::Receiver<NonZeroU32>) -> Self {
        let agent_turn_lock = Arc::new(Mutex::new(()));
        Self {
            config,
            lanes: ModelLanes::new(Arc::clone(&agent_turn_lock), slots),
            agent_turn_lock,
            user_turn_lock: Arc::default(),
            conversation: Arc::default(),
            task_queue: TaskQueue::default(),
            interrupt: StdMutex::new(CancellationToken::new()),
        }
    }

    /// A token for a turn starting now, cancelled by the next
    /// [`Self::interrupt_turns`].
    pub(super) fn turn_cancellation(&self) -> CancellationToken {
        self.interrupt.lock().child_token()
    }

    /// Cancel every turn started so far, queued or running; later turns
    /// are unaffected.
    pub(super) fn interrupt_turns(&self) {
        let interrupted = std::mem::replace(&mut *self.interrupt.lock(), CancellationToken::new());
        interrupted.cancel();
    }

    /// The task queue and the model lanes as they stand.
    pub(super) fn queue_snapshot(&self) -> TaskQueueSnapshot {
        self.task_queue
            .snapshot(self.lanes.holders(), self.conversation_hold())
    }

    /// Whether the user's conversation keeps task runs off the model's only lane.
    pub(super) fn conversation_holds_queue(&self) -> bool {
        self.lanes.is_single() && self.conversation.is_open()
    }

    /// Whether a run on the main lane must give it to the user's conversation now.
    pub(super) fn run_must_yield(&self) -> bool {
        self.config.tasks.interruption == TaskInterruption::Interrupt
            && self.conversation_holds_queue()
    }

    fn conversation_hold(&self) -> Option<ConversationHold> {
        if !self.lanes.is_single() {
            return None;
        }
        match self.conversation.state() {
            ConversationState::Idle => None,
            ConversationState::Talking { .. } => Some(ConversationHold { resumes_at: None }),
            ConversationState::Quiet { resumes_at, .. } => Some(ConversationHold {
                resumes_at: Some(resumes_at),
            }),
        }
    }

    /// How long the user's conversation stays open after its last reply finishes:
    /// `resume_after_idle_secs`, or the wake word's follow-up window when longer.
    pub(super) fn conversation_quiet_time(&self) -> Duration {
        let idle = Duration::from_secs(self.config.tasks.resume_after_idle_secs.into());
        let voice = &self.config.voice;
        if voice.enabled && voice.wakeword.enabled {
            idle.max(Duration::from_millis(voice.wakeword.follow_up_ms.into()))
        } else {
            idle
        }
    }

    /// Wait for the main lane for query `id`'s turn, getting past a task run that holds it
    /// as `[tasks] interruption` says; `None` when the turn was dropped from the queue.
    pub(super) async fn take_main_lane(
        &self,
        id: &str,
        text: &str,
        tx: &EventSink,
    ) -> Option<MainLaneTurn> {
        let lanes = &self.lanes;
        match self.config.tasks.interruption {
            _ if !lanes.is_single() => Some(self.join_conversation().await),
            TaskInterruption::AfterTask => {
                let turn = self.conversation.begin_turn(self.conversation_quiet_time());
                if lanes.main_held_by_run() {
                    let _ = tx
                        .send(Event::new(id.to_string(), EventBody::Queued { behind: 1 }))
                        .await;
                }
                let lane = self.agent_turn_lock.clone().lock_owned().await;
                Some(self.main_lane_turn(lane, turn).await)
            }
            TaskInterruption::AfterQueue if self.turn_waits_for_queue() => {
                self.queue_behind_tasks(id, text, tx).await
            }
            TaskInterruption::Interrupt | TaskInterruption::AfterQueue => {
                Some(self.join_conversation().await)
            }
        }
    }

    /// Open or continue the conversation, which a task run on the main lane yields to in
    /// `interrupt` mode, then wait for the lane.
    async fn join_conversation(&self) -> MainLaneTurn {
        let turn = self.conversation.begin_turn(self.conversation_quiet_time());
        let lane = self.agent_turn_lock.clone().lock_owned().await;
        self.main_lane_turn(lane, turn).await
    }

    /// Take the user-turn lock for `turn`, which already holds `lane`. Always lane first:
    /// branch operations take only the user-turn lock, so they never wait on the lane.
    async fn main_lane_turn(
        &self,
        lane: OwnedMutexGuard<()>,
        turn: ConversationTurn,
    ) -> MainLaneTurn {
        let user_turn = self.user_turn_lock.clone().lock_owned().await;
        MainLaneTurn {
            lane,
            user_turn,
            turn,
        }
    }

    /// Whether an `after_queue` turn joins the queue: no conversation holds it, and a task
    /// run holds the lane or user turns already wait.
    fn turn_waits_for_queue(&self) -> bool {
        !self.conversation.is_open()
            && (self.lanes.main_held_by_run() || self.task_queue.has_waiting_turns())
    }

    async fn queue_behind_tasks(
        &self,
        id: &str,
        text: &str,
        tx: &EventSink,
    ) -> Option<MainLaneTurn> {
        let (start, started) = oneshot::channel::<TurnStart>();
        let preview = text.chars().take(TURN_PREVIEW_CHARS).collect();
        let queue = &self.task_queue;
        let behind = queue.push_turn(id.to_string(), preview, start);
        let _queued = QueuedTurnGuard {
            queue,
            id: id.to_string(),
        };
        let _ = tx
            .send(Event::new(id.to_string(), EventBody::Queued { behind }))
            .await;
        let TurnStart { lane, turn } = started.await.ok()?;
        Some(self.main_lane_turn(lane, turn).await)
    }
}
