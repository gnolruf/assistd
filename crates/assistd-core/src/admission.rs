//! `TurnAdmission`: who may use the model next. How a user turn reaches the main model
//! lane when a task run may hold it, as `[tasks] interruption` says.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex as StdMutex;
use tokio::sync::{Mutex, OwnedMutexGuard, oneshot, watch};
use tokio_util::sync::CancellationToken;

use assistd_config::{Config, TaskInterruption};
use assistd_ipc::{Event, EventBody, TaskQueueSnapshot};

use self::activity::{Activity, ActivityWatch, Work};
use self::interaction::{InteractionTurn, UserInteraction};
use self::lanes::ModelLanes;
use self::queue::{TaskQueue, TurnStart};
use crate::bus::EventSink;
use crate::presence::SettledPresence;
use crate::recovery::spawn_supervised;

pub(crate) mod activity;
pub(crate) mod interaction;
pub(crate) mod lanes;
pub(crate) mod queue;

/// Longest preview of a queued user turn's text, in characters.
const TURN_PREVIEW_CHARS: usize = 80;

/// What a user turn holds: the main lane, the main backend against branch
/// operations, and its place in the user's interaction.
#[derive(Debug)]
pub(crate) struct MainLaneTurn {
    pub(crate) lane: OwnedMutexGuard<()>,
    pub(crate) user_turn: OwnedMutexGuard<()>,
    pub(crate) turn: InteractionTurn,
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
/// take, the user's interaction, the queue of waiting runs and turns, and the token
/// that interrupts turns.
#[derive(Debug)]
pub(crate) struct TurnAdmission {
    config: Arc<Config>,
    /// The main model lane, held by user turns and by a task run on it.
    pub(crate) agent_turn_lock: Arc<Mutex<()>>,
    /// Serialises user turns with the branch operations that edit the main backend's history.
    pub(crate) user_turn_lock: Arc<Mutex<()>>,
    pub(crate) lanes: Arc<ModelLanes>,
    /// The user's interaction, which holds the task queue off a lone lane.
    pub(crate) interaction: Arc<UserInteraction>,
    pub(crate) task_queue: TaskQueue,
    activity: ActivityWatch,
    /// Parent of every turn's cancellation token; an interrupt cancels it
    /// and installs a fresh one, a stop cancels it for good.
    interrupt: StdMutex<CancellationToken>,
}

impl TurnAdmission {
    /// Admission over as many lanes as `slots`, the server's slot count, reports,
    /// beside presence's settled state `presence`.
    pub(crate) fn new(
        config: Arc<Config>,
        slots: watch::Receiver<NonZeroU32>,
        presence: watch::Receiver<SettledPresence>,
    ) -> Self {
        let agent_turn_lock = Arc::new(Mutex::new(()));
        let lanes = ModelLanes::new(Arc::clone(&agent_turn_lock), slots);
        let interaction = Arc::<UserInteraction>::default();
        Self {
            config,
            activity: ActivityWatch::new(
                interaction.subscribe(),
                lanes.subscribe_runs(),
                lanes.subscribe_count(),
                presence,
            ),
            lanes,
            agent_turn_lock,
            user_turn_lock: Arc::default(),
            interaction,
            task_queue: TaskQueue::default(),
            interrupt: StdMutex::new(CancellationToken::new()),
        }
    }

    /// A token for a turn starting now, cancelled by the next
    /// [`Self::interrupt_turns`]; already cancelled after [`Self::stop_turns`].
    pub(crate) fn turn_cancellation(&self) -> CancellationToken {
        self.interrupt.lock().child_token()
    }

    /// Cancel every turn started so far, queued or running; later turns
    /// are unaffected.
    pub(crate) fn interrupt_turns(&self) {
        let mut interrupt = self.interrupt.lock();
        if !interrupt.is_cancelled() {
            std::mem::replace(&mut *interrupt, CancellationToken::new()).cancel();
        }
    }

    /// Cancel every turn, queued or running, and every turn started afterwards.
    pub(crate) fn stop_turns(&self) {
        self.interrupt.lock().cancel();
    }

    /// Who is using the model now.
    pub(crate) fn activity(&self) -> Activity {
        self.watch_activity().current()
    }

    /// Watch who is using the model.
    pub(crate) fn watch_activity(&self) -> ActivityWatch {
        self.activity.clone()
    }

    /// Watch whether `work` may run now. A supervised task keeps the answer current
    /// until every receiver is dropped.
    pub(crate) fn permit(&self, work: Work) -> watch::Receiver<bool> {
        let mut activity = self.watch_activity();
        let permitted_now = activity.current().permits(work, &self.config);
        let (permit, permitted) = watch::channel(permitted_now);
        spawn_supervised(
            "work_permit",
            work.component(),
            keep_permit(Arc::clone(&self.config), activity, work, permit),
        );
        permitted
    }

    /// The task queue and the model lanes as they stand.
    pub(crate) fn queue_snapshot(&self) -> TaskQueueSnapshot {
        self.task_queue.snapshot(&self.activity())
    }

    /// How long the user's interaction stays open after its last reply finishes:
    /// `resume_after_idle_secs`, or the wake word's follow-up window when longer.
    pub(crate) fn interaction_quiet_time(&self) -> Duration {
        let idle = Duration::from_secs(self.config.tasks.resume_after_idle_secs.get().into());
        let voice = &self.config.voice;
        if voice.enabled && voice.wakeword.enabled {
            idle.max(Duration::from_millis(
                voice.wakeword.follow_up_ms.get().into(),
            ))
        } else {
            idle
        }
    }

    /// Wait for the main lane for query `id`'s turn, getting past a task run that holds it
    /// as `[tasks] interruption` says; `None` when the turn was dropped from the queue.
    pub(crate) async fn take_main_lane(
        &self,
        id: &str,
        text: &str,
        tx: &EventSink,
    ) -> Option<MainLaneTurn> {
        let activity = self.activity();
        match self.config.tasks.interruption {
            _ if !activity.single_lane() => Some(self.join_interaction().await),
            TaskInterruption::AfterTask => {
                let turn = self.interaction.begin_turn(self.interaction_quiet_time());
                if activity.main_held_by_run() {
                    let _ = tx
                        .send(Event::new(id.to_string(), EventBody::Queued { behind: 1 }))
                        .await;
                }
                let lane = self.agent_turn_lock.clone().lock_owned().await;
                Some(self.main_lane_turn(lane, turn).await)
            }
            TaskInterruption::AfterQueue if self.turn_waits_for_queue(&activity) => {
                self.queue_behind_tasks(id, text, tx).await
            }
            TaskInterruption::Interrupt | TaskInterruption::AfterQueue => {
                Some(self.join_interaction().await)
            }
        }
    }

    /// Open or continue the user's interaction, which a task run on the main lane yields to in
    /// `interrupt` mode, then wait for the lane.
    async fn join_interaction(&self) -> MainLaneTurn {
        let turn = self.interaction.begin_turn(self.interaction_quiet_time());
        let lane = self.agent_turn_lock.clone().lock_owned().await;
        self.main_lane_turn(lane, turn).await
    }

    /// Take the user-turn lock for `turn`, which already holds `lane`. Always lane first:
    /// branch operations take only the user-turn lock, so they never wait on the lane.
    async fn main_lane_turn(
        &self,
        lane: OwnedMutexGuard<()>,
        turn: InteractionTurn,
    ) -> MainLaneTurn {
        let user_turn = self.user_turn_lock.clone().lock_owned().await;
        MainLaneTurn {
            lane,
            user_turn,
            turn,
        }
    }

    /// Whether an `after_queue` turn joins the queue: no interaction holds it, and a task
    /// run holds the lane or user turns already wait.
    fn turn_waits_for_queue(&self, activity: &Activity) -> bool {
        !activity.interaction_open()
            && (activity.main_held_by_run() || self.task_queue.has_waiting_turns())
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

/// Keep `permit` at whether `activity` permits `work`, until its last receiver is dropped.
async fn keep_permit(
    config: Arc<Config>,
    mut activity: ActivityWatch,
    work: Work,
    permit: watch::Sender<bool>,
) {
    loop {
        tokio::select! {
            () = permit.closed() => return,
            () = activity.changed() => {}
        }
        let permitted = activity.current().permits(work, &config);
        permit.send_if_modified(|current| {
            let changed = *current != permitted;
            *current = permitted;
            changed
        });
    }
}
