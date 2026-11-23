//! `TurnAdmission`: who may use the model next. Everything that wants a model lane
//! waits in one queue, a user turn placed in it as `[tasks] interruption` says.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex as StdMutex;
use tokio::sync::{Mutex, OwnedMutexGuard, watch};
use tokio_util::sync::CancellationToken;

use assistd_config::Config;
use assistd_ipc::{Event, EventBody, TaskQueueSnapshot};

use self::activity::{Activity, ActivityWatch, Work};
use self::interaction::{InteractionTurn, UserInteraction};
use self::lanes::LaneLease;
use self::queue::{TaskQueue, TurnPlace, WaitingTurn};
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
    pub(crate) lane: LaneLease,
    pub(crate) user_turn: OwnedMutexGuard<()>,
    pub(crate) turn: InteractionTurn,
}

/// Who may use the model next: the lanes and the queue for them, the lock user turns
/// and branch operations take, the user's interaction, and the token that interrupts
/// turns.
#[derive(Debug)]
pub(crate) struct TurnAdmission {
    config: Arc<Config>,
    /// Serialises user turns with the branch operations that edit the main backend's history.
    pub(crate) user_turn_lock: Arc<Mutex<()>>,
    /// The user's interaction, which holds the task queue off a lone lane.
    pub(crate) interaction: Arc<UserInteraction>,
    pub(crate) task_queue: Arc<TaskQueue>,
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
        let interaction = Arc::<UserInteraction>::default();
        let task_queue = TaskQueue::new(interaction.subscribe(), slots, presence);
        Self {
            config,
            activity: task_queue.watch_activity(),
            user_turn_lock: Arc::default(),
            interaction,
            task_queue,
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

    /// Wait in the queue for the main lane for query `id`'s turn, telling `tx` how many
    /// go before it when any do; `None` when the turn was dropped from the queue. The
    /// user-turn lock is taken last, so a branch operation never waits on a lane.
    pub(crate) async fn take_main_lane(
        &self,
        id: &str,
        text: &str,
        tx: &EventSink,
    ) -> Option<MainLaneTurn> {
        let preview = text.chars().take(TURN_PREVIEW_CHARS).collect();
        let interruption = self.config.tasks.interruption;
        let WaitingTurn {
            waiter,
            place,
            behind,
        } = self
            .task_queue
            .queue_turn(id.to_string(), preview, interruption);
        let opened_on_arrival = (place == TurnPlace::Front).then(|| self.begin_turn());
        if behind > 0 {
            let _ = tx
                .send(Event::new(id.to_string(), EventBody::Queued { behind }))
                .await;
        }
        let lane = waiter.granted().await?;
        let turn = opened_on_arrival.unwrap_or_else(|| self.begin_turn());
        let user_turn = self.user_turn_lock.clone().lock_owned().await;
        Some(MainLaneTurn {
            lane,
            user_turn,
            turn,
        })
    }

    /// Open or continue the user's interaction, which holds waiting task runs off a lone
    /// lane and, in `interrupt` mode, makes the run on it yield.
    fn begin_turn(&self) -> InteractionTurn {
        self.interaction.begin_turn(self.interaction_quiet_time())
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
