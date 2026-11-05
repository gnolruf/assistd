//! How a user turn reaches the main model lane when a task run may hold it, as
//! `[tasks] interruption` says.

use std::time::Duration;

use tokio::sync::{OwnedMutexGuard, mpsc, oneshot};

use assistd_config::TaskInterruption;
use assistd_ipc::Event;

use super::AppState;
use super::conversation::ConversationTurn;
use super::task_queue::{TaskQueue, TurnStart};

/// Longest preview of a queued user turn's text, in characters.
const TURN_PREVIEW_CHARS: usize = 80;

/// What a user turn holds: the main lane, and its place in the user's conversation.
#[derive(Debug)]
pub(super) struct MainLaneTurn {
    pub(super) lane: OwnedMutexGuard<()>,
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

impl AppState {
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
        tx: &mpsc::Sender<Event>,
    ) -> Option<MainLaneTurn> {
        let lanes = &self.runtime.lanes;
        if !lanes.is_single() {
            return Some(self.join_conversation().await);
        }
        match self.config.tasks.interruption {
            TaskInterruption::Interrupt => Some(self.join_conversation().await),
            TaskInterruption::AfterTask => {
                let turn = self
                    .runtime
                    .conversation
                    .begin_turn(self.conversation_quiet_time());
                if lanes.main_holder().is_some() {
                    let _ = tx
                        .send(Event::Queued {
                            id: id.to_string(),
                            behind: 1,
                        })
                        .await;
                }
                let lane = self.runtime.agent_turn_lock.clone().lock_owned().await;
                Some(MainLaneTurn { lane, turn })
            }
            TaskInterruption::AfterQueue if self.turn_waits_for_queue() => {
                self.queue_behind_tasks(id, text, tx).await
            }
            TaskInterruption::AfterQueue => Some(self.join_conversation().await),
        }
    }

    /// Open or continue the conversation, which a task run on the main lane yields to in
    /// `interrupt` mode, then wait for the lane.
    async fn join_conversation(&self) -> MainLaneTurn {
        let turn = self
            .runtime
            .conversation
            .begin_turn(self.conversation_quiet_time());
        let lane = self.runtime.agent_turn_lock.clone().lock_owned().await;
        MainLaneTurn { lane, turn }
    }

    /// Whether an `after_queue` turn joins the queue: no conversation holds it, and a task
    /// run holds the lane or user turns already wait.
    fn turn_waits_for_queue(&self) -> bool {
        !self.runtime.conversation.is_open()
            && (self.runtime.lanes.main_holder().is_some()
                || self.runtime.task_queue.has_waiting_turns())
    }

    async fn queue_behind_tasks(
        &self,
        id: &str,
        text: &str,
        tx: &mpsc::Sender<Event>,
    ) -> Option<MainLaneTurn> {
        let (start, started) = oneshot::channel::<TurnStart>();
        let preview = text.chars().take(TURN_PREVIEW_CHARS).collect();
        let queue = &self.runtime.task_queue;
        let behind = queue.push_turn(id.to_string(), preview, start);
        let _queued = QueuedTurnGuard {
            queue,
            id: id.to_string(),
        };
        let _ = tx
            .send(Event::Queued {
                id: id.to_string(),
                behind,
            })
            .await;
        let TurnStart { lane, turn } = started.await.ok()?;
        Some(MainLaneTurn { lane, turn })
    }
}
