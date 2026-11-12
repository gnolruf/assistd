//! `Subscribe` handler: forwards bus events to a passive client.

use std::sync::Arc;

use tokio::sync::broadcast::error::RecvError;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::SendError;
use tracing::{debug, warn};

use assistd_ipc::{Event, EventBody, EventKind, SubscribeFilter, VoiceCaptureState};

use super::AppState;

/// The subscriber's client went away, so nothing more can reach it.
#[derive(Debug)]
struct SubscriberGone;

impl From<SendError<Event>> for SubscriberGone {
    fn from(_: SendError<Event>) -> Self {
        Self
    }
}

impl AppState {
    pub(super) async fn handle_subscribe(
        self: Arc<Self>,
        id: String,
        filter: SubscribeFilter,
        tx: mpsc::Sender<Event>,
    ) {
        debug!(
            target: "assistd::subscribe",
            id = %id,
            kinds = ?filter.kinds,
            "subscriber attached"
        );
        let wants_presence = filter.matches(EventKind::Presence);
        let mut presence = self.subsystems.presence.subscribe();
        let mut rx = self.runtime.subscribe_events(filter.clone());
        if self.send_current_state(&id, &filter, &tx).await.is_err() {
            return;
        }
        loop {
            tokio::select! {
                () = tx.closed() => {
                    debug!(
                        target: "assistd::subscribe",
                        id = %id,
                        "subscriber detached (client closed)"
                    );
                    return;
                }
                Ok(()) = presence.changed(), if wants_presence => {
                    let state = *presence.borrow_and_update();
                    if tx.send(Event::new(id.clone(), EventBody::Presence { state })).await.is_err() {
                        return;
                    }
                }
                recv = rx.recv() => match recv {
                    Ok(event) => {
                        if tx.send(event).await.is_err() {
                            return;
                        }
                    }
                    Err(RecvError::Lagged(skipped)) => {
                        warn!(
                            target: "assistd::subscribe",
                            id = %id,
                            skipped,
                            "subscriber lagged; resyncing current state and paused prompts"
                        );
                        if self.resync_lagged(&id, &filter, &tx).await.is_err() {
                            return;
                        }
                    }
                    Err(RecvError::Closed) => return,
                },
            }
        }
    }

    /// Resend what the skipped events may have carried that a subscriber cannot recover
    /// from later ones: the current state, and the prompts of runs paused for input.
    async fn resync_lagged(
        &self,
        id: &str,
        filter: &SubscribeFilter,
        tx: &mpsc::Sender<Event>,
    ) -> Result<(), SubscriberGone> {
        self.send_current_state(id, filter, tx).await?;
        if filter.matches(EventKind::TaskRun) {
            for prompt in self.paused_prompt_events() {
                tx.send(prompt).await?;
            }
        }
        Ok(())
    }

    /// Send the task queue, presence, listen phase and capture state as they stand, each
    /// when `filter` wants its kind.
    async fn send_current_state(
        &self,
        id: &str,
        filter: &SubscribeFilter,
        tx: &mpsc::Sender<Event>,
    ) -> Result<(), SubscriberGone> {
        let id = || id.to_string();
        let voice = &self.subsystems.voice;
        let current = [
            Event::new(
                id(),
                EventBody::TaskQueue {
                    snapshot: self.task_queue_snapshot(),
                },
            ),
            Event::new(
                id(),
                EventBody::Presence {
                    state: self.subsystems.presence.state(),
                },
            ),
            Event::new(
                id(),
                EventBody::ListenState {
                    phase: voice.listen_phase(),
                },
            ),
            Event::new(
                id(),
                EventBody::VoiceState {
                    state: voice
                        .capture()
                        .map_or(VoiceCaptureState::Idle, |capture| capture.input.state()),
                },
            ),
        ];
        for event in current {
            if event.kind().is_some_and(|kind| filter.matches(kind)) {
                tx.send(event).await?;
            }
        }
        Ok(())
    }
}
