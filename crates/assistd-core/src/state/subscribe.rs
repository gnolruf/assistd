//! `Subscribe` handler: forwards bus events to a passive client.

use tokio::sync::broadcast::error::RecvError;
use tracing::{debug, warn};

use assistd_ipc::{Event, EventBody, EventKind, SubscribeFilter, VoiceCaptureState};

use super::status_service::StatusService;
use crate::bus::{ClientGone, EventSink};

impl StatusService {
    pub(super) async fn handle_subscribe(
        &self,
        id: String,
        filter: SubscribeFilter,
        tx: EventSink,
    ) {
        debug!(
            target: "assistd::subscribe",
            id = %id,
            kinds = ?filter.kinds,
            "subscriber attached"
        );
        let wants_presence = filter.matches(EventKind::Presence);
        let mut presence = self.presence.subscribe();
        let mut rx = self.bus.subscribe(filter.clone());
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
                            "subscriber lagged; resending current state"
                        );
                        if self.send_current_state(&id, &filter, &tx).await.is_err() {
                            return;
                        }
                    }
                    Err(RecvError::Closed) => return,
                },
            }
        }
    }

    /// Send the task queue, presence, listen phase, capture state and document activity
    /// as they stand, each when `filter` wants its kind.
    async fn send_current_state(
        &self,
        id: &str,
        filter: &SubscribeFilter,
        tx: &EventSink,
    ) -> Result<(), ClientGone> {
        let id = || id.to_string();
        let voice = &self.voice;
        let current = [
            Event::new(
                id(),
                EventBody::TaskQueue {
                    snapshot: self.admission.queue_snapshot(),
                },
            ),
            Event::new(
                id(),
                EventBody::Presence {
                    state: self.presence.state(),
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
            Event::new(
                id(),
                EventBody::DocumentActivity {
                    activity: self.document_activity.borrow().clone(),
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
