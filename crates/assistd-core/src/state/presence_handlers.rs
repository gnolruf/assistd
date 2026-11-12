//! Handlers for the presence state-machine variants of `Request`.

use std::sync::Arc;

use tokio::sync::mpsc;

use assistd_ipc::{Event, EventBody, PresenceTarget};

use super::{AppState, DispatchError, send_error};

impl AppState {
    pub(super) async fn handle_set_presence(
        self: Arc<Self>,
        id: String,
        target: PresenceTarget,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        match self.subsystems.presence.set_presence(target).await {
            Ok(()) => {
                let _ = tx
                    .send(Event::new(
                        id.clone(),
                        EventBody::Presence {
                            state: self.subsystems.presence.state(),
                        },
                    ))
                    .await;
                let _ = tx.send(Event::new(id, EventBody::Done)).await;
                Ok(())
            }
            Err(e) => {
                send_error(&tx, id, format!("set_presence failed: {e}")).await;
                Err(e.into())
            }
        }
    }

    pub(super) async fn handle_get_presence(self: Arc<Self>, id: String, tx: mpsc::Sender<Event>) {
        let state = self.subsystems.presence.state();
        let _ = tx
            .send(Event::new(id.clone(), EventBody::Presence { state }))
            .await;
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
    }
}
