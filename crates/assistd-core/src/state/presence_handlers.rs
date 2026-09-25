//! Handlers for the presence state-machine variants of `Request`.

use std::sync::Arc;

use tokio::sync::mpsc;

use assistd_ipc::{Event, PresenceState};

use super::{AppState, DispatchError, send_error};

impl AppState {
    pub(super) async fn handle_set_presence(
        self: Arc<Self>,
        id: String,
        target: PresenceState,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        match self.subsystems.presence.set_presence(target).await {
            Ok(()) => {
                let _ = tx
                    .send(Event::Presence {
                        id: id.clone(),
                        state: self.subsystems.presence.state(),
                    })
                    .await;
                let _ = tx.send(Event::Done { id }).await;
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
            .send(Event::Presence {
                id: id.clone(),
                state,
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
    }

    pub(super) async fn handle_cycle(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        match self.subsystems.presence.cycle().await {
            Ok(new_state) => {
                let _ = tx
                    .send(Event::Presence {
                        id: id.clone(),
                        state: new_state,
                    })
                    .await;
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                send_error(&tx, id, format!("cycle failed: {e}")).await;
                Err(e.into())
            }
        }
    }
}
