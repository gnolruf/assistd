//! Handlers for the presence state-machine variants of `Request`.

use super::AppState;
use anyhow::Result;
use assistd_ipc::{Event, PresenceState};
use std::sync::Arc;
use tokio::sync::mpsc;

impl AppState {
    pub(super) async fn handle_set_presence(
        self: Arc<Self>,
        id: String,
        target: PresenceState,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
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
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("set_presence failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    pub(super) async fn handle_get_presence(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        let state = self.subsystems.presence.state();
        let _ = tx
            .send(Event::Presence {
                id: id.clone(),
                state,
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    pub(super) async fn handle_cycle(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
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
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("cycle failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }
}
