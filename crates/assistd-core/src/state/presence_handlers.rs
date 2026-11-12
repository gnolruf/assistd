//! Handlers for the presence state-machine variants of `Request`.

use assistd_ipc::{Event, EventBody, PresenceTarget};

use super::status_service::StatusService;
use super::{DispatchError, send_error};
use crate::bus::EventSink;

impl StatusService {
    pub(super) async fn handle_set_presence(
        &self,
        id: String,
        target: PresenceTarget,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        match self.presence.set_presence(target).await {
            Ok(()) => {
                let _ = tx
                    .send(Event::new(
                        id.clone(),
                        EventBody::Presence {
                            state: self.presence.state(),
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

    pub(super) async fn handle_get_presence(&self, id: String, tx: EventSink) {
        let state = self.presence.state();
        let _ = tx
            .send(Event::new(id.clone(), EventBody::Presence { state }))
            .await;
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
    }
}
