//! Handlers for the presence state-machine variants of `Request`.

use assistd_ipc::{Event, EventBody, PresenceTarget};

use super::status_service::StatusService;
use super::{DispatchError, FailedStep};
use crate::bus::EventSink;

impl StatusService {
    pub(super) async fn handle_set_presence(
        &self,
        id: String,
        target: PresenceTarget,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        self.presence
            .set_presence(target)
            .await
            .context("set_presence failed")?;
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

    pub(super) async fn handle_get_presence(&self, id: String, tx: EventSink) {
        let state = self.presence.state();
        let _ = tx
            .send(Event::new(id.clone(), EventBody::Presence { state }))
            .await;
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
    }
}
