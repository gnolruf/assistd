//! Handlers for the presence state-machine variants of `Request`.

use assistd_ipc::{EventBody, PresenceTarget};

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
        let state = self.presence.state();
        tx.reply(id, EventBody::Presence { state }).await;
        Ok(())
    }

    pub(super) async fn handle_get_presence(
        &self,
        id: String,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let state = self.presence.state();
        tx.reply(id, EventBody::Presence { state }).await;
        Ok(())
    }
}
