//! Handler for [`assistd_ipc::Request::Subscribe`] — passive fan-out
//! connections that forward broadcast-eligible events from the
//! daemon-wide bus to a long-lived client (tray icon, status bar,
//! external dashboard) until the client disconnects or the daemon
//! shuts down.

use super::AppState;
use anyhow::Result;
use assistd_ipc::{Event, SubscribeFilter};
use std::sync::Arc;
use tokio::sync::broadcast::error::RecvError;
use tokio::sync::mpsc;
use tracing::{debug, warn};

impl AppState {
    pub(super) async fn handle_subscribe(
        self: Arc<Self>,
        id: String,
        filter: SubscribeFilter,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        let mut rx = self.runtime.subscribe_events();
        debug!(
            target: "assistd::subscribe",
            id = %id,
            kinds = ?filter.kinds,
            "subscriber attached"
        );
        loop {
            tokio::select! {
                _ = tx.closed() => {
                    debug!(
                        target: "assistd::subscribe",
                        id = %id,
                        "subscriber detached (client closed)"
                    );
                    return Ok(());
                }
                recv = rx.recv() => match recv {
                    Ok(event) => {
                        if let Some(kind) = event.kind()
                            && filter.matches(kind)
                            && tx.send(event).await.is_err()
                        {
                            return Ok(());
                        }
                    }
                    Err(RecvError::Lagged(skipped)) => {
                        warn!(
                            target: "assistd::subscribe",
                            id = %id,
                            skipped,
                            "subscriber lagged; dropping events"
                        );
                    }
                    Err(RecvError::Closed) => return Ok(()),
                },
            }
        }
    }
}
