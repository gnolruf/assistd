//! `Subscribe` handler: forwards bus events to a passive client.

use super::AppState;
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
    ) {
        debug!(
            target: "assistd::subscribe",
            id = %id,
            kinds = ?filter.kinds,
            "subscriber attached"
        );
        let mut rx = self.runtime.subscribe_events(filter);
        loop {
            tokio::select! {
                _ = tx.closed() => {
                    debug!(
                        target: "assistd::subscribe",
                        id = %id,
                        "subscriber detached (client closed)"
                    );
                    return;
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
                            "subscriber lagged; dropping events"
                        );
                    }
                    Err(RecvError::Closed) => return,
                },
            }
        }
    }
}
