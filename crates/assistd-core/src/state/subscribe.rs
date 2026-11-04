//! `Subscribe` handler: forwards bus events to a passive client.

use std::sync::Arc;

use tokio::sync::broadcast::error::RecvError;
use tokio::sync::mpsc;
use tracing::{debug, warn};

use assistd_ipc::{Event, EventKind, SubscribeFilter};

use super::AppState;

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
        let wants_queue = filter.matches(EventKind::TaskQueue);
        let mut rx = self.runtime.subscribe_events(filter);
        if wants_queue {
            let snapshot = self.task_queue_snapshot();
            let _ = tx
                .send(Event::TaskQueue {
                    id: id.clone(),
                    snapshot,
                })
                .await;
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
