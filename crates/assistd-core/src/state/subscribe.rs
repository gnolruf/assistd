//! `Subscribe` handler: forwards bus events to a passive client.

use std::sync::Arc;

use tokio::sync::broadcast::error::RecvError;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::SendError;
use tracing::{debug, warn};

use assistd_ipc::{Event, EventKind, SubscribeFilter};

use super::AppState;

/// The subscriber's client went away, so nothing more can reach it.
#[derive(Debug)]
struct SubscriberGone;

impl From<SendError<Event>> for SubscriberGone {
    fn from(_: SendError<Event>) -> Self {
        Self
    }
}

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
        let wants_runs = filter.matches(EventKind::TaskRun);
        let mut rx = self.runtime.subscribe_events(filter);
        if wants_queue && self.send_task_queue(&id, &tx).await.is_err() {
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
                            "subscriber lagged; resyncing the task queue and paused prompts"
                        );
                        if self.resync_lagged(&id, wants_queue, wants_runs, &tx).await.is_err() {
                            return;
                        }
                    }
                    Err(RecvError::Closed) => return,
                },
            }
        }
    }

    /// Resend what the skipped events may have carried that a subscriber cannot recover
    /// from later ones: the task queue, and the prompts of runs paused for input.
    async fn resync_lagged(
        &self,
        id: &str,
        wants_queue: bool,
        wants_runs: bool,
        tx: &mpsc::Sender<Event>,
    ) -> Result<(), SubscriberGone> {
        if wants_queue {
            self.send_task_queue(id, tx).await?;
        }
        if wants_runs {
            for prompt in self.paused_prompt_events() {
                tx.send(prompt).await?;
            }
        }
        Ok(())
    }

    async fn send_task_queue(
        &self,
        id: &str,
        tx: &mpsc::Sender<Event>,
    ) -> Result<(), SubscriberGone> {
        let snapshot = self.task_queue_snapshot();
        tx.send(Event::TaskQueue {
            id: id.to_string(),
            snapshot,
        })
        .await?;
        Ok(())
    }
}
