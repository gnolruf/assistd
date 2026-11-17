//! Follows the daemon's broadcast bus, resubscribing whenever the daemon goes away.

use std::sync::Arc;
use std::time::Duration;

use assistd_ipc::{EventKind, EventStream, IpcClient, Request, RequestBody, SubscribeFilter};
use tokio::sync::{Notify, mpsc, watch};
use uuid::Uuid;

use super::app::ChatEvent;

const BUS_RECONNECT_DELAY: Duration = Duration::from_secs(2);

/// What the bus subscription task reports to.
pub(super) struct BusFollower {
    pub(super) ipc: Arc<IpcClient>,
    pub(super) chat_tx: mpsc::Sender<ChatEvent>,
    /// Notified on every successful subscribe.
    pub(super) attached: Arc<Notify>,
}

/// Follow the daemon's broadcast bus for the daemon's state, session titles and turns this
/// chat did not start (push-to-talk, continuous listening, other clients).
/// Reconnects on a fixed delay, reporting the loss and the reconnect to the app;
/// `start_lost` reports the first subscribe as a reconnect too. Returns when `shutdown` flips.
pub(super) async fn follow_bus(
    follower: BusFollower,
    mut shutdown: watch::Receiver<bool>,
    start_lost: bool,
) {
    let mut lost = start_lost;
    loop {
        tokio::select! {
            _ = shutdown.changed() => break,
            () = pump_bus(&follower, &mut lost) => {
                tokio::select! {
                    _ = shutdown.changed() => break,
                    () = tokio::time::sleep(BUS_RECONNECT_DELAY) => {}
                }
            }
        }
    }
}

/// Forward bus events until the subscription ends. `lost` says a previous
/// subscription ended, so a successful subscribe sends `BusRestored`.
async fn pump_bus(follower: &BusFollower, lost: &mut bool) {
    let BusFollower {
        ipc,
        chat_tx,
        attached,
    } = follower;
    let Some(mut stream) = subscribe_bus(ipc).await else {
        return;
    };
    attached.notify_one();
    if std::mem::take(lost) && chat_tx.send(ChatEvent::BusRestored).await.is_err() {
        return;
    }
    while let Ok(Some(ev)) = stream.next_event().await {
        if chat_tx.send(ChatEvent::Bus(ev)).await.is_err() {
            return;
        }
    }
    *lost = true;
    let _ = chat_tx.send(ChatEvent::BusLost).await;
}

async fn subscribe_bus(ipc: &IpcClient) -> Option<EventStream> {
    let req = Request::new(
        Uuid::new_v4().to_string(),
        RequestBody::Subscribe {
            filter: SubscribeFilter {
                kinds: vec![
                    EventKind::SessionTitle,
                    EventKind::Transcription,
                    EventKind::Presence,
                    EventKind::VoiceState,
                    EventKind::ListenState,
                    EventKind::WakeWordDetected,
                    EventKind::Delta,
                    EventKind::ReasoningDelta,
                    EventKind::ToolCall,
                    EventKind::ToolResult,
                    EventKind::Relay,
                    EventKind::ContextSize,
                    EventKind::TaskQueue,
                    EventKind::Readiness,
                    EventKind::Done,
                    EventKind::Error,
                ],
            },
        },
    );
    ipc.one_shot(req)
        .await
        .inspect_err(|e| tracing::debug!("bus subscribe failed: {e}"))
        .ok()
}
