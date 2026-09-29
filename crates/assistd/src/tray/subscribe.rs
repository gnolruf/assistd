//! Long-lived passive subscription to the daemon's broadcast bus.

use assistd_ipc::{
    Event, EventKind, IpcClient, IpcClientError, Request, SubscribeFilter,
    client::EventStream as IpcEventStream,
};
use assistd_utils::backoff::backoff_delay;
use ksni::Handle;
use uuid::Uuid;

use super::menu::TrayItem;
#[cfg(feature = "tray-popup")]
use super::popup::PopupSink;

#[cfg(feature = "tray-popup")]
pub(super) type OptionalPopup = Option<PopupSink>;
#[cfg(not(feature = "tray-popup"))]
pub type OptionalPopup = Option<()>;

#[cfg(feature = "tray-popup")]
type PopupSinkRef = PopupSink;
#[cfg(not(feature = "tray-popup"))]
type PopupSinkRef = ();

enum ExitReason {
    ServiceShutdown,
    DaemonClosed,
}

pub(super) async fn run(handle: Handle<TrayItem>, ipc: IpcClient, popup: OptionalPopup) {
    let mut attempt: u32 = 0;
    loop {
        match try_once(&handle, &ipc, popup.as_ref()).await {
            Ok(ExitReason::ServiceShutdown) => return,
            Ok(ExitReason::DaemonClosed) => {
                tracing::info!(target: "tray", "daemon closed the subscribe connection");
                attempt = 0;
            }
            Err(e) => {
                tracing::warn!(target: "tray", "subscribe attempt failed: {e}");
            }
        }
        if push(&handle, TrayItem::set_disconnected).await.is_none() {
            return;
        }
        disconnect_popup(popup.as_ref());
        tokio::time::sleep(backoff_delay(attempt)).await;
        attempt = attempt.saturating_add(1);
    }
}

async fn try_once(
    handle: &Handle<TrayItem>,
    ipc: &IpcClient,
    popup: Option<&PopupSinkRef>,
) -> Result<ExitReason, IpcClientError> {
    let req = Request::Subscribe {
        id: Uuid::new_v4().to_string(),
        filter: subscribe_filter(),
    };
    let stream = ipc.one_shot(req).await?;

    if push(handle, TrayItem::set_connected).await.is_none() {
        return Ok(ExitReason::ServiceShutdown);
    }

    seed_initial_state(handle, ipc, popup).await;

    pump_events(handle, stream, popup).await
}

fn subscribe_filter() -> SubscribeFilter {
    SubscribeFilter {
        kinds: vec![
            EventKind::Delta,
            EventKind::LastDelta,
            EventKind::ReasoningDelta,
            EventKind::ToolCall,
            EventKind::ToolResult,
            EventKind::Done,
            EventKind::Error,
            EventKind::Presence,
            EventKind::ListenState,
            EventKind::SpeakingState,
        ],
    }
}

async fn pump_events(
    handle: &Handle<TrayItem>,
    mut stream: IpcEventStream,
    popup: Option<&PopupSinkRef>,
) -> Result<ExitReason, IpcClientError> {
    loop {
        match stream.next_event().await? {
            Some(ev) => {
                if push(handle, |item| item.ingest(&ev)).await.is_none() {
                    return Ok(ExitReason::ServiceShutdown);
                }
                forward_to_popup(popup, &ev);
            }
            None => return Ok(ExitReason::DaemonClosed),
        }
    }
}

async fn seed_initial_state(
    handle: &Handle<TrayItem>,
    ipc: &IpcClient,
    popup: Option<&PopupSinkRef>,
) {
    let presence_req = Request::GetPresence {
        id: Uuid::new_v4().to_string(),
    };
    let listen_req = Request::GetListenState {
        id: Uuid::new_v4().to_string(),
    };
    let (presence_res, listen_res) =
        tokio::join!(ipc.one_shot(presence_req), ipc.one_shot(listen_req));

    if let Ok(stream) = presence_res {
        consume_until_terminal(handle, stream, popup).await;
    }
    if let Ok(stream) = listen_res {
        consume_until_terminal(handle, stream, popup).await;
    }
}

async fn consume_until_terminal(
    handle: &Handle<TrayItem>,
    mut stream: IpcEventStream,
    popup: Option<&PopupSinkRef>,
) {
    loop {
        match stream.next_event().await {
            Ok(Some(ev)) => {
                let terminal = matches!(ev, Event::Done { .. } | Event::Error { .. });
                if push(handle, |item| item.ingest(&ev)).await.is_none() {
                    return;
                }
                forward_to_popup(popup, &ev);
                if terminal {
                    return;
                }
            }
            Ok(None) | Err(_) => return,
        }
    }
}

async fn push<F>(handle: &Handle<TrayItem>, update: F) -> Option<bool>
where
    F: FnOnce(&mut TrayItem) -> bool + Send,
{
    handle.update(update).await
}

#[cfg(feature = "tray-popup")]
fn forward_to_popup(popup: Option<&PopupSink>, ev: &Event) {
    if let Some(p) = popup {
        p.ingest(ev);
    }
}

#[cfg(not(feature = "tray-popup"))]
fn forward_to_popup(_popup: Option<&()>, _ev: &Event) {}

#[cfg(feature = "tray-popup")]
fn disconnect_popup(popup: Option<&PopupSink>) {
    if let Some(p) = popup {
        p.set_disconnected();
    }
}

#[cfg(not(feature = "tray-popup"))]
fn disconnect_popup(_popup: Option<&()>) {}
