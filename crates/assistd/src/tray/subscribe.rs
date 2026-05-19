//! Long-lived passive subscription to the daemon's broadcast bus.
//!
//! Connects, seeds the current presence and listening state, then
//! pumps events into the [`TrayItem`] through `Handle::update` for as
//! long as the daemon is reachable. On any disconnect — clean EOF or
//! I/O error — flips the tray to `Disconnected` and retries with
//! exponential backoff (1, 2, 4, … s, capped at 60 s).
//!
//! When the popup feature is on, a [`PopupSink`](super::popup::PopupSink)
//! also forwarded every event (plus `Disconnected` transitions) so the
//! popup driver can update its content and apply its wake-on rules.

use std::time::Duration;

use assistd_ipc::{
    Event, EventKind, IpcClient, IpcClientError, Request, SubscribeFilter,
    client::EventStream as IpcEventStream,
};
use ksni::Handle;
use uuid::Uuid;

use super::menu::TrayItem;
#[cfg(feature = "tray-popup")]
use super::popup::PopupSink;

/// Optional popup hook handed to the subscribe loop. Cloned by the
/// caller into [`run`]; `None` in builds without the popup feature
/// (or when the popup is disabled in config).
#[cfg(feature = "tray-popup")]
pub type OptionalPopup = Option<PopupSink>;
#[cfg(not(feature = "tray-popup"))]
pub type OptionalPopup = Option<()>;

/// Run the reconnect loop forever. Returns only if the ksni handle has
/// shut down (signalling app exit).
pub async fn run(handle: Handle<TrayItem>, ipc: IpcClient, popup: OptionalPopup) {
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
        if push(&handle, |item| item.set_disconnected())
            .await
            .is_none()
        {
            return;
        }
        #[cfg(feature = "tray-popup")]
        if let Some(p) = popup.as_ref() {
            p.set_disconnected();
        }
        tokio::time::sleep(backoff_delay(attempt)).await;
        attempt = attempt.saturating_add(1);
    }
}

/// Why a single subscribe attempt returned.
enum ExitReason {
    /// ksni told us the service is gone — quit the loop.
    ServiceShutdown,
    /// The daemon closed cleanly; reconnect after the usual backoff.
    DaemonClosed,
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

    if push(handle, |item| item.set_connected()).await.is_none() {
        return Ok(ExitReason::ServiceShutdown);
    }

    seed_initial_state(handle, ipc, popup).await;

    pump_events(handle, stream, popup).await
}

/// Build the filter requested from the daemon. Extracted into a
/// helper so a unit test can verify both tray-only and popup-relevant
/// kinds are present.
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
                #[cfg(feature = "tray-popup")]
                if let Some(p) = popup {
                    p.ingest(&ev);
                }
                let _ = popup;
                let _ = &ev;
            }
            None => return Ok(ExitReason::DaemonClosed),
        }
    }
}

#[cfg(feature = "tray-popup")]
type PopupSinkRef = PopupSink;
#[cfg(not(feature = "tray-popup"))]
type PopupSinkRef = ();

/// Subscribe is passive — it doesn't replay the daemon's current
/// presence or listening state. Run a pair of one-shot probes against
/// fresh connections so the tray icon reflects reality immediately
/// after a reconnect.
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
                #[cfg(feature = "tray-popup")]
                if let Some(p) = popup {
                    p.ingest(&ev);
                }
                let _ = popup;
                if terminal {
                    return;
                }
            }
            Ok(None) | Err(_) => return,
        }
    }
}

async fn push<F>(handle: &Handle<TrayItem>, f: F) -> Option<bool>
where
    F: FnOnce(&mut TrayItem) -> bool + Send,
{
    handle.update(f).await
}

/// Exponential backoff capped at 60 s. Mirrors the shape used in
/// `assistd-llm`'s llama-server supervisor without taking a dependency
/// on it — the workspace's three-use rule says inline.
fn backoff_delay(attempt: u32) -> Duration {
    const CAP_SECS: u64 = 60;
    let secs = 1u64.checked_shl(attempt).unwrap_or(CAP_SECS).min(CAP_SECS);
    Duration::from_secs(secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subscribe_filter_lists_tray_and_popup_event_kinds() {
        let f = subscribe_filter();
        // Pre-existing tray-icon needs (presence priority machine).
        for k in [
            EventKind::Delta,
            EventKind::ToolCall,
            EventKind::Done,
            EventKind::Error,
            EventKind::Presence,
            EventKind::ListenState,
        ] {
            assert!(f.kinds.contains(&k), "filter missing {k:?}");
        }
        // Added for the popup: server-coalesced reply text + tool
        // results. Both are no-ops on TrayTracker but feed the popup.
        assert!(f.kinds.contains(&EventKind::LastDelta));
        assert!(f.kinds.contains(&EventKind::ToolResult));
    }

    #[test]
    fn backoff_matches_spec_sequence() {
        let expected = [1, 2, 4, 8, 16, 32, 60, 60, 60];
        for (i, want) in expected.iter().enumerate() {
            assert_eq!(
                backoff_delay(i as u32),
                Duration::from_secs(*want),
                "attempt {i}"
            );
        }
    }

    #[test]
    fn backoff_caps_for_large_attempts() {
        assert_eq!(backoff_delay(64), Duration::from_secs(60));
        assert_eq!(backoff_delay(u32::MAX), Duration::from_secs(60));
    }
}
