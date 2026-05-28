//! Long-lived passive subscription to the daemon's broadcast bus.

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

#[cfg(feature = "tray-popup")]
pub type OptionalPopup = Option<PopupSink>;
#[cfg(not(feature = "tray-popup"))]
pub type OptionalPopup = Option<()>;

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

enum ExitReason {
    ServiceShutdown,
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
        assert!(f.kinds.contains(&EventKind::LastDelta));
        assert!(f.kinds.contains(&EventKind::ToolResult));
        assert!(f.kinds.contains(&EventKind::SpeakingState));
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
