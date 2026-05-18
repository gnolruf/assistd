//! Long-lived passive subscription to the daemon's broadcast bus.
//!
//! Connects, seeds the current presence and listening state, then
//! pumps events into the [`TrayItem`] through `Handle::update` for as
//! long as the daemon is reachable. On any disconnect — clean EOF or
//! I/O error — flips the tray to `Disconnected` and retries with
//! exponential backoff (1, 2, 4, … s, capped at 60 s).

use std::time::Duration;

use assistd_ipc::{
    Event, EventKind, IpcClient, IpcClientError, Request, SubscribeFilter,
    client::EventStream as IpcEventStream,
};
use ksni::Handle;
use uuid::Uuid;

use super::menu::TrayItem;

/// Run the reconnect loop forever. Returns only if the ksni handle has
/// shut down (signalling app exit).
pub async fn run(handle: Handle<TrayItem>, ipc: IpcClient) {
    let mut attempt: u32 = 0;
    loop {
        match try_once(&handle, &ipc).await {
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
) -> Result<ExitReason, IpcClientError> {
    let req = Request::Subscribe {
        id: Uuid::new_v4().to_string(),
        filter: SubscribeFilter {
            kinds: vec![
                EventKind::Delta,
                EventKind::ToolCall,
                EventKind::Done,
                EventKind::Error,
                EventKind::Presence,
                EventKind::ListenState,
            ],
        },
    };
    let stream = ipc.one_shot(req).await?;

    if push(handle, |item| item.set_connected()).await.is_none() {
        return Ok(ExitReason::ServiceShutdown);
    }

    seed_initial_state(handle, ipc).await;

    pump_events(handle, stream).await
}

async fn pump_events(
    handle: &Handle<TrayItem>,
    mut stream: IpcEventStream,
) -> Result<ExitReason, IpcClientError> {
    loop {
        match stream.next_event().await? {
            Some(ev) => {
                if push(handle, |item| item.ingest(&ev)).await.is_none() {
                    return Ok(ExitReason::ServiceShutdown);
                }
            }
            None => return Ok(ExitReason::DaemonClosed),
        }
    }
}

/// Subscribe is passive — it doesn't replay the daemon's current
/// presence or listening state. Run a pair of one-shot probes against
/// fresh connections so the tray icon reflects reality immediately
/// after a reconnect.
async fn seed_initial_state(handle: &Handle<TrayItem>, ipc: &IpcClient) {
    let presence_req = Request::GetPresence {
        id: Uuid::new_v4().to_string(),
    };
    let listen_req = Request::GetListenState {
        id: Uuid::new_v4().to_string(),
    };
    let (presence_res, listen_res) =
        tokio::join!(ipc.one_shot(presence_req), ipc.one_shot(listen_req));

    if let Ok(stream) = presence_res {
        consume_until_terminal(handle, stream).await;
    }
    if let Ok(stream) = listen_res {
        consume_until_terminal(handle, stream).await;
    }
}

async fn consume_until_terminal(handle: &Handle<TrayItem>, mut stream: IpcEventStream) {
    loop {
        match stream.next_event().await {
            Ok(Some(ev)) => {
                let terminal = matches!(ev, Event::Done { .. } | Event::Error { .. });
                if push(handle, |item| item.ingest(&ev)).await.is_none() {
                    return;
                }
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
