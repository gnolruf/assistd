//! Reports the chat's keyboard focus to the daemon, from terminal focus
//! reports and keypresses.

use std::sync::Arc;
use std::time::Duration;

use assistd_ipc::{IpcClient, Request};
use crossterm::event::Event as TermEvent;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use uuid::Uuid;

/// How often the current focus is re-sent, so a restarted daemon learns it.
const RESEND_INTERVAL: Duration = Duration::from_secs(5);
const CLOSE_TIMEOUT: Duration = Duration::from_secs(1);

/// Update `focus` from a terminal event. A keypress counts as focus,
/// which covers terminals that never report it.
pub(super) fn track(focus: &watch::Sender<bool>, ev: &TermEvent) {
    let focused = match ev {
        TermEvent::FocusGained | TermEvent::Key(_) => true,
        TermEvent::FocusLost => false,
        _ => return,
    };
    focus.send_if_modified(|current| std::mem::replace(current, focused) != focused);
}

/// Send the focus to the daemon now, on every change, and every
/// [`RESEND_INTERVAL`] until `shutdown` flips.
pub(super) fn spawn_reporter(
    ipc: Arc<IpcClient>,
    mut focus: watch::Receiver<bool>,
    mut shutdown: watch::Receiver<bool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            let focused = *focus.borrow_and_update();
            send(
                &ipc,
                Request::ChatState {
                    id: Uuid::new_v4().to_string(),
                    focused,
                },
            )
            .await;
            tokio::select! {
                _ = shutdown.changed() => break,
                changed = focus.changed() => {
                    if changed.is_err() {
                        break;
                    }
                }
                () = tokio::time::sleep(RESEND_INTERVAL) => {}
            }
        }
    })
}

/// Tell the daemon the chat is exiting, giving up after [`CLOSE_TIMEOUT`].
pub(super) async fn report_closed(ipc: &IpcClient) {
    let req = Request::ChatClosed {
        id: Uuid::new_v4().to_string(),
    };
    let _ = tokio::time::timeout(CLOSE_TIMEOUT, send(ipc, req)).await;
}

async fn send(ipc: &IpcClient, req: Request) {
    let kind = req.kind();
    let result = match ipc.one_shot(req).await {
        Ok(stream) => stream.collect().await.map(drop),
        Err(e) => Err(e),
    };
    if let Err(e) = result {
        tracing::debug!("{kind} failed: {e}");
    }
}

#[cfg(test)]
mod tests {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    use super::*;

    #[test]
    fn focus_follows_reports_and_keypresses() {
        let (focus, rx) = watch::channel(true);
        let key = TermEvent::Key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE));
        for (ev, expected) in [
            (TermEvent::FocusLost, false),
            (TermEvent::Resize(80, 24), false),
            (key, true),
            (TermEvent::FocusLost, false),
            (TermEvent::FocusGained, true),
        ] {
            track(&focus, &ev);
            assert_eq!(*rx.borrow(), expected, "{ev:?}");
        }
    }
}
