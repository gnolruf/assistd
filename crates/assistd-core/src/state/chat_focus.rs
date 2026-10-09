//! Whether the user's chat window has keyboard focus, as last reported by
//! the chat and checked against its process still running.

use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::mpsc;

use assistd_ipc::Event;

use super::AppState;

/// The single chat's last focus report, plus the value last broadcast.
#[derive(Debug, Default)]
pub(in crate::state) struct ChatFocusSlot {
    inner: Mutex<SlotState>,
}

#[derive(Debug, Default)]
struct SlotState {
    chat: Option<ChatClient>,
    published: bool,
}

#[derive(Debug, Clone, Copy)]
struct ChatClient {
    /// `None` when the socket could not report the sender's PID; such a
    /// chat is never treated as dead.
    pid: Option<u32>,
    focused: bool,
}

impl ChatFocusSlot {
    /// Record a focus report from the chat running as `pid`.
    fn report(&self, pid: Option<u32>, focused: bool) {
        self.inner.lock().chat = Some(ChatClient { pid, focused });
    }

    /// Forget the chat running as `pid`; a report from another PID stays.
    fn close(&self, pid: Option<u32>) {
        let mut state = self.inner.lock();
        if state.chat.is_some_and(|chat| chat.pid == pid) {
            state.chat = None;
        }
    }

    /// Whether a live chat has focus, forgetting a chat whose process is
    /// gone. `Some` carries the new value when it differs from the one
    /// last returned this way.
    fn refresh(&self) -> (bool, Option<bool>) {
        let mut state = self.inner.lock();
        if state
            .chat
            .is_some_and(|chat| chat.pid.is_some_and(|pid| !process_is_alive(pid)))
        {
            state.chat = None;
        }
        let focused = state.chat.is_some_and(|chat| chat.focused);
        let changed = (focused != state.published).then_some(focused);
        state.published = focused;
        (focused, changed)
    }
}

impl AppState {
    pub(super) async fn handle_chat_state(
        self: Arc<Self>,
        id: String,
        focused: bool,
        peer_pid: Option<u32>,
        tx: mpsc::Sender<Event>,
    ) {
        self.runtime.chat_focus.report(peer_pid, focused);
        self.refresh_chat_focus(&id);
        let _ = tx.send(Event::Done { id }).await;
    }

    pub(super) async fn handle_chat_closed(
        self: Arc<Self>,
        id: String,
        peer_pid: Option<u32>,
        tx: mpsc::Sender<Event>,
    ) {
        self.runtime.chat_focus.close(peer_pid);
        self.refresh_chat_focus(&id);
        let _ = tx.send(Event::Done { id }).await;
    }

    pub(super) async fn handle_get_chat_focus(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) {
        let focused = self.refresh_chat_focus(&id);
        let _ = tx
            .send(Event::ChatFocus {
                id: id.clone(),
                focused,
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
    }

    /// Whether a live chat has focus, broadcasting `ChatFocus` tagged with
    /// `id` when that changed since the last broadcast.
    pub(super) fn refresh_chat_focus(&self, id: &str) -> bool {
        let (focused, changed) = self.runtime.chat_focus.refresh();
        if let Some(focused) = changed {
            self.runtime.publish(&Event::ChatFocus {
                id: id.to_string(),
                focused,
            });
        }
        focused
    }
}

fn process_is_alive(pid: u32) -> bool {
    Path::new("/proc").join(pid.to_string()).exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEAD_PID: u32 = u32::MAX;

    #[test]
    fn refresh_reports_only_changes() {
        let slot = ChatFocusSlot::default();
        assert_eq!(slot.refresh(), (false, None));
        slot.report(Some(std::process::id()), true);
        assert_eq!(slot.refresh(), (true, Some(true)));
        assert_eq!(slot.refresh(), (true, None));
        slot.report(Some(std::process::id()), false);
        assert_eq!(slot.refresh(), (false, Some(false)));
    }

    #[test]
    fn dead_chat_reads_as_unfocused_and_is_forgotten() {
        let slot = ChatFocusSlot::default();
        slot.report(Some(DEAD_PID), true);
        assert_eq!(slot.refresh(), (false, None));
        assert!(slot.inner.lock().chat.is_none());
    }

    #[test]
    fn close_from_another_pid_keeps_the_chat() {
        let slot = ChatFocusSlot::default();
        slot.report(Some(std::process::id()), true);
        slot.close(Some(DEAD_PID));
        assert_eq!(slot.refresh(), (true, Some(true)));
        slot.close(Some(std::process::id()));
        assert_eq!(slot.refresh(), (false, Some(false)));
    }
}
