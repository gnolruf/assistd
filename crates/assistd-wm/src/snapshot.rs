//! Focus snapshot shared by the i3 and Sway backends, updated from
//! window events projected onto `(kind, id, class, title)`.

use tokio::sync::RwLock;

use crate::{FocusedWindowContext, WindowId};

/// Cached focus state.
#[derive(Default, Clone)]
pub(crate) struct Snapshot {
    pub focused_id: Option<WindowId>,
    pub focused_class: Option<String>,
    pub focused_title: Option<String>,
    pub active_workspace: Option<String>,
}

/// The window-event kinds the snapshot reacts to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WindowChangeKind {
    Focus,
    Title,
    Close,
}

/// Apply a window event to the snapshot. `Title` and `Close` only take
/// effect when their id is the focused one.
pub(crate) async fn apply_window_event(
    snap: &RwLock<Snapshot>,
    kind: WindowChangeKind,
    id: Option<WindowId>,
    class: Option<String>,
    title: Option<String>,
) {
    match kind {
        WindowChangeKind::Focus => {
            let mut state = snap.write().await;
            state.focused_id = id;
            state.focused_class = class;
            state.focused_title = title;
        }
        WindowChangeKind::Title => {
            let mut state = snap.write().await;
            if state.focused_id == id && id.is_some() {
                state.focused_title = title;
                state.focused_class = class;
            }
        }
        WindowChangeKind::Close => {
            let mut state = snap.write().await;
            if state.focused_id == id && id.is_some() {
                state.focused_id = None;
                state.focused_class = None;
                state.focused_title = None;
            }
        }
    }
}

pub(crate) async fn apply_workspace_focus(snap: &RwLock<Snapshot>, workspace: Option<String>) {
    snap.write().await.active_workspace = workspace;
}

pub(crate) async fn read_focused_id(snap: &RwLock<Snapshot>) -> Option<WindowId> {
    snap.read().await.focused_id
}

/// `None` only when every field is empty; a partial snapshot still
/// yields a context.
pub(crate) async fn read_focused_context(snap: &RwLock<Snapshot>) -> Option<FocusedWindowContext> {
    let state = snap.read().await;
    if state.focused_id.is_none()
        && state.focused_class.is_none()
        && state.focused_title.is_none()
        && state.active_workspace.is_none()
    {
        return None;
    }
    Some(FocusedWindowContext {
        id: state.focused_id,
        class: state.focused_class.clone(),
        title: state.focused_title.clone(),
        workspace: state.active_workspace.clone(),
    })
}

#[cfg(test)]
mod tests;
