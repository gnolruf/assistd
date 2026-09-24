//! Focus snapshot shared by the i3 and Sway backends. Backends project
//! their native window events onto `(kind, id, class, title)` and hand
//! them to [`apply_window_event`]; the update rules live here.

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

/// Apply a window event to the snapshot. Events are keyed on con_id,
/// not class, so two windows of the same app never alias: `Title` and
/// `Close` only take effect when their id is the focused one.
pub(crate) async fn apply_window_event(
    snap: &RwLock<Snapshot>,
    kind: WindowChangeKind,
    id: Option<WindowId>,
    class: Option<String>,
    title: Option<String>,
) {
    match kind {
        WindowChangeKind::Focus => {
            let mut s = snap.write().await;
            s.focused_id = id;
            s.focused_class = class;
            s.focused_title = title;
        }
        WindowChangeKind::Title => {
            let mut s = snap.write().await;
            if s.focused_id == id && id.is_some() {
                s.focused_title = title;
                s.focused_class = class;
            }
        }
        WindowChangeKind::Close => {
            let mut s = snap.write().await;
            if s.focused_id == id && id.is_some() {
                s.focused_id = None;
                s.focused_class = None;
                s.focused_title = None;
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
    let s = snap.read().await;
    if s.focused_id.is_none()
        && s.focused_class.is_none()
        && s.focused_title.is_none()
        && s.active_workspace.is_none()
    {
        return None;
    }
    Some(FocusedWindowContext {
        id: s.focused_id,
        class: s.focused_class.clone(),
        title: s.focused_title.clone(),
        workspace: s.active_workspace.clone(),
    })
}

#[cfg(test)]
mod tests;
