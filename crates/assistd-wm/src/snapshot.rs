//! Focus snapshot shared by the i3 and Sway backends. Backends project
//! their native window events onto `(kind, id, class, title)` and hand
//! them to [`apply_window_event`]; the update rules live here.

use std::sync::Arc;

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
    snap: &Arc<RwLock<Snapshot>>,
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

pub(crate) async fn apply_workspace_focus(snap: &Arc<RwLock<Snapshot>>, workspace: Option<String>) {
    snap.write().await.active_workspace = workspace;
}

pub(crate) async fn read_focused_id(snap: &Arc<RwLock<Snapshot>>) -> Option<WindowId> {
    snap.read().await.focused_id
}

/// `None` only when every field is empty; a partial snapshot still
/// yields a context.
pub(crate) async fn read_focused_context(
    snap: &Arc<RwLock<Snapshot>>,
) -> Option<FocusedWindowContext> {
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
mod tests {
    use super::*;

    fn id(n: u64) -> WindowId {
        WindowId::new(n).expect("test ids are non-zero")
    }

    fn snap() -> Arc<RwLock<Snapshot>> {
        Arc::new(RwLock::new(Snapshot::default()))
    }

    #[tokio::test]
    async fn focus_event_overwrites_all_fields() {
        let s = snap();
        apply_window_event(
            &s,
            WindowChangeKind::Focus,
            Some(id(42)),
            Some("Firefox".into()),
            Some("GitHub".into()),
        )
        .await;
        let r = s.read().await;
        assert_eq!(r.focused_id, Some(id(42)));
        assert_eq!(r.focused_class.as_deref(), Some("Firefox"));
        assert_eq!(r.focused_title.as_deref(), Some("GitHub"));
    }

    #[tokio::test]
    async fn title_event_for_focused_id_updates_title() {
        let s = snap();
        apply_window_event(
            &s,
            WindowChangeKind::Focus,
            Some(id(42)),
            Some("Firefox".into()),
            Some("Old".into()),
        )
        .await;
        apply_window_event(
            &s,
            WindowChangeKind::Title,
            Some(id(42)),
            Some("Firefox".into()),
            Some("New".into()),
        )
        .await;
        assert_eq!(s.read().await.focused_title.as_deref(), Some("New"));
    }

    #[tokio::test]
    async fn title_event_for_other_id_is_ignored() {
        let s = snap();
        apply_window_event(
            &s,
            WindowChangeKind::Focus,
            Some(id(42)),
            Some("Firefox".into()),
            Some("foreground".into()),
        )
        .await;
        apply_window_event(
            &s,
            WindowChangeKind::Title,
            Some(id(99)),
            Some("Firefox".into()),
            Some("background drift".into()),
        )
        .await;
        assert_eq!(s.read().await.focused_title.as_deref(), Some("foreground"));
    }

    #[tokio::test]
    async fn close_event_for_focused_id_clears_focus() {
        let s = snap();
        apply_window_event(
            &s,
            WindowChangeKind::Focus,
            Some(id(42)),
            Some("Firefox".into()),
            Some("GitHub".into()),
        )
        .await;
        apply_window_event(
            &s,
            WindowChangeKind::Close,
            Some(id(42)),
            Some("Firefox".into()),
            Some("GitHub".into()),
        )
        .await;
        let r = s.read().await;
        assert!(r.focused_id.is_none());
        assert!(r.focused_class.is_none());
        assert!(r.focused_title.is_none());
    }

    #[tokio::test]
    async fn close_event_for_other_id_is_ignored() {
        let s = snap();
        apply_window_event(
            &s,
            WindowChangeKind::Focus,
            Some(id(42)),
            Some("Firefox".into()),
            Some("GitHub".into()),
        )
        .await;
        apply_window_event(
            &s,
            WindowChangeKind::Close,
            Some(id(99)),
            Some("Other".into()),
            Some("Other".into()),
        )
        .await;
        assert_eq!(s.read().await.focused_id, Some(id(42)));
    }

    #[tokio::test]
    async fn workspace_focus_updates_active_workspace() {
        let s = snap();
        apply_workspace_focus(&s, Some("3".into())).await;
        assert_eq!(s.read().await.active_workspace.as_deref(), Some("3"));
    }

    #[tokio::test]
    async fn read_focused_context_returns_none_for_empty() {
        let s = snap();
        assert!(read_focused_context(&s).await.is_none());
    }

    #[tokio::test]
    async fn read_focused_context_returns_some_for_partial() {
        let s = snap();
        apply_workspace_focus(&s, Some("3".into())).await;
        let ctx = read_focused_context(&s).await.unwrap();
        assert!(ctx.id.is_none());
        assert_eq!(ctx.workspace.as_deref(), Some("3"));
    }
}
