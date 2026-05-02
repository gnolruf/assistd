//! Shared focus-snapshot state and event-apply logic for the i3 and
//! Sway backends.
//!
//! Both compositors deliver the same kinds of `Window` events (Focus,
//! Title, Close) and the same `Workspace::Focus` event. The race rules
//! for keeping the snapshot consistent under those events are
//! identical (key on the unique con_id, drop title updates from
//! background windows, etc.). Originally those rules lived in two
//! near-identical `handle_window_event` copies — one per backend.
//!
//! Each backend now extracts the container's `(id, class, title)` from
//! its native `WindowEvent` payload (the only piece that's compositor-
//! specific, since the reply types differ between `tokio-i3ipc` and
//! `swayipc-async`) and hands off to [`apply_window_event`] /
//! [`apply_workspace_focus`] here. The "what changes when" lives in
//! one place; the "how do I parse the IPC frame" stays in the
//! backend-specific module.
//!
//! No sealed `Compositor` trait yet — that was the more aggressive
//! goal in the M9 plan; it requires an async-trait + associated-stream
//! shape that risks GAT pitfalls and would force every backend to
//! own-and-pin its event stream. The simpler share-the-rules form
//! removes the bulk of the duplication without the lifetime gymnastics.
//! Revisit if PR 5 (reconnection supervisor) ends up wanting the same
//! generic shape.

use std::sync::Arc;

use tokio::sync::RwLock;

use crate::{FocusedWindowContext, WindowId};

/// Cached focus state. Shared between the i3 and Sway backends.
///
/// Stores the compositor con_id (the [`WindowId`] returned to callers)
/// alongside the raw `app_id` / X11 class string and window title used
/// to render [`FocusedWindowContext::class`] / `::title` for the
/// system-prompt context block.
#[derive(Default, Clone)]
pub(crate) struct Snapshot {
    pub focused_id: Option<WindowId>,
    pub focused_class: Option<String>,
    pub focused_title: Option<String>,
    pub active_workspace: Option<String>,
}

/// The subset of `Window`-event change kinds the snapshot reacts to.
/// Backends translate their native enum (`tokio_i3ipc::event::WindowChange`
/// / `swayipc_async::WindowChange`) into this shared shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WindowChangeKind {
    Focus,
    Title,
    Close,
}

/// Apply a `Window` event to the cached focus snapshot.
///
/// Race rules (key on con_id rather than class — two windows of the
/// same app would otherwise alias each other in title/close updates):
/// - `Focus` overwrites id + class + title from the new container.
/// - `Title` only takes effect when the event's con_id matches the
///   currently-focused id. Title events from background windows must
///   not overwrite the foreground title; class also drifts to match
///   the latest container metadata for the same id.
/// - `Close` clears id + class + title only when the closed container
///   is the focused one. Workspace is left intact.
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

/// Apply a `Workspace::Focus` event to the snapshot. Used by both
/// backends to keep `active_workspace` aligned with what the
/// compositor reports as the currently-focused workspace.
pub(crate) async fn apply_workspace_focus(snap: &Arc<RwLock<Snapshot>>, workspace: Option<String>) {
    snap.write().await.active_workspace = workspace;
}

/// Read the currently-focused window's id from the snapshot. Cheap
/// (single RwLock read), so callers don't need to cache the result.
pub(crate) async fn read_focused_id(snap: &Arc<RwLock<Snapshot>>) -> Option<WindowId> {
    snap.read().await.focused_id
}

/// Build a [`FocusedWindowContext`] from the snapshot for the daemon's
/// per-turn system-prompt injection. Returns `None` only when every
/// field is empty — a partial snapshot still produces a context block.
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
        // The crucial regression check: a background window's Title
        // event must NOT overwrite the foreground title. Pre-PR-3b
        // this used to be keyed on class, which aliased two windows
        // of the same app.
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
            Some(id(99)), // different con_id
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
