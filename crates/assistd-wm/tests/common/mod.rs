//! Read-only assertions shared by the live i3 and Sway tests. None
//! issues a focus or move command, so they are safe to run against a
//! live session.

use std::sync::Arc;

use assistd_wm::WindowManager;

pub async fn assert_focused_window_present(wm: &Arc<dyn WindowManager>) {
    let focused = wm
        .focused_window()
        .await
        .expect("focused_window query failed");
    assert!(
        focused.is_some(),
        "expected at least one focused window in the live session"
    );
}

/// Asserts that `focused_context().id` matches `focused_window()`.
pub async fn assert_focused_context_agrees(wm: &Arc<dyn WindowManager>) {
    let focused = wm
        .focused_window()
        .await
        .expect("focused_window query failed");
    let ctx = wm
        .focused_context()
        .await
        .expect("focused_context query failed")
        .expect("expected Some(FocusedWindowContext) for a focused session");
    assert_eq!(
        ctx.id, focused,
        "focused_context().id should agree with focused_window()"
    );
}

/// Multi-monitor setups report one focused workspace per output, so
/// this asserts at least one rather than exactly one.
pub async fn assert_at_least_one_workspace_focused(wm: &Arc<dyn WindowManager>) {
    let workspaces = wm
        .list_workspaces()
        .await
        .expect("list_workspaces query failed");
    assert!(
        !workspaces.is_empty(),
        "expected list_workspaces() to return at least one row"
    );
    assert!(
        workspaces.iter().any(|w| w.focused),
        "expected at least one workspace to be marked focused"
    );
}

/// Catches drift between the cached focus snapshot and a freshly
/// walked tree.
pub async fn assert_focused_window_in_list_windows(wm: &Arc<dyn WindowManager>) {
    let focused = match wm
        .focused_window()
        .await
        .expect("focused_window query failed")
    {
        Some(c) => c,
        None => return,
    };
    let windows = wm.list_windows().await.expect("list_windows query failed");
    assert!(
        windows.iter().any(|w| w.id == focused),
        "expected focused class {focused:?} to appear in list_windows() result"
    );
}

pub async fn assert_is_connected(wm: &Arc<dyn WindowManager>) {
    assert!(
        wm.is_connected(),
        "live backend should report is_connected() == true"
    );
}
