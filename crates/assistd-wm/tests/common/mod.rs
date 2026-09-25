//! Read-only assertions shared by the live i3 and Sway tests, safe to
//! run against a live session.

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
    let context = wm
        .focused_context()
        .await
        .expect("focused_context query failed")
        .expect("expected Some(FocusedWindowContext) for a focused session");
    assert_eq!(
        context.id, focused,
        "focused_context().id should agree with focused_window()"
    );
}

/// At least one, since multi-monitor setups focus one workspace per output.
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
        workspaces.iter().any(|workspace| workspace.focused),
        "expected at least one workspace to be marked focused"
    );
}

/// Catches drift between the cached focus snapshot and the live tree.
pub async fn assert_focused_window_in_list_windows(wm: &Arc<dyn WindowManager>) {
    let Some(focused) = wm
        .focused_window()
        .await
        .expect("focused_window query failed")
    else {
        return;
    };
    let windows = wm.list_windows().await.expect("list_windows query failed");
    assert!(
        windows.iter().any(|window| window.id == focused),
        "expected focused class {focused:?} to appear in list_windows() result"
    );
}

pub async fn assert_is_connected(wm: &Arc<dyn WindowManager>) {
    assert!(
        wm.is_connected(),
        "live backend should report is_connected() == true"
    );
}
