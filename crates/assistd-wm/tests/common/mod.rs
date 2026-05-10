//! Shared, read-only assertions exercised by both `i3_live.rs` and
//! `sway_live.rs`. These are the "i3 test suite" that the Sway backend
//! must also pass: every assertion reads cached or freshly-fetched
//! state and never issues a focus / move command, so they're safe to
//! run against the developer's live session.
//!
//! Both backends store this module via `mod common;` at the top of
//! their integration test file. Cargo treats `tests/common/mod.rs` as
//! a non-test helper module; only files directly under `tests/` are
//! compiled as test binaries.

use std::sync::Arc;

use assistd_wm::WindowManager;

/// `focused_window()` should return Some(class) for a session with at
/// least one mapped, focused window, which is the common case on a
/// developer machine running the test.
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

/// `focused_context().class` should agree with `focused_window()`.
/// This is the contract the daemon relies on when injecting passive
/// desktop context into the LLM's per-turn system prompt.
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
    // PR 3b: focused_window() returns the compositor con_id; the
    // focused_context().id field carries the same id, surfaced from the
    // same snapshot read. Both are `Option<WindowId>` (NonZeroU64).
    assert_eq!(
        ctx.id, focused,
        "focused_context().id should agree with focused_window()"
    );
}

/// At least one workspace must be marked focused on a session that's
/// actually displaying windows. (Multi-monitor setups have one focused
/// workspace per output; we don't assert *exactly* one because all
/// outputs report `focused == true` for their respective active
/// workspaces.)
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

/// Sanity check: the focused window's class appears in the
/// `list_windows()` enumeration. Catches drift between the cached
/// snapshot and the freshly-walked tree.
pub async fn assert_focused_window_in_list_windows(wm: &Arc<dyn WindowManager>) {
    let focused = match wm
        .focused_window()
        .await
        .expect("focused_window query failed")
    {
        Some(c) => c,
        None => return, // No focused window → nothing to check.
    };
    let windows = wm.list_windows().await.expect("list_windows query failed");
    assert!(
        windows.iter().any(|w| w.id == focused),
        "expected focused class {focused:?} to appear in list_windows() result"
    );
}

/// Backend reports itself as connected. Both real backends override
/// `is_connected()` to true; only `NoWindowManager` returns false.
pub async fn assert_is_connected(wm: &Arc<dyn WindowManager>) {
    assert!(
        wm.is_connected(),
        "live backend should report is_connected() == true"
    );
}
