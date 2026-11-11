//! Live i3 integration test, ignored by default; run it with
//! `cargo test -p assistd-wm --test i3_live -- --ignored --nocapture`.
//! Read-only: it never changes the user's window state.

#![cfg(feature = "i3")]

mod common;

use std::time::Duration;

use assistd_config::Compositor;
use tokio::sync::watch;

#[tokio::test]
#[ignore = "requires a live i3 session ($I3SOCK reachable)"]
async fn connects_and_passes_shared_assertions() {
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let (wm, handle) = assistd_wm::start(Compositor::I3, shutdown_rx).await;
    let handle = handle.expect("i3 backend did not start; is i3 running and $I3SOCK set?");

    common::assert_is_connected(&wm);
    common::assert_focused_window_present(&wm).await;
    common::assert_focused_context_agrees(&wm).await;
    common::assert_at_least_one_workspace_focused(&wm).await;
    common::assert_focused_window_in_list_windows(&wm).await;

    tokio::time::sleep(Duration::from_millis(50)).await;

    let _ = shutdown_tx.send(true);
    handle.shutdown().await;
}
