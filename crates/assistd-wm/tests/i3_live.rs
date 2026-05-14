#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Live i3 integration test. `#[ignore]` so it never runs in CI or
//! `cargo test`; opt in with:
//!
//! ```text
//! cargo test -p assistd-wm --test i3_live -- --ignored --nocapture
//! ```
//!
//! Read-only: connects, exercises the shared assertions in `common`,
//! and shuts down. Does NOT issue `focus` or `move_to_workspace`
//! against the live session; those would mutate the user's window
//! state.

mod common;

use std::sync::Arc;
use std::time::Duration;

use assistd_wm::{I3Backend, WindowManager};
use tokio::sync::watch;

#[tokio::test]
#[ignore = "requires a live i3 session ($I3SOCK reachable)"]
async fn connects_and_passes_shared_assertions() {
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let handle = I3Backend::start(shutdown_rx)
        .await
        .expect("I3Backend::start failed; is i3 running and $I3SOCK set?");

    let wm: Arc<dyn WindowManager> = handle.backend.clone();

    common::assert_is_connected(&wm).await;
    common::assert_focused_window_present(&wm).await;
    common::assert_focused_context_agrees(&wm).await;
    common::assert_at_least_one_workspace_focused(&wm).await;
    common::assert_focused_window_in_list_windows(&wm).await;

    // Give the event task a brief window to confirm it's running and
    // subscribed (no assertion, just exercising the loop).
    tokio::time::sleep(Duration::from_millis(50)).await;

    let _ = shutdown_tx.send(true);
    handle.shutdown().await;
}
