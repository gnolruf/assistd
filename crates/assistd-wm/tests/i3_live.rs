//! Live i3 integration test. `#[ignore]` so it never runs in CI or
//! `cargo test` — opt in with:
//!
//! ```text
//! cargo test -p assistd-wm --test i3_live -- --ignored --nocapture
//! ```
//!
//! Read-only: connects, reads the focused window's class, and shuts
//! down. Does NOT issue `focus` or `move_to_workspace` against the live
//! session — those would mutate the user's window state.

use std::time::Duration;

use assistd_wm::{I3Backend, WindowManager};
use tokio::sync::watch;

#[tokio::test]
#[ignore = "requires a live i3 session ($I3SOCK reachable)"]
async fn connects_and_reads_focused_window() {
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let handle = I3Backend::start(shutdown_rx)
        .await
        .expect("I3Backend::start failed — is i3 running and $I3SOCK set?");

    // Snapshot is seeded synchronously inside `start()` from get_tree(),
    // so focused_window() should already have a value here without
    // waiting for an event.
    let focused = handle
        .backend
        .focused_window()
        .await
        .expect("focused_window query failed");
    println!("focused class = {:?}", focused);
    assert!(
        focused.is_some(),
        "expected at least one focused window in the live i3 session"
    );

    // Give the event task a brief window to confirm it's running and
    // subscribed (no assertion — just exercising the loop).
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Graceful shutdown via the watch channel — same path the daemon
    // exercises on SIGTERM.
    let _ = shutdown_tx.send(true);
    handle.shutdown().await;
}
