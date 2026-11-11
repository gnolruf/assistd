//! Live Sway integration test, ignored by default; run it with
//! `cargo test -p assistd-wm --test sway_live -- --ignored --nocapture`.
//! Read-only: it never changes the user's window state.

#![cfg(feature = "sway")]

mod common;

use std::time::Duration;

use assistd_config::Compositor;
use tokio::sync::watch;

#[tokio::test]
#[ignore = "requires a live sway session ($SWAYSOCK reachable)"]
async fn connects_and_passes_shared_assertions() {
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let (wm, handle) = assistd_wm::start(Compositor::Sway, shutdown_rx).await;
    let handle = handle.expect("sway backend did not start; is sway running and $SWAYSOCK set?");

    common::assert_is_connected(&wm);
    common::assert_focused_window_present(&wm).await;
    common::assert_focused_context_agrees(&wm).await;
    common::assert_at_least_one_workspace_focused(&wm).await;
    common::assert_focused_window_in_list_windows(&wm).await;

    let outputs = wm
        .list_outputs()
        .await
        .expect("sway list_outputs query failed");
    println!("sway outputs = {outputs:?}");
    assert!(
        !outputs.is_empty(),
        "expected at least one output from a running sway session"
    );

    tokio::time::sleep(Duration::from_millis(50)).await;

    let _ = shutdown_tx.send(true);
    handle.shutdown().await;
}
