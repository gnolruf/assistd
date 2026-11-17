//! Server handles for tests, backed by no supervisor or child and reporting what the test
//! scripts.

use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::watch;

use super::service::{ChildServer, ChildServerStatus, ReadyState};

/// A handle reporting `state` and `pid` until the returned sender scripts otherwise.
/// Dropping the sender reads as a stopped supervisor.
pub fn scripted_server(
    state: ReadyState,
    pid: Option<u32>,
) -> (watch::Sender<ReadyState>, ChildServer) {
    let (ready_tx, status) = scripted_status(state, pid);
    let server = ChildServer {
        server: "scripted",
        task: None,
        status,
    };
    (ready_tx, server)
}

/// A status driven by the returned sender, reporting `pid` as the child's.
/// Dropping the sender reads as a stopped supervisor.
pub fn scripted_status(
    state: ReadyState,
    pid: Option<u32>,
) -> (watch::Sender<ReadyState>, ChildServerStatus) {
    let (ready_tx, ready_rx) = watch::channel(state);
    let status = ChildServerStatus {
        ready_rx,
        pid: Arc::new(Mutex::new(pid)),
    };
    (ready_tx, status)
}
