//! One-shot CLI subcommands: each sends one request to the running daemon
//! and prints the events it streams back.

use anyhow::{Error, Result};
use assistd_ipc::{Event, IpcClient, IpcClientError, Request};

pub(crate) mod listen;
pub(crate) mod mcp;
pub(crate) mod memory;
pub(crate) mod presence;
pub(crate) mod ptt;
pub(crate) mod task;
pub(crate) mod terminal_text;
pub(crate) mod voice_ctl;

/// Send `req` and hand every event, terminal ones included, to `on_event`.
/// Returns after `Done`; fails with [`IpcClientError::Daemon`] on `Error`, and when the
/// connection closes without a terminal event.
async fn run_one_shot(req: Request, on_event: impl FnMut(&Event) -> Result<()>) -> Result<()> {
    IpcClient::new()
        .request(req, on_event)
        .await
        .map_err(|failure| {
            failure
                .downcast()
                .map_or_else(|other| other, map_not_reachable)
        })
}

/// Phrase `NotReachable` as "daemon is not running" with the socket path.
fn map_not_reachable(e: IpcClientError) -> Error {
    match e {
        IpcClientError::NotReachable { path, source } => Error::msg(format!(
            "assistd daemon is not running (could not connect to {}): {source}",
            path.display()
        )),
        other => Error::from(other),
    }
}
