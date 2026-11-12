//! One-shot CLI subcommands: each sends one request to the running daemon
//! and prints the events it streams back.

use std::io::{self, Write};

use anyhow::{Error, Result};
use assistd_ipc::{Event, IpcClient, IpcClientError, Request};

use terminal_text::escape_controls;

pub(crate) mod listen;
pub(crate) mod mcp;
pub(crate) mod memory;
pub(crate) mod presence;
pub(crate) mod ptt;
pub(crate) mod query;
pub(crate) mod task;
pub(crate) mod terminal_text;
pub(crate) mod voice_ctl;

/// Send `req` and hand every event, terminal ones included, to `on_event`.
/// Returns after `Done`; on `Error` prints the message and exits 1. A connection
/// that closes without a terminal event is an error.
async fn run_one_shot(req: Request, on_event: impl FnMut(&Event) -> Result<()>) -> Result<()> {
    let Err(failure) = IpcClient::new().request(req, on_event).await else {
        return Ok(());
    };
    match failure.downcast() {
        Ok(IpcClientError::Daemon(message)) => {
            writeln!(io::stderr(), "daemon error: {}", escape_controls(&message))?;
            std::process::exit(1);
        }
        Ok(other) => Err(map_not_reachable(other)),
        Err(other) => Err(other),
    }
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
