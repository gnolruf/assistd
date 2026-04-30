//! Shared IPC client error mapping for the CLI subcommands.
//!
//! Every CLI subcommand wants the same "daemon not running" error
//! formatted with the socket path so the user knows where to look.
//! Centralized here so a path change or message tweak lands in one
//! place.

use anyhow::Error;
use assistd_ipc::IpcClientError;

/// Convert an [`IpcClientError`] into an `anyhow::Error`, special-casing
/// `NotReachable` with a CLI-friendly "daemon is not running" message.
pub fn map_not_reachable(e: IpcClientError) -> Error {
    match e {
        IpcClientError::NotReachable { path, source } => Error::msg(format!(
            "assistd daemon is not running (could not connect to {}): {source}",
            path.display()
        )),
        other => Error::from(other),
    }
}
