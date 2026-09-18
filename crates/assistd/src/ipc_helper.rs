//! Shared IPC plumbing for the CLI subcommands.

use anyhow::Error;
use assistd_ipc::IpcClientError;

/// Convert an [`IpcClientError`] to `anyhow`, phrasing `NotReachable`
/// as "daemon is not running" with the socket path.
pub fn map_not_reachable(e: IpcClientError) -> Error {
    match e {
        IpcClientError::NotReachable { path, source } => Error::msg(format!(
            "assistd daemon is not running (could not connect to {}): {source}",
            path.display()
        )),
        other => Error::from(other),
    }
}
