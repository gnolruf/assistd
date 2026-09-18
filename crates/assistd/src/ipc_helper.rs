//! Shared IPC plumbing for the CLI subcommands.

use anyhow::{Error, Result};
use assistd_ipc::{Event, IpcClient, IpcClientError, Request};

/// Send `req` and hand every event, terminal ones included, to
/// `on_event`. Returns after `Done`; on `Error` prints the daemon's
/// message and exits 1 after `on_event` has seen it. A connection that
/// closes without a terminal event is an error.
pub async fn run_one_shot(
    req: Request,
    mut on_event: impl FnMut(&Event) -> Result<()>,
) -> Result<()> {
    let mut stream = IpcClient::new()
        .one_shot(req)
        .await
        .map_err(map_not_reachable)?;
    loop {
        let Some(event) = stream.next_event().await? else {
            anyhow::bail!("daemon closed the connection without sending a terminal event");
        };
        on_event(&event)?;
        match event {
            Event::Done { .. } => return Ok(()),
            Event::Error { message, .. } => {
                eprintln!("daemon error: {message}");
                std::process::exit(1);
            }
            _ => {}
        }
    }
}

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
