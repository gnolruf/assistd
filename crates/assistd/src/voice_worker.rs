//! The hidden `voice-worker` subcommand: the daemon's voice child, spoken to
//! over its stdin and stdout and logging to stderr.

use anyhow::Result;
use assistd_utils::tracing_init::env_filter_or;

/// Log filter when `RUST_LOG` is unset.
const DEFAULT_LOG_DIRECTIVES: &str = "info";

/// Serve the daemon until it closes the worker's stdin.
pub(crate) async fn run() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(env_filter_or(DEFAULT_LOG_DIRECTIVES))
        .with_writer(std::io::stderr)
        .init();
    assistd_core::install_panic_hook();
    assistd_voice::worker::serve().await?;
    Ok(())
}
