//! The hidden `extract-worker` subcommand: the daemon's document extraction
//! child, spoken to over its stdin and stdout and logging to stderr.

use anyhow::Result;

/// Log filter when `RUST_LOG` is unset; the PDF library's remarks on each file
/// it reads and on the fonts it finds are left out.
const DEFAULT_LOG_DIRECTIVES: &str = "info,pdf_oxide=error,fontdb=error";

/// Serve the daemon until it closes the worker's stdin.
pub(crate) async fn run() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(crate::env_filter_or(DEFAULT_LOG_DIRECTIVES))
        .with_writer(std::io::stderr)
        .init();
    assistd_core::install_panic_hook();
    assistd_documents::serve().await?;
    Ok(())
}
