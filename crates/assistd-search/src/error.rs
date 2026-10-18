use std::io;
use std::path::PathBuf;

use thiserror::Error;

/// Why a SearXNG instance could not be prepared for launch.
#[derive(Debug, Error)]
pub enum SearchServerError {
    #[error("failed to generate the SearXNG secret: {0}")]
    Secret(#[source] io::Error),

    #[error("failed to write SearXNG settings to {}: {source}", path.display())]
    Settings {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}
