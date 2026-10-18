use std::io;
use std::path::PathBuf;

use reqwest::StatusCode;
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

/// Why a search returned no results page.
#[derive(Debug, Error)]
pub enum SearchError {
    #[error("search server is not ready; request not sent")]
    NotServing,

    #[error("build search reqwest client: {0}")]
    Client(#[source] reqwest::Error),

    #[error("search request failed: {0}")]
    Transport(#[source] reqwest::Error),

    #[error("search server returned {0}")]
    Status(StatusCode),

    #[error("search response exceeded {max} bytes")]
    TooLarge { max: usize },

    #[error("parse search response: {0}")]
    Decode(#[source] serde_json::Error),
}
