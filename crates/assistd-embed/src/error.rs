use std::sync::Arc;

use assistd_utils::readiness::NotReady;
use thiserror::Error;

/// Failures producing an embedding vector.
#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("embedder disabled")]
    Disabled,

    /// The embedder is still starting or failed to start.
    #[error("embedding is {0}")]
    Unavailable(NotReady),

    /// The supervised embed server was not serving, so nothing was sent.
    #[error("embed server is not ready; request not sent")]
    NotReady,

    #[error("build embed reqwest client: {0}")]
    Client(#[source] reqwest::Error),

    #[error("POST {url}: {source}")]
    Request {
        url: String,
        #[source]
        source: reqwest::Error,
    },

    /// Non-2xx response; `body` holds at most the first 200 chars.
    #[error("embed server returned {status} (body: {body})")]
    Status {
        status: reqwest::StatusCode,
        body: String,
    },

    #[error("parse /v1/embeddings response body: {0}")]
    Decode(#[source] reqwest::Error),

    /// The response did not hold exactly one `data` entry per input.
    #[error("embed response had {got} data entries for {expected} inputs")]
    CountMismatch { got: usize, expected: usize },

    /// A response entry's `index` was out of range or repeated.
    #[error("embed response entry index {index} is out of range or repeated for {expected} inputs")]
    BadIndex { index: usize, expected: usize },

    #[error("embed server returned an empty vector during dim probe")]
    DimProbeEmpty,

    /// The server's vector length changed after the dimension probe.
    #[error("embed server returned dim {got} but probe latched {expected}; refusing to mix")]
    DimMismatch { got: usize, expected: usize },

    /// The batch this input was part of failed for a reason not tied to any one input.
    #[error("embed batch failed: {0}")]
    BatchFailed(Arc<EmbedError>),
}

impl EmbedError {
    /// Whether retrying the inputs one at a time could isolate the failure to one of them.
    pub fn is_input_caused(&self) -> bool {
        match self {
            Self::Status { status, .. } => status.is_client_error(),
            Self::DimMismatch { .. } => true,
            _ => false,
        }
    }
}
