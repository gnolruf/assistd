use thiserror::Error;

/// Failures producing an embedding vector.
#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("embedder disabled")]
    Disabled,

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

    #[error("embed response had no data entries")]
    NoData,

    #[error("embed server returned an empty vector during dim probe")]
    DimProbeEmpty,

    /// The server's vector length changed after the dimension probe.
    #[error("embed server returned dim {got} but probe latched {expected}; refusing to mix")]
    DimMismatch { got: usize, expected: usize },
}
