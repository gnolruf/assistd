use thiserror::Error;

#[derive(Debug, Error)]
pub enum ChatClientError {
    #[error("HTTP transport error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("llama-server returned {status}: {body}")]
    Server { status: u16, body: String },

    #[error("SSE parse error: {0}")]
    Sse(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("summarization failed: {0}")]
    Summarize(String),

    #[error("response body exceeded cap of {cap} bytes")]
    BodyTooLarge { cap: usize },
}
