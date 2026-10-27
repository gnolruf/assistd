use std::time::Duration;

use assistd_utils::child_server::ChildServerError;
use assistd_utils::http::BodyError;
use thiserror::Error;

/// Errors produced by the llama-server lifecycle manager and HTTP control plane.
#[derive(Debug, Error)]
pub enum LlamaServerError {
    #[error(transparent)]
    Server(#[from] ChildServerError),

    #[error("HTTP client error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("llama-server control response: {0}")]
    Body(#[from] BodyError),

    #[error("parse llama-server control response: {0}")]
    Json(#[from] serde_json::Error),

    #[error("llama-server {method} {path} returned status {status}")]
    ControlHttp {
        method: &'static str,
        path: &'static str,
        status: u16,
    },

    #[error("llama-server did not report {model} loaded within {timeout:?}")]
    LoadTimeout { model: String, timeout: Duration },

    /// The supervised llama-server was not serving, so nothing was sent.
    #[error("llama-server is not ready; request not sent")]
    NotReady,
}
