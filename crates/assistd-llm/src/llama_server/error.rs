use std::time::Duration;

use assistd_utils::child_server::ChildServerError;
use thiserror::Error;

/// Errors produced by the llama-server lifecycle manager and HTTP control plane.
#[derive(Debug, Error)]
pub enum LlamaServerError {
    #[error(transparent)]
    Server(#[from] ChildServerError),

    #[error("HTTP client error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("llama-server {method} {path} returned status {status}")]
    ControlHttp {
        method: &'static str,
        path: &'static str,
        status: u16,
    },

    #[error("llama-server did not report {model} loaded within {timeout:?}")]
    LoadTimeout { model: String, timeout: Duration },
}
