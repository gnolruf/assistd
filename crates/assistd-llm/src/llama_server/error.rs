//! llama-server lifecycle and control-plane errors.

use std::time::Duration;

use assistd_utils::child_server::ChildServerError;
use assistd_utils::http::BodyError;
use thiserror::Error;

use crate::LlamaClientError;

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

    #[error("llama-server model event stream: {0}")]
    EventStream(#[source] LlamaClientError),

    /// The server has no model download endpoint, which llama.cpp gained in b9688.
    #[error(
        "llama-server cannot download {model}: it has no download endpoint (llama.cpp b9688 or \
         newer); update it, or run `llama-server -hf {model}` once"
    )]
    DownloadUnsupported { model: String },

    /// The server found no file to fetch: the name is wrong, or Hugging Face
    /// was unreachable; its `reason` is the same for both.
    #[error("llama-server would not download {model}: {reason}; check the name and the network")]
    DownloadRefused { model: String, reason: String },

    #[error("llama-server failed to download {model}; its log says why")]
    DownloadFailed { model: String },

    #[error("the download of {model} received no data for {idle:?}")]
    DownloadStalled { model: String, idle: Duration },

    #[error("llama-server closed its event stream while downloading {model}")]
    DownloadInterrupted { model: String },

    /// The files arrived but the server lists them under another name, as it
    /// does for a quant tag that is missing or differs in case.
    #[error(
        "llama-server downloaded {model} but lists no model of that name; give the quant tag \
         exactly as `GET /models` shows it"
    )]
    DownloadNotListed { model: String },

    /// The supervised llama-server was not serving, so nothing was sent.
    #[error("llama-server is not ready; request not sent")]
    NotReady,
}
