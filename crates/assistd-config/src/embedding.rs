//! Embedding-server and semantic-search configuration.

use std::net::IpAddr;
use std::num::{NonZeroU16, NonZeroU32, NonZeroU64};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::custom_args::{CustomArgs, EmbeddingServer};
use crate::defaults::{
    DEFAULT_EMBEDDING_AUTO_INJECT, DEFAULT_EMBEDDING_ENABLED, DEFAULT_EMBEDDING_GPU_LAYERS,
    DEFAULT_EMBEDDING_HOST, DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_PORT,
    DEFAULT_EMBEDDING_READY_TIMEOUT_SECS, DEFAULT_EMBEDDING_SERVER_BINARY, DEFAULT_EMBEDDING_TOP_K,
};

/// Dedicated embedding server and semantic-recall settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct EmbeddingConfig {
    /// When `false`, semantic recall is unavailable.
    pub enabled: bool,
    /// `--hf-repo` value, `<owner>/<repo>:<quant>`. The suffix must be a
    /// quant tag; a `.gguf` filename fails with "no GGUF files found".
    pub model: String,
    /// Server binary path, or a name on `$PATH`. Must not be empty when
    /// enabled.
    pub server_binary: PathBuf,
    /// Bind host. Must be loopback: the server has no authentication.
    pub host: IpAddr,
    /// Bind port. Must differ from `model.port`.
    pub port: NonZeroU16,
    /// `-ngl` count. `0` keeps it on CPU.
    pub gpu_layers: u32,
    /// Last-ditch cap, in seconds, on becoming healthy; the wait never trips
    /// while the model is still downloading or loading.
    pub ready_timeout_secs: NonZeroU64,
    /// Extra server arguments. Flags assistd sets itself, or that expose
    /// files, tools or state through the server, fail to parse.
    pub custom_args: CustomArgs<EmbeddingServer>,
    /// Nearest-neighbour matches per query, for auto-injection and as the
    /// `reminisce` default. `1..=20`, the most `reminisce` returns.
    pub top_k: NonZeroU32,
    /// Prepend the `top_k` most similar past chunks to every query; when
    /// `false`, recall happens only through `reminisce`.
    pub auto_inject: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            enabled: DEFAULT_EMBEDDING_ENABLED,
            model: DEFAULT_EMBEDDING_MODEL.to_string(),
            server_binary: DEFAULT_EMBEDDING_SERVER_BINARY.into(),
            host: DEFAULT_EMBEDDING_HOST,
            port: DEFAULT_EMBEDDING_PORT,
            gpu_layers: DEFAULT_EMBEDDING_GPU_LAYERS,
            ready_timeout_secs: DEFAULT_EMBEDDING_READY_TIMEOUT_SECS,
            custom_args: CustomArgs::default(),
            top_k: DEFAULT_EMBEDDING_TOP_K,
            auto_inject: DEFAULT_EMBEDDING_AUTO_INJECT,
        }
    }
}
