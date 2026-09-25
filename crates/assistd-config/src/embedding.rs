//! Embedding-server and semantic-search configuration.

use std::net::IpAddr;
use std::num::{NonZeroU16, NonZeroU32};

use serde::{Deserialize, Serialize};

use crate::defaults::{
    DEFAULT_EMBEDDING_AUTO_INJECT, DEFAULT_EMBEDDING_ENABLED, DEFAULT_EMBEDDING_GPU_LAYERS,
    DEFAULT_EMBEDDING_HOST, DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_PORT,
    DEFAULT_EMBEDDING_TOP_K,
};

/// Dedicated embedding llama-server and semantic-recall settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct EmbeddingConfig {
    /// When `false`, semantic recall is unavailable.
    pub enabled: bool,
    /// `--hf-repo` value, `<owner>/<repo>:<quant>`. The suffix must be a
    /// quant tag; a `.gguf` filename fails with "no GGUF files found".
    pub model: String,
    /// Bind host. Should be loopback.
    pub host: IpAddr,
    /// Bind port. Must differ from `llama_server.port`.
    pub port: NonZeroU16,
    /// `-ngl` count. `0` keeps it on CPU.
    pub gpu_layers: u32,
    /// Nearest-neighbour matches per query, for auto-injection and as the
    /// `reminisce` default.
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
            host: DEFAULT_EMBEDDING_HOST,
            port: DEFAULT_EMBEDDING_PORT,
            gpu_layers: DEFAULT_EMBEDDING_GPU_LAYERS,
            top_k: DEFAULT_EMBEDDING_TOP_K,
            auto_inject: DEFAULT_EMBEDDING_AUTO_INJECT,
        }
    }
}
