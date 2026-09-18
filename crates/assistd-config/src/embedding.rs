//! Embedding-server and semantic-search configuration.

use std::net::IpAddr;
use std::num::{NonZeroU16, NonZeroU32};

use serde::{Deserialize, Serialize};

use crate::defaults::{
    DEFAULT_EMBEDDING_AUTO_INJECT, DEFAULT_EMBEDDING_ENABLED, DEFAULT_EMBEDDING_GPU_LAYERS,
    DEFAULT_EMBEDDING_HOST, DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_PORT,
    DEFAULT_EMBEDDING_TOP_K,
};

/// `[embedding]` section of `config.toml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct EmbeddingConfig {
    /// Master switch. When `false` semantic recall is unavailable.
    pub enabled: bool,
    /// HuggingFace model id passed verbatim to llama-server's
    /// `--hf-repo` flag. Format: `<owner>/<repo>:<file>`.
    pub model: String,
    /// Bind host for the embedding llama-server. Should be loopback.
    pub host: IpAddr,
    /// TCP port the embedding llama-server binds to. Must differ from
    /// `llama_server.port`.
    pub port: NonZeroU16,
    /// `-ngl` count for the embed server. `0` keeps it on CPU.
    pub gpu_layers: u32,
    /// How many nearest-neighbor matches to retrieve per query for
    /// auto-injection and the `reminisce` tool's default.
    pub top_k: NonZeroU32,
    /// When `true`, every user query embeds the prompt and prepends
    /// the top-K conversation chunks as a "Relevant past context:"
    /// system message. Disable to require explicit `reminisce` tool
    /// invocations instead.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_round_trips_through_toml() {
        let cfg = EmbeddingConfig::default();
        let s = toml::to_string(&cfg).unwrap();
        let back: EmbeddingConfig = toml::from_str(&s).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn omitted_section_uses_defaults() {
        #[derive(Deserialize)]
        struct Wrap {
            #[serde(default)]
            embedding: EmbeddingConfig,
        }
        let parsed: Wrap = toml::from_str("[embedding]").unwrap();
        assert_eq!(parsed.embedding, EmbeddingConfig::default());
    }

    #[test]
    fn defaults_match_expected_values() {
        let cfg = EmbeddingConfig::default();
        assert!(cfg.enabled);
        assert!(!cfg.model.is_empty());
        assert_eq!(cfg.host, DEFAULT_EMBEDDING_HOST);
        assert_eq!(cfg.gpu_layers, 0);
        assert_eq!(cfg.top_k, DEFAULT_EMBEDDING_TOP_K);
        assert!(cfg.auto_inject);
    }
}
