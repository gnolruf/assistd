//! Embedding-server and semantic-search configuration.
//!
//! `[embedding]` is a top-level config section parallel to
//! `[llama_server]`. Embeddings are produced by a separate, dedicated
//! llama-server instance held resident on its own port. Keeping it
//! independent of the chat server means semantic retrieval still works
//! when the chat router is `Drowsy` or `Sleeping`. The embed model is
//! intentionally small enough to stay on CPU (`gpu_layers = 0`) so it
//! never contends with the chat model for VRAM.

use serde::{Deserialize, Serialize};

use crate::defaults::{
    DEFAULT_EMBEDDING_AUTO_INJECT, DEFAULT_EMBEDDING_CHUNK_CHARS, DEFAULT_EMBEDDING_CHUNK_OVERLAP,
    DEFAULT_EMBEDDING_ENABLED, DEFAULT_EMBEDDING_GPU_LAYERS, DEFAULT_EMBEDDING_HOST,
    DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_PORT, DEFAULT_EMBEDDING_READY_TIMEOUT_SECS,
    DEFAULT_EMBEDDING_REQUEST_TIMEOUT_SECS, DEFAULT_EMBEDDING_TOP_K,
};

/// `[embedding]` section of `config.toml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct EmbeddingConfig {
    /// Master switch. When `false` the daemon binds the `NoEmbedder`
    /// placeholder; the `recall` and `reminisce` tools both fall back
    /// to "(no memories)" / "(no past conversations indexed)".
    pub enabled: bool,
    /// HuggingFace model id passed verbatim to llama-server's
    /// `--hf-repo` flag. Format: `<owner>/<repo>:<file>`.
    pub model: String,
    /// Bind host for the embedding llama-server. Should be loopback.
    pub host: String,
    /// TCP port the embedding llama-server binds to. Must differ from
    /// `llama_server.port` and `remote.port`.
    pub port: u16,
    /// `-ngl` count for the embed server. Defaults to `0` (CPU only) so
    /// the embed model never competes with the chat model for VRAM.
    /// Small embedding models (~30-300MB Q4) are CPU-fast.
    pub gpu_layers: u32,
    /// Seconds to wait for the embed server's `/health` to come up
    /// after spawn. First-time HuggingFace downloads may need minutes.
    pub ready_timeout_secs: u64,
    /// Per-request HTTP timeout against `/v1/embeddings`. Embedding a
    /// short text (~few hundred tokens) on CPU completes in <1s for
    /// small models, but the first call may include cache warm-up.
    pub request_timeout_secs: u64,
    /// How many nearest-neighbor matches to retrieve per query for
    /// auto-injection and the `reminisce` tool's default.
    pub top_k: u32,
    /// Char-window size for chunking conversation messages. Messages
    /// shorter than this become a single chunk; longer ones are split
    /// into overlapping windows.
    pub chunk_chars: usize,
    /// Char-overlap between consecutive chunks so semantically
    /// coherent runs aren't cut at chunk boundaries. Must be strictly
    /// less than `chunk_chars`.
    pub chunk_overlap_chars: usize,
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
            host: DEFAULT_EMBEDDING_HOST.to_string(),
            port: DEFAULT_EMBEDDING_PORT,
            gpu_layers: DEFAULT_EMBEDDING_GPU_LAYERS,
            ready_timeout_secs: DEFAULT_EMBEDDING_READY_TIMEOUT_SECS,
            request_timeout_secs: DEFAULT_EMBEDDING_REQUEST_TIMEOUT_SECS,
            top_k: DEFAULT_EMBEDDING_TOP_K,
            chunk_chars: DEFAULT_EMBEDDING_CHUNK_CHARS,
            chunk_overlap_chars: DEFAULT_EMBEDDING_CHUNK_OVERLAP,
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
        assert_eq!(cfg.host, "127.0.0.1");
        assert_eq!(cfg.gpu_layers, 0);
        assert!(cfg.top_k > 0);
        assert!(cfg.chunk_chars > cfg.chunk_overlap_chars);
        assert!(cfg.auto_inject);
    }
}
