//! Embedding-server and semantic-search configuration.

use std::fmt;
use std::num::{NonZeroU16, NonZeroU64};
use std::str::FromStr;

use assistd_utils::hf::HfFileId;
use serde::{Deserialize, Serialize};

use crate::custom_args::{CustomArgs, EmbeddingServer};
use crate::defaults::{
    DEFAULT_EMBEDDING_AUTO_INJECT, DEFAULT_EMBEDDING_ENABLED, DEFAULT_EMBEDDING_GPU_LAYERS,
    DEFAULT_EMBEDDING_HOST, DEFAULT_EMBEDDING_MIN_SIMILARITY, DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PORT, DEFAULT_EMBEDDING_READY_TIMEOUT_SECS, DEFAULT_EMBEDDING_SERVER_BINARY,
    DEFAULT_EMBEDDING_TOP_K,
};
use crate::errors::EmbeddingModelError;
use crate::settings::{DescribeSettings, describe_fields};
use crate::values::{Bounded, HfFile, LoopbackAddr, NonBlankString};
use crate::voice::UnitInterval;

/// Dedicated embedding server and semantic-recall settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct EmbeddingConfig {
    /// When `false`, semantic recall is unavailable.
    pub enabled: bool,
    /// `--hf-repo` value, `<owner>/<repo>:<quant>`. The suffix must be a
    /// quant tag; a `.gguf` filename fails with "no GGUF files found".
    pub model: EmbeddingModel,
    /// Server binary path, or a name on `$PATH`.
    pub server_binary: NonBlankString,
    /// Bind host; loopback because the per-launch API key travels over
    /// plain HTTP.
    pub host: LoopbackAddr,
    /// Bind port. Must differ from `model.port` when enabled.
    pub port: NonZeroU16,
    /// `-ngl` count. `0` keeps it on CPU.
    pub gpu_layers: u32,
    /// Last-ditch cap, in seconds, on becoming healthy; the wait never trips
    /// while the model is still downloading or loading.
    pub ready_timeout_secs: NonZeroU64,
    /// Extra server arguments. Flags assistd sets itself, or that expose
    /// files, tools or state through the server, fail to parse. Any change
    /// drops the stored vectors, which are then embedded again.
    pub custom_args: CustomArgs<EmbeddingServer>,
    /// Nearest-neighbour matches per search: the most saved memories and
    /// past-chat matches auto-injected, and the `reminisce` default. `1..=20`.
    pub top_k: Bounded<1, 20>,
    /// Prepend matching saved memories, and one pointer per matching past chat, to a
    /// new session's first query of three or more characters; resumed sessions get
    /// none. When `false`, recall is tools-only. Off while `memory.enabled` is `false`.
    pub auto_inject: bool,
    /// Cosine similarity, `0.0..=1.0`, an auto-injected memory or past-chat
    /// match must reach; a query with no match above it gets no recalled
    /// context. `0` injects every match that is not opposed to the query.
    pub min_similarity: UnitInterval,
}

impl DescribeSettings for EmbeddingConfig {
    describe_fields! {
        enabled => ("enabled", "Enables semantic recall of past chats"),
        model => ("embedding model", "Embedding model, as owner/repo:quant"),
        server_binary => ("llama-server binary", "llama-server path, or a name on $PATH", advanced),
        host => (
            "server address",
            "Address the embedding server binds; must be loopback",
            advanced
        ),
        port => (
            "server port",
            "Port the embedding server binds; must differ from model.port",
            advanced
        ),
        gpu_layers => (
            "layers offloaded to GPU",
            "Layers offloaded to the GPU; 0 keeps it on CPU",
            advanced
        ),
        ready_timeout_secs => (
            "startup timeout (s)",
            "Longest wait for the embedding server to start",
            advanced
        ),
        custom_args => (
            "extra server arguments",
            "Extra llama-server arguments; flags assistd sets are refused",
            advanced
        ),
        top_k => ("memories recalled per query", "Matches recalled per query, 1 to 20"),
        auto_inject => (
            "add recalled memories to queries",
            "Prepend matching saved memories and pointers to past chats to a new chat's first query"
        ),
        min_similarity => (
            "recall similarity floor",
            "How close a match must be to be recalled, 0 to 1",
            advanced
        ),
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            enabled: DEFAULT_EMBEDDING_ENABLED,
            model: DEFAULT_EMBEDDING_MODEL
                .parse()
                .expect("the default embedding model parses"),
            server_binary: DEFAULT_EMBEDDING_SERVER_BINARY
                .parse()
                .expect("the default server binary is not blank"),
            host: DEFAULT_EMBEDDING_HOST,
            port: DEFAULT_EMBEDDING_PORT,
            gpu_layers: DEFAULT_EMBEDDING_GPU_LAYERS,
            ready_timeout_secs: DEFAULT_EMBEDDING_READY_TIMEOUT_SECS,
            custom_args: CustomArgs::default(),
            top_k: DEFAULT_EMBEDDING_TOP_K,
            auto_inject: DEFAULT_EMBEDDING_AUTO_INJECT,
            min_similarity: DEFAULT_EMBEDDING_MIN_SIMILARITY,
        }
    }
}

/// The embedding model as `<owner>/<repo>:<quant>`, with no `@<revision>` pin.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct EmbeddingModel(HfFileId);

impl fmt::Display for EmbeddingModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl FromStr for EmbeddingModel {
    type Err = EmbeddingModelError;

    fn from_str(model: &str) -> Result<Self, Self::Err> {
        let file: HfFile = model.parse()?;
        if file.id().revision().is_some() {
            Err(EmbeddingModelError::Pinned(model.to_owned()))
        } else {
            Ok(Self(file.id().clone()))
        }
    }
}

impl TryFrom<String> for EmbeddingModel {
    type Error = EmbeddingModelError;

    fn try_from(model: String) -> Result<Self, Self::Error> {
        model.parse()
    }
}

impl From<EmbeddingModel> for String {
    fn from(model: EmbeddingModel) -> Self {
        model.to_string()
    }
}
