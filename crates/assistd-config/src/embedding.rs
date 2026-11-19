//! Embedding-server and semantic-search configuration.

use std::fmt;
use std::net::{Ipv4Addr, SocketAddr};
use std::num::{NonZeroU16, NonZeroU64};
use std::str::FromStr;

use assistd_utils::hf::HfFileId;
use serde::{Deserialize, Serialize};

use crate::custom_args::{CustomArgs, EmbeddingServer};
use crate::defaults::{bounded, nz16, nz64, unit};
use crate::errors::EmbeddingModelError;
use crate::settings::{DescribeSettings, describe_fields};
use crate::values::{Bounded, HfFile};
use crate::voice::UnitInterval;

/// Semantic-recall settings and the embedding server behind them, which runs
/// `model.server_binary`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct EmbeddingConfig {
    /// When `false`, semantic recall is unavailable.
    pub enabled: bool,
    /// `--hf-repo` value, `<owner>/<repo>:<quant>`. The suffix must be a
    /// quant tag; a `.gguf` filename fails with "no GGUF files found". The two
    /// prefixes below belong to the model and change with it.
    pub model: EmbeddingModel,
    /// Text sent before each stored text, exactly as the model's card gives it,
    /// trailing space included; empty for a model trained without one. Any
    /// change drops the stored vectors, which are then embedded again.
    pub document_prefix: String,
    /// Text sent before each search query, exactly as the model's card gives
    /// it; empty for a model trained without one.
    pub query_prefix: String,
    /// Port the server binds on `127.0.0.1`. Must differ from `model.port`
    /// when enabled.
    pub port: NonZeroU16,
    /// `-ngl` count. `0` keeps it on CPU, where it never contends with the
    /// agent's model for VRAM.
    pub gpu_layers: u32,
    /// Last-ditch cap, in seconds, on becoming healthy; the wait never trips
    /// while the model is still downloading or loading.
    pub ready_timeout_secs: NonZeroU64,
    /// Extra server arguments. Flags assistd sets itself, or that expose
    /// files, tools or state through the server, fail to parse. Any change
    /// drops the stored vectors, which are then embedded again.
    pub custom_args: CustomArgs<EmbeddingServer>,
    /// Nearest-neighbour matches per search: the most saved memories and past-chat
    /// matches auto-injected, and the `reminisce` and `docs search` default. `1..=20`.
    pub top_k: Bounded<1, 20>,
    /// Prepend matching saved memories, and one pointer per matching past chat, to a
    /// new session's first query of three or more characters; resumed sessions get
    /// none. When `false`, recall is tools-only. Off while `memory.enabled` is `false`.
    pub auto_inject: bool,
    /// Cosine similarity, `0.0..=1.0`, an auto-injected match must reach; `0`
    /// injects every match not opposed to the query. Each model and prefix
    /// pair scores on a scale of its own, so the floor changes with them.
    pub min_similarity: UnitInterval,
}

impl DescribeSettings for EmbeddingConfig {
    describe_fields! {
        enabled => ("enabled", "Enables semantic recall of past chats"),
        model => ("embedding model", "Embedding model, as owner/repo:quant"),
        document_prefix => (
            "stored-text prefix",
            "Text the model expects before each stored text; empty for none",
            advanced
        ),
        query_prefix => (
            "query prefix",
            "Text the model expects before each search query; empty for none",
            advanced
        ),
        port => (
            "server port",
            "Loopback port the embedding server binds; must differ from model.port",
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
            enabled: true,
            model: "ggml-org/embeddinggemma-2-GGUF:Q8_0"
                .parse()
                .expect("the default embedding model parses"),
            document_prefix: "title: none | text: ".to_owned(),
            query_prefix: "task: search result | query: ".to_owned(),
            port: nz16(8386),
            gpu_layers: 0,
            ready_timeout_secs: nz64(300),
            custom_args: CustomArgs::default(),
            top_k: bounded(5),
            auto_inject: true,
            min_similarity: unit(0.70),
        }
    }
}

impl EmbeddingConfig {
    /// Where the server listens: `port` on `127.0.0.1`.
    pub fn listen_addr(&self) -> SocketAddr {
        SocketAddr::from((Ipv4Addr::LOCALHOST, self.port.get()))
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
