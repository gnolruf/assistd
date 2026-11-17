//! Chat model and llama-server process configuration.

use std::net::{Ipv4Addr, SocketAddr};
use std::num::{NonZeroU16, NonZeroU32, NonZeroU64};

use serde::{Deserialize, Serialize};

use crate::custom_args::{CustomArgs, ModelServer};
use crate::defaults::{nz16, nz32, nz64};
use crate::settings::{DescribeSettings, describe_fields};
use crate::values::NonBlankString;

/// The agent's model and the local server process that serves it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ModelConfig {
    /// HuggingFace model the server loads, as `owner/repo:quant`.
    pub name: NonBlankString,
    /// Context window length in tokens.
    pub context_length: NonZeroU32,
    /// Server binary path, or a name on `$PATH`; the embedding server runs
    /// it too.
    pub server_binary: NonBlankString,
    /// Port the server binds on `127.0.0.1`; loopback because the per-launch
    /// API key travels over plain HTTP.
    pub port: NonZeroU16,
    /// Layers offloaded to the GPU; values above the model's layer count
    /// offload every layer.
    pub gpu_layers: u32,
    /// Last-ditch cap, in seconds, on becoming healthy and loading the model;
    /// neither wait trips while progressing.
    pub ready_timeout_secs: NonZeroU64,
    /// Extra server arguments. Flags assistd sets itself, the slot save
    /// directory included, or that expose files, tools or state through the
    /// server, fail to parse.
    pub custom_args: CustomArgs<ModelServer>,
}

impl DescribeSettings for ModelConfig {
    describe_fields! {
        name => ("model", "Hugging Face model to load, as owner/repo:quant"),
        context_length => (
            "context window (tokens)",
            "Tokens the model can see at once, prompt and reply together"
        ),
        server_binary => ("llama-server binary", "llama-server path, or a name on $PATH", advanced),
        port => ("server port", "Loopback port the model server binds", advanced),
        gpu_layers => (
            "layers offloaded to GPU",
            "Layers offloaded to the GPU; above the model's count offloads all",
            advanced
        ),
        ready_timeout_secs => (
            "startup timeout (s)",
            "Longest wait for the model server to start and load",
            advanced
        ),
        custom_args => (
            "extra server arguments",
            "Extra llama-server arguments; flags assistd sets are refused",
            advanced
        ),
    }
}

impl ModelConfig {
    /// Context length less a 10% margin for the bytes/4 token estimate's
    /// under-counting.
    pub fn context_budget(&self) -> u32 {
        u32::try_from(u64::from(self.context_length.get()) * 9 / 10).unwrap_or(u32::MAX)
    }

    /// Where the server listens: `port` on `127.0.0.1`.
    pub fn listen_addr(&self) -> SocketAddr {
        SocketAddr::from((Ipv4Addr::LOCALHOST, self.port.get()))
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M"
                .parse()
                .expect("the default model name is not blank"),
            context_length: nz32(8192),
            server_binary: "llama-server"
                .parse()
                .expect("the default server binary is not blank"),
            port: nz16(8385),
            gpu_layers: 9999,
            ready_timeout_secs: nz64(300),
            custom_args: CustomArgs::default(),
        }
    }
}
