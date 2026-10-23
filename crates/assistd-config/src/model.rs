use std::net::IpAddr;
use std::num::{NonZeroU16, NonZeroU32, NonZeroU64};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::custom_args::{ChatServer, CustomArgs};
use crate::defaults::{
    DEFAULT_MODEL_CONTEXT_LENGTH, DEFAULT_MODEL_GPU_LAYERS, DEFAULT_MODEL_HOST, DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PORT, DEFAULT_MODEL_READY_TIMEOUT_SECS, DEFAULT_MODEL_SERVER_BINARY,
};
use crate::settings::DescribeSettings;

/// The chat model and the local server process that serves it.
#[derive(DescribeSettings, Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ModelConfig {
    /// HuggingFace model the server loads, as `owner/repo:quant`. Must
    /// not be empty.
    #[setting(
        label = "model",
        description = "Hugging Face model to load, as owner/repo:quant"
    )]
    pub name: String,
    /// Context window length in tokens.
    #[setting(
        label = "context window (tokens)",
        description = "Tokens the model can see at once, prompt and reply together"
    )]
    pub context_length: NonZeroU32,
    /// Server binary path, or a name on `$PATH`. Must not be empty.
    #[setting(
        label = "llama-server binary",
        description = "llama-server path, or a name on $PATH",
        advanced
    )]
    pub server_binary: PathBuf,
    /// Bind host. Must be loopback: the server has no authentication.
    #[setting(
        label = "server address",
        description = "Address the model server binds; must be loopback",
        advanced
    )]
    pub host: IpAddr,
    /// Bind port.
    #[setting(
        label = "server port",
        description = "Port the model server binds",
        advanced
    )]
    pub port: NonZeroU16,
    /// Layers offloaded to the GPU; values above the model's layer count
    /// offload every layer.
    #[setting(
        label = "layers offloaded to GPU",
        description = "Layers offloaded to the GPU; above the model's count offloads all",
        advanced
    )]
    pub gpu_layers: u32,
    /// Last-ditch cap, in seconds, on becoming healthy and loading the model;
    /// neither wait trips while progressing.
    #[setting(
        label = "startup timeout (s)",
        description = "Longest wait for the model server to start and load",
        advanced
    )]
    pub ready_timeout_secs: NonZeroU64,
    /// Extra server arguments. Flags assistd sets itself, or that expose
    /// files, tools or state through the server, fail to parse.
    #[setting(
        label = "extra server arguments",
        description = "Extra llama-server arguments; flags assistd sets are refused",
        advanced
    )]
    pub custom_args: CustomArgs<ChatServer>,
}

impl ModelConfig {
    /// Context length less a 10% margin for the bytes/4 token estimate's
    /// under-counting.
    pub fn context_budget(&self) -> u32 {
        u32::try_from(u64::from(self.context_length.get()) * 9 / 10).unwrap_or(u32::MAX)
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: DEFAULT_MODEL_NAME.to_string(),
            context_length: DEFAULT_MODEL_CONTEXT_LENGTH,
            server_binary: DEFAULT_MODEL_SERVER_BINARY.into(),
            host: DEFAULT_MODEL_HOST,
            port: DEFAULT_MODEL_PORT,
            gpu_layers: DEFAULT_MODEL_GPU_LAYERS,
            ready_timeout_secs: DEFAULT_MODEL_READY_TIMEOUT_SECS,
            custom_args: CustomArgs::default(),
        }
    }
}
