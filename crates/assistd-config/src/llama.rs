use crate::defaults::{
    DEFAULT_GPU_LAYERS, DEFAULT_LLAMA_BINARY, DEFAULT_LLAMA_HOST, DEFAULT_LLAMA_PORT,
    DEFAULT_READY_TIMEOUT_SECS,
};
use serde::{Deserialize, Serialize};

/// llama-server process lifecycle settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlamaServerConfig {
    /// Path to the llama-server binary. Absolute path or a name resolvable via `$PATH`.
    pub binary_path: String,
    /// Host the managed llama-server binds to. Should be a loopback address.
    pub host: String,
    /// TCP port the managed llama-server binds to.
    pub port: u16,
    /// GPU layer count passed as `-ngl`. Default `9999` offloads all layers;
    /// llama.cpp clamps to the model's actual layer count.
    #[serde(default = "default_gpu_layers")]
    pub gpu_layers: u32,
    /// Timeout in seconds for the health check to succeed after spawning
    /// llama-server. First-time HuggingFace downloads may need several minutes.
    #[serde(default = "default_ready_timeout_secs")]
    pub ready_timeout_secs: u64,
}

impl Default for LlamaServerConfig {
    fn default() -> Self {
        Self {
            binary_path: DEFAULT_LLAMA_BINARY.to_string(),
            host: DEFAULT_LLAMA_HOST.to_string(),
            port: DEFAULT_LLAMA_PORT,
            gpu_layers: DEFAULT_GPU_LAYERS,
            ready_timeout_secs: DEFAULT_READY_TIMEOUT_SECS,
        }
    }
}

fn default_gpu_layers() -> u32 {
    DEFAULT_GPU_LAYERS
}

fn default_ready_timeout_secs() -> u64 {
    DEFAULT_READY_TIMEOUT_SECS
}
