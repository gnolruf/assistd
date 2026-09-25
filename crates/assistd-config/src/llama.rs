use std::net::IpAddr;
use std::num::{NonZeroU16, NonZeroU32, NonZeroU64};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::defaults::{
    DEFAULT_GPU_LAYERS, DEFAULT_LLAMA_BINARY, DEFAULT_LLAMA_HOST, DEFAULT_LLAMA_PORT,
    DEFAULT_READY_TIMEOUT_SECS,
};

/// Chat llama-server process settings. `None` omits the matching flag.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct LlamaServerConfig {
    /// Binary path, or a name on `$PATH`. Must not be empty.
    pub binary_path: PathBuf,
    /// Bind host. Should be loopback.
    pub host: IpAddr,
    /// Bind port.
    pub port: NonZeroU16,
    /// `-ngl`; values above the model's layer count offload every layer.
    pub gpu_layers: u32,
    /// Last-ditch cap, in seconds, on becoming healthy and loading the model
    /// (chat and embedding servers); neither wait trips while progressing.
    pub ready_timeout_secs: NonZeroU64,
    /// `--alias`: the model name reported in `/v1/models`.
    pub alias: Option<String>,
    /// `-ot` tensor-override regex, e.g. `\.ffn_(up|down|gate)_exps\.=CPU`.
    pub override_tensor: Option<String>,
    /// `--flash-attn on|off`.
    pub flash_attn: Option<bool>,
    /// `--cache-type-k` KV-cache quantization, e.g. `q8_0`.
    pub cache_type_k: Option<String>,
    /// `--cache-type-v` KV-cache quantization, e.g. `q8_0`.
    pub cache_type_v: Option<String>,
    /// `--threads`.
    pub threads: Option<NonZeroU32>,
    /// `--batch-size` (llama-server default 2048).
    pub batch_size: Option<u32>,
    /// `--ubatch-size` (llama-server default 512); larger speeds MoE prefill
    /// for a little VRAM.
    pub ubatch_size: Option<u32>,
    /// `--n-cpu-moe`: layers whose MoE experts stay on CPU, counted from
    /// the first.
    pub n_cpu_moe: Option<u32>,
    /// `--cache-ram`: prompt-checkpoint cache in MiB (llama-server default
    /// 8192).
    pub cache_ram_mib: Option<u32>,
    /// `--mlock`: pin model pages in RAM. Needs a large enough
    /// `RLIMIT_MEMLOCK`.
    pub mlock: Option<bool>,
    /// `--mmproj-offload` / `--no-mmproj-offload`: run the vision encoder on
    /// GPU; `false` keeps it on CPU.
    pub mmproj_offload: Option<bool>,
}

impl Default for LlamaServerConfig {
    fn default() -> Self {
        Self {
            binary_path: DEFAULT_LLAMA_BINARY.into(),
            host: DEFAULT_LLAMA_HOST,
            port: DEFAULT_LLAMA_PORT,
            gpu_layers: DEFAULT_GPU_LAYERS,
            ready_timeout_secs: DEFAULT_READY_TIMEOUT_SECS,
            alias: None,
            override_tensor: None,
            flash_attn: None,
            cache_type_k: None,
            cache_type_v: None,
            threads: None,
            batch_size: None,
            ubatch_size: None,
            n_cpu_moe: None,
            cache_ram_mib: None,
            mlock: None,
            mmproj_offload: None,
        }
    }
}
