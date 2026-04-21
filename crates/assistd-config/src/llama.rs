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
    /// Optional alias passed as `--alias`. Useful when llama-server reports
    /// the model name in `/v1/models`.
    #[serde(default)]
    pub alias: Option<String>,
    /// Optional tensor-override regex passed as `-ot`. Typical MoE offload
    /// pattern: `\.ffn_(up|down|gate)_exps\.=CPU`.
    #[serde(default)]
    pub override_tensor: Option<String>,
    /// Enable flash-attention (`--flash-attn on|off`). `None` omits the flag.
    #[serde(default)]
    pub flash_attn: Option<bool>,
    /// KV-cache K-tensor quantization (`--cache-type-k`), e.g. `q8_0`.
    #[serde(default)]
    pub cache_type_k: Option<String>,
    /// KV-cache V-tensor quantization (`--cache-type-v`), e.g. `q8_0`.
    #[serde(default)]
    pub cache_type_v: Option<String>,
    /// CPU thread count passed as `--threads`. `None` lets llama.cpp decide.
    #[serde(default)]
    pub threads: Option<u32>,
    /// Logical max batch size passed as `--batch-size`. `None` uses
    /// llama-server's default (2048).
    #[serde(default)]
    pub batch_size: Option<u32>,
    /// Physical max batch size passed as `--ubatch-size`. `None` uses
    /// llama-server's default (512). Raising to 1024–2048 speeds prefill
    /// on MoE models at the cost of a small amount of compute-buffer VRAM.
    #[serde(default)]
    pub ubatch_size: Option<u32>,
    /// `--n-cpu-moe N`: keep the MoE expert weights of the first N layers
    /// on CPU; the remainder go to GPU. Preferred over a blanket
    /// `override_tensor` regex because it's layer-granular. Tune down until
    /// VRAM is nearly full.
    #[serde(default)]
    pub n_cpu_moe: Option<u32>,
    /// `--cache-ram N` prompt-checkpoint cache size in MiB. Default in
    /// llama-server is 8192; raising this speeds up re-prefill on long
    /// conversations at the cost of system RAM.
    #[serde(default)]
    pub cache_ram_mib: Option<u32>,
    /// `--mlock`: pin loaded model pages in RAM so the OS can't page them
    /// out. Guarantees no surprise page-faults mid-inference.
    #[serde(default)]
    pub mlock: Option<bool>,
    /// `--mmproj-offload` / `--no-mmproj-offload`: whether the multimodal
    /// projector (vision encoder) runs on GPU. Set `false` when VRAM is
    /// fully consumed by the LLM — image preprocessing falls back to CPU.
    #[serde(default)]
    pub mmproj_offload: Option<bool>,
}

impl Default for LlamaServerConfig {
    fn default() -> Self {
        Self {
            binary_path: DEFAULT_LLAMA_BINARY.to_string(),
            host: DEFAULT_LLAMA_HOST.to_string(),
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

fn default_gpu_layers() -> u32 {
    DEFAULT_GPU_LAYERS
}

fn default_ready_timeout_secs() -> u64 {
    DEFAULT_READY_TIMEOUT_SECS
}
