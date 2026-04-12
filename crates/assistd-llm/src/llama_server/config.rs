//! Inputs to [`super::LlamaService::start`].
//!
//! These are intentionally separate from `assistd_core::config` — `assistd-core`
//! already depends on `assistd-llm`, so we can't pull its config types in
//! without a cycle. The daemon copies values out of its loaded config into
//! these structs at startup.

/// Fields needed to spawn and health-check the llama-server child.
#[derive(Debug, Clone)]
pub struct ServerSpec {
    pub binary_path: String,
    pub host: String,
    pub port: u16,
    /// GPU layer count passed as `-ngl`.
    pub gpu_layers: u32,
    /// Timeout in seconds for the health check after spawning the child.
    pub ready_timeout_secs: u64,
}

/// Model identity and context length, passed to the llama-server child.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// Model identifier passed to `--hf` (e.g. `"bartowski/Qwen3-14B-GGUF:Q4_K_M"`).
    pub name: String,
    pub context_length: u32,
}
