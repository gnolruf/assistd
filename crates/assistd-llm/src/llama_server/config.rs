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
    /// If set, passed to `-ngl` verbatim, bypassing GGUF parsing.
    pub gpu_layers: Option<u32>,
}

/// Fields needed to compute `-ngl` and pass the model path/context length on.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub path: String,
    pub vram_budget_mb: u64,
    pub context_length: u32,
}
