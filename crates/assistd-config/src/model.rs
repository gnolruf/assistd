use crate::defaults::{DEFAULT_MODEL_CONTEXT_LENGTH, DEFAULT_MODEL_NAME};
use serde::{Deserialize, Serialize};

/// Local model settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    /// Model identifier passed to llama-server's `--hf` flag.
    /// Format: `owner/repo:quant` (e.g. `"bartowski/Qwen3-14B-GGUF:Q4_K_M"`).
    pub name: String,
    /// Context window length in tokens.
    pub context_length: u32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: DEFAULT_MODEL_NAME.to_string(),
            context_length: DEFAULT_MODEL_CONTEXT_LENGTH,
        }
    }
}
