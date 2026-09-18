use std::num::NonZeroU32;

use crate::defaults::{DEFAULT_MODEL_CONTEXT_LENGTH, DEFAULT_MODEL_NAME};
use serde::{Deserialize, Serialize};

/// Local model settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    /// Model identifier passed to llama-server's `--hf` flag.
    /// Format: `owner/repo:quant` (e.g. `"unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M"`).
    pub name: String,
    /// Context window length in tokens.
    pub context_length: NonZeroU32,
}

impl ModelConfig {
    /// Context length less a 10% margin, since the bytes/4 token
    /// heuristic under-counts relative to the real tokenizer.
    pub fn context_budget(&self) -> u32 {
        (u64::from(self.context_length.get()) * 9 / 10) as u32
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: DEFAULT_MODEL_NAME.to_string(),
            context_length: DEFAULT_MODEL_CONTEXT_LENGTH,
        }
    }
}
