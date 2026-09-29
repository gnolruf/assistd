use std::num::NonZeroU32;

use serde::{Deserialize, Serialize};

use crate::defaults::{DEFAULT_MODEL_CONTEXT_LENGTH, DEFAULT_MODEL_NAME};

/// Local model settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ModelConfig {
    /// HuggingFace model llama-server loads, as `owner/repo:quant`. Must
    /// not be empty.
    pub name: String,
    /// Context window length in tokens.
    pub context_length: NonZeroU32,
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
        }
    }
}
