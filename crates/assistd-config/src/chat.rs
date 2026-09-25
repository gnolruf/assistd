use std::num::{NonZeroU32, NonZeroU64};

use serde::{Deserialize, Serialize};

use crate::defaults::{
    DEFAULT_CHAT_MAX_HISTORY_TOKENS, DEFAULT_CHAT_MAX_RESPONSE_TOKENS,
    DEFAULT_CHAT_PRESERVE_RECENT_TURNS, DEFAULT_CHAT_REQUEST_TIMEOUT_SECS,
    DEFAULT_CHAT_SUMMARY_TARGET_TOKENS, DEFAULT_CHAT_SUMMARY_TEMPERATURE, DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_SYSTEM_PROMPT,
};

/// System prompt, history window and sampling. Sampler fields left `None`
/// are omitted from requests, so llama-server applies its own default.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ChatConfig {
    /// First message of every request; empty disables it. Need not list
    /// tools: requests carry their schemas.
    pub system_prompt: String,
    /// Approximate token budget for system prompt plus history; older
    /// messages are summarized past it. Must be below `model.context_length`.
    pub max_history_tokens: NonZeroU32,
    /// Target summary length in tokens. Must be below `max_history_tokens`.
    pub summary_target_tokens: NonZeroU32,
    /// Recent user/assistant exchanges kept verbatim when summarizing.
    pub preserve_recent_turns: NonZeroU32,
    /// Sampling temperature, `0.0..=2.0`.
    pub temperature: f32,
    /// Max tokens per response. Must be below `model.context_length`.
    pub max_response_tokens: NonZeroU32,
    /// Seconds to the first streamed byte (prompt prefill); generation
    /// itself is unbounded. Also caps the whole summarization call.
    pub request_timeout_secs: NonZeroU64,
    /// Temperature for the summarization call, `0.0..=2.0`.
    pub summary_temperature: f32,
    /// Nucleus sampling cutoff, `0.0..=1.0`.
    pub top_p: Option<f32>,
    pub top_k: Option<NonZeroU32>,
    /// Min-p sampling cutoff, `0.0..=1.0`.
    pub min_p: Option<f32>,
    /// `-2.0..=2.0`.
    pub presence_penalty: Option<f32>,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            max_history_tokens: DEFAULT_CHAT_MAX_HISTORY_TOKENS,
            summary_target_tokens: DEFAULT_CHAT_SUMMARY_TARGET_TOKENS,
            preserve_recent_turns: DEFAULT_CHAT_PRESERVE_RECENT_TURNS,
            temperature: DEFAULT_CHAT_TEMPERATURE,
            max_response_tokens: DEFAULT_CHAT_MAX_RESPONSE_TOKENS,
            request_timeout_secs: DEFAULT_CHAT_REQUEST_TIMEOUT_SECS,
            summary_temperature: DEFAULT_CHAT_SUMMARY_TEMPERATURE,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
        }
    }
}

impl ChatConfig {
    /// `max_tokens` for the summarization call: 20% above the target so the
    /// model can finish its sentence.
    pub fn max_summary_tokens(&self) -> u32 {
        self.summary_target_tokens.get().saturating_mul(6) / 5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defaults::nz32;

    #[test]
    fn max_summary_tokens_leaves_a_fifth_of_headroom_above_target() {
        let cfg = ChatConfig {
            summary_target_tokens: nz32(1000),
            ..ChatConfig::default()
        };
        assert_eq!(cfg.max_summary_tokens(), 1200);
    }
}
