use std::num::{NonZeroU32, NonZeroU64};

use serde::{Deserialize, Serialize};

use crate::defaults::{
    DEFAULT_CHAT_MAX_HISTORY_TOKENS, DEFAULT_CHAT_MAX_REASONING_TOKENS,
    DEFAULT_CHAT_MAX_RESPONSE_TOKENS, DEFAULT_CHAT_REQUEST_TIMEOUT_SECS,
    DEFAULT_CHAT_SUMMARY_TARGET_TOKENS, DEFAULT_CHAT_SUMMARY_TEMPERATURE, DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_SYSTEM_PROMPT,
};
use crate::settings::DescribeSettings;

/// System prompt, history window and sampling. Sampler fields left `None`
/// are omitted from requests, so llama-server applies its own default.
#[derive(DescribeSettings, Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ChatConfig {
    /// First message of every request; empty disables it. Need not list
    /// tools: requests carry their schemas.
    #[setting(
        label = "system prompt",
        description = "First message of every request; empty sends none"
    )]
    pub system_prompt: String,
    /// Approximate token budget for system prompt plus history; older
    /// messages are summarized past it. Together with `max_response_tokens`,
    /// must fit in 90% of `model.context_length`.
    #[setting(
        label = "history budget (tokens)",
        description = "Prompt and history budget; older messages are summarized past it",
        advanced
    )]
    pub max_history_tokens: NonZeroU32,
    /// Target summary length in tokens. Must be below `max_history_tokens`.
    #[setting(
        label = "summary length (tokens)",
        description = "Target length of a history summary; below the history budget",
        advanced
    )]
    pub summary_target_tokens: NonZeroU32,
    /// Sampling temperature, `0.0..=2.0`.
    #[setting(label = "temperature", description = "Sampling randomness, 0.0 to 2.0")]
    pub temperature: f32,
    /// Max tokens per model response, reasoning included; a response that
    /// hits it is discarded and retried. See `max_history_tokens` for the
    /// bound.
    #[setting(
        label = "max reply length (tokens)",
        description = "Longest reply, thinking included; longer replies are retried"
    )]
    pub max_response_tokens: NonZeroU32,
    /// Max tokens the model may think for in one response before the server
    /// ends its thinking and it must act. Must be below `max_response_tokens`.
    #[setting(
        label = "max thinking (tokens)",
        description = "Longest the model may think before it must act",
        advanced
    )]
    pub max_reasoning_tokens: NonZeroU32,
    /// Seconds to the first streamed byte (prompt prefill); generation
    /// itself is unbounded. Also caps the whole summarization call.
    #[setting(
        label = "first-token timeout (s)",
        description = "Longest wait for the first streamed byte of a reply",
        advanced
    )]
    pub request_timeout_secs: NonZeroU64,
    /// Temperature for the summarization call, `0.0..=2.0`.
    #[setting(
        label = "summary temperature",
        description = "Sampling randomness for summaries, 0.0 to 2.0",
        advanced
    )]
    pub summary_temperature: f32,
    /// Nucleus sampling cutoff, `0.0..=1.0`.
    #[setting(
        label = "top-p cutoff",
        description = "Nucleus sampling cutoff, 0.0 to 1.0",
        advanced
    )]
    pub top_p: Option<f32>,
    #[setting(
        label = "top-k cutoff",
        description = "Sample only from the k most likely tokens",
        advanced
    )]
    pub top_k: Option<NonZeroU32>,
    /// Min-p sampling cutoff, `0.0..=1.0`.
    #[setting(
        label = "min-p cutoff",
        description = "Min-p sampling cutoff, 0.0 to 1.0",
        advanced
    )]
    pub min_p: Option<f32>,
    /// `-2.0..=2.0`.
    #[setting(
        label = "presence penalty",
        description = "Penalty on tokens already used, -2.0 to 2.0",
        advanced
    )]
    pub presence_penalty: Option<f32>,
    /// `reasoning_effort` chat-template variable sent with every request, e.g.
    /// `low`. Valid values depend on the model's template; `None` keeps the
    /// template's default. Must not be empty.
    #[setting(
        label = "reasoning effort",
        description = "Template's reasoning_effort value, e.g. low; unset keeps its default",
        advanced
    )]
    pub reasoning_effort: Option<String>,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            max_history_tokens: DEFAULT_CHAT_MAX_HISTORY_TOKENS,
            summary_target_tokens: DEFAULT_CHAT_SUMMARY_TARGET_TOKENS,
            temperature: DEFAULT_CHAT_TEMPERATURE,
            max_response_tokens: DEFAULT_CHAT_MAX_RESPONSE_TOKENS,
            max_reasoning_tokens: DEFAULT_CHAT_MAX_REASONING_TOKENS,
            request_timeout_secs: DEFAULT_CHAT_REQUEST_TIMEOUT_SECS,
            summary_temperature: DEFAULT_CHAT_SUMMARY_TEMPERATURE,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            reasoning_effort: None,
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
