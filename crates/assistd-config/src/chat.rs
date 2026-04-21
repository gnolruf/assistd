use crate::defaults::{
    DEFAULT_CHAT_MAX_HISTORY_TOKENS, DEFAULT_CHAT_MAX_RESPONSE_TOKENS,
    DEFAULT_CHAT_MAX_SUMMARY_TOKENS, DEFAULT_CHAT_PRESERVE_RECENT_TURNS,
    DEFAULT_CHAT_REQUEST_TIMEOUT_SECS, DEFAULT_CHAT_SUMMARY_TARGET_TOKENS,
    DEFAULT_CHAT_SUMMARY_TEMPERATURE, DEFAULT_CHAT_TEMPERATURE, DEFAULT_SYSTEM_PROMPT,
};
use crate::model::ModelConfig;
use serde::{Deserialize, Serialize};

/// Chat backend behaviour: system prompt, history window, sampling.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatConfig {
    /// System prompt injected as the first message of every request.
    /// An empty string disables system-prompt injection.
    pub system_prompt: String,
    /// Approximate token budget for the full request messages (system + history).
    /// Must be strictly less than `model.context_length`. When exceeded, older
    /// messages are summarized and replaced by a single summary message.
    pub max_history_tokens: u32,
    /// Approximate target length of a history summary, in tokens.
    pub summary_target_tokens: u32,
    /// Number of recent user/assistant exchanges to preserve verbatim when
    /// summarizing. Must be at least 1.
    pub preserve_recent_turns: u32,
    /// Sampling temperature in the range `0.0..=2.0`.
    pub temperature: f32,
    /// Maximum tokens the model may emit in a single streamed response.
    pub max_response_tokens: u32,
    /// Maximum tokens for the summarization (non-streaming) call. Typically a
    /// bit above `summary_target_tokens`.
    pub max_summary_tokens: u32,
    /// HTTP request timeout for a single chat call, in seconds.
    pub request_timeout_secs: u64,
    /// Sampling temperature for the (non-streaming) summarization call.
    /// Lower than `temperature` by default because summaries should be
    /// deterministic. Range: `0.0..=2.0`.
    #[serde(default = "default_summary_temperature")]
    pub summary_temperature: f32,
    /// Nucleus sampling cutoff in `0.0..=1.0`. `None` omits the field from the
    /// request, letting llama-server apply its own default.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Top-k sampling limit. `None` omits the field.
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Min-p sampling cutoff in `0.0..=1.0`. `None` omits the field.
    #[serde(default)]
    pub min_p: Option<f32>,
    /// Presence penalty in `-2.0..=2.0`. `None` omits the field. Qwen3 reasoning
    /// variants recommend a small positive value (e.g. 1.5) to reduce repetition.
    #[serde(default)]
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
            max_summary_tokens: DEFAULT_CHAT_MAX_SUMMARY_TOKENS,
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
    /// Effective budget for the entire request, after applying a 10% safety
    /// margin against the real model context length. Protects us from the
    /// token heuristic (bytes/4) under-counting relative to the real BPE
    /// tokenizer.
    pub fn effective_context_budget(&self, model: &ModelConfig) -> u32 {
        (model.context_length as u64 * 9 / 10) as u32
    }
}

fn default_summary_temperature() -> f32 {
    DEFAULT_CHAT_SUMMARY_TEMPERATURE
}
