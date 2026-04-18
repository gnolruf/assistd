//! Runtime configuration handed to the chat client at construction time.
//!
//! Mirrors the `ServerSpec`/`ModelSpec` split used by `llama_server`: the
//! daemon copies values out of the loaded `Config` at startup, so this
//! crate stays free of an `assistd-core` dependency.

#[derive(Debug, Clone)]
pub struct ChatSpec {
    pub host: String,
    pub port: u16,
    pub system_prompt: String,
    pub max_history_tokens: u32,
    pub summary_target_tokens: u32,
    pub preserve_recent_turns: u32,
    pub temperature: f32,
    pub max_response_tokens: u32,
    pub max_summary_tokens: u32,
    pub request_timeout_secs: u64,
    /// The real context window size of the running model, in tokens.
    /// Used to derive a safety-margin budget that the chat client never
    /// lets the request exceed, even if `max_history_tokens` is set high.
    pub model_context_length: u32,
    /// Model identifier sent as the `"model"` field on every
    /// `/v1/chat/completions` request. Must match the name llama-server
    /// has registered for the loaded weights — in router mode that is
    /// the string passed to `POST /models/load` (i.e. `config.model.name`).
    pub model_name: String,
}

impl ChatSpec {
    /// Effective budget for the entire request, after applying a 10% safety
    /// margin against the real model context length. Protects us from the
    /// token heuristic (bytes/4) under-counting relative to the real BPE
    /// tokenizer.
    pub fn effective_context_budget(&self) -> u32 {
        (self.model_context_length as u64 * 9 / 10) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(ctx: u32) -> ChatSpec {
        ChatSpec {
            host: "127.0.0.1".into(),
            port: 8385,
            system_prompt: "sys".into(),
            max_history_tokens: 4000,
            summary_target_tokens: 500,
            preserve_recent_turns: 2,
            temperature: 0.7,
            max_response_tokens: 1024,
            max_summary_tokens: 800,
            request_timeout_secs: 60,
            model_context_length: ctx,
            model_name: "test-model".into(),
        }
    }

    #[test]
    fn effective_budget_applies_ten_percent_margin() {
        assert_eq!(spec(1000).effective_context_budget(), 900);
        assert_eq!(spec(8192).effective_context_budget(), 7372);
    }
}
