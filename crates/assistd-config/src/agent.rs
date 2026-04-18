use crate::defaults::DEFAULT_AGENT_MAX_ITERATIONS;
use serde::{Deserialize, Serialize};

/// Agent-loop configuration. The loop sends the LLM a single-tool
/// (`run`) schema and iterates through tool calls until the model
/// produces plain text or the cap is hit.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentConfig {
    /// Maximum number of LLM invocations per user turn. Caps runaway
    /// loops where the model keeps calling tools without resolving the
    /// user's query. Default 20 — higher than a multi-tool agent because
    /// each `run` call is cheap and composable (pipes replace what
    /// would be multiple tool calls in other systems).
    #[serde(default = "default_agent_max_iterations")]
    pub max_iterations: u32,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: default_agent_max_iterations(),
        }
    }
}

fn default_agent_max_iterations() -> u32 {
    DEFAULT_AGENT_MAX_ITERATIONS
}
