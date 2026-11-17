//! Deadlines on operations that cross a process boundary.

use crate::defaults::CONFIRMATION_PROMPT_SECS;

/// Safety-valve deadlines, in seconds, on operations that cross a process
/// boundary. Not read from TOML; a struct so tests can shorten them.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeoutsConfig {
    /// Stopping llama-server on entering `Sleeping`, SIGTERM grace included.
    pub presence_sleep_secs: u64,
    /// The model-unload HTTP call on entering `Drowsy`.
    pub presence_drowse_secs: u64,
    /// Handling one IPC request end to end. Agent turns (`query`,
    /// `ptt_stop`) are exempt: each of their steps is bounded by
    /// `tool_call_secs` and `stream_inactivity_secs` instead.
    pub dispatch_envelope_secs: u64,
    /// Gap between SSE chunks once a chat stream has started; the first
    /// byte is bounded by `chat.request_timeout_secs` instead.
    pub stream_inactivity_secs: u64,
    /// One tool call; a trip becomes an error result and the turn goes on.
    /// Sits above a confirmation prompt plus a tool's own timeout.
    pub tool_call_secs: u64,
}

impl Default for TimeoutsConfig {
    fn default() -> Self {
        Self {
            presence_sleep_secs: 30,
            presence_drowse_secs: 10,
            dispatch_envelope_secs: 600,
            stream_inactivity_secs: 30,
            tool_call_secs: 300,
        }
    }
}

impl TimeoutsConfig {
    /// The longest timeout a tool may set for itself, in seconds, and still
    /// finish under `tool_call_secs` after a confirmation prompt ran out.
    pub fn max_tool_timeout_secs(&self) -> u64 {
        self.tool_call_secs.saturating_sub(CONFIRMATION_PROMPT_SECS)
    }
}
