use crate::defaults::{
    DEFAULT_TIMEOUT_DISPATCH_ENVELOPE_SECS, DEFAULT_TIMEOUT_PRESENCE_DROWSE_SECS,
    DEFAULT_TIMEOUT_PRESENCE_SLEEP_SECS, DEFAULT_TIMEOUT_STREAM_INACTIVITY_SECS,
    DEFAULT_TIMEOUT_TOOL_CALL_SECS,
};

/// Safety-valve deadlines, in seconds, on operations that cross a process
/// boundary. Not read from TOML; a struct so tests can shorten them.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeoutsConfig {
    /// Stopping llama-server on entering `Sleeping`, SIGTERM grace included.
    pub presence_sleep_secs: u64,
    /// The model-unload HTTP call on entering `Drowsy`.
    pub presence_drowse_secs: u64,
    /// Handling one IPC request end to end.
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
            presence_sleep_secs: DEFAULT_TIMEOUT_PRESENCE_SLEEP_SECS,
            presence_drowse_secs: DEFAULT_TIMEOUT_PRESENCE_DROWSE_SECS,
            dispatch_envelope_secs: DEFAULT_TIMEOUT_DISPATCH_ENVELOPE_SECS,
            stream_inactivity_secs: DEFAULT_TIMEOUT_STREAM_INACTIVITY_SECS,
            tool_call_secs: DEFAULT_TIMEOUT_TOOL_CALL_SECS,
        }
    }
}
