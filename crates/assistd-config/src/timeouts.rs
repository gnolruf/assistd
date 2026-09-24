use crate::defaults::{
    DEFAULT_TIMEOUT_DISPATCH_ENVELOPE_SECS, DEFAULT_TIMEOUT_PRESENCE_DROWSE_SECS,
    DEFAULT_TIMEOUT_PRESENCE_SLEEP_SECS, DEFAULT_TIMEOUT_STREAM_INACTIVITY_SECS,
    DEFAULT_TIMEOUT_TOOL_CALL_SECS,
};

/// Hard deadlines for operations that cross a process boundary
/// (HTTP, child stdin/stdout, SSE chunk reads). Every value is a
/// safety valve, not a normal-case latency budget; defaults are
/// generous and a trip should always be logged at `warn!`.
///
/// Not part of the TOML surface: `Config` skips this field. It stays a
/// struct so tests can shorten a deadline instead of waiting out the
/// real one.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeoutsConfig {
    /// Cap on stopping llama-server when entering `Sleeping`. Covers the
    /// SIGTERM grace plus slack for the supervisor to exit. Default: 30s.
    pub presence_sleep_secs: u64,
    /// Cap on the model-unload HTTP call when entering `Drowsy`.
    /// Default: 10s.
    pub presence_drowse_secs: u64,
    /// Outer envelope on handling one IPC request, so a stuck connection
    /// cannot wedge its daemon-side task forever. Stream-level inactivity
    /// timeouts catch the granular case. Default: 600s.
    pub dispatch_envelope_secs: u64,
    /// Inactivity deadline between SSE chunks of a streamed chat
    /// completion, applied once the stream has produced its first byte.
    /// If the model then goes quiet for this long, the call errors out
    /// instead of hanging. The wait for the first byte is prompt prefill
    /// rather than a stall and is bounded by `chat.request_timeout_secs`.
    /// Default: 30s.
    pub stream_inactivity_secs: u64,
    /// Cap on a single tool invocation within an agent turn. A call
    /// that trips it becomes an error result and the turn continues.
    /// Sits above a confirmation prompt plus a tool's own timeout and
    /// below `dispatch_envelope_secs`, so it only catches a tool that
    /// failed to bound itself. Default: 300s.
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
