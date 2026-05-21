use crate::defaults::{
    DEFAULT_TIMEOUT_DISPATCH_ENVELOPE_SECS, DEFAULT_TIMEOUT_PRESENCE_DROWSE_SECS,
    DEFAULT_TIMEOUT_PRESENCE_SLEEP_SECS, DEFAULT_TIMEOUT_STREAM_INACTIVITY_SECS,
};
use serde::{Deserialize, Serialize};

/// Hard deadlines for operations that cross a process boundary
/// (HTTP, child stdin/stdout, SSE chunk reads). Every value is a
/// safety valve, not a normal-case latency budget; defaults are
/// generous and a trip should always be logged at `warn!`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TimeoutsConfig {
    /// Cap on `PresenceManager::sleep`'s `service.shutdown()` call.
    /// Covers the SIGTERM grace plus slack for the supervisor to
    /// observe the watch flip and exit. Default: 30s.
    #[serde(default = "default_presence_sleep_secs")]
    pub presence_sleep_secs: u64,
    /// Cap on `PresenceManager::drowse`'s `control.unload_model()`
    /// HTTP call. Default: 10s.
    #[serde(default = "default_presence_drowse_secs")]
    pub presence_drowse_secs: u64,
    /// Outer envelope on `AppState::dispatch`. Strictly a safety valve
    /// so a stuck connection cannot wedge a daemon-side connection
    /// task forever. Stream-level inactivity timeouts catch the
    /// granular case. Default: 600s (10 min).
    #[serde(default = "default_dispatch_envelope_secs")]
    pub dispatch_envelope_secs: u64,
    /// Per-chunk inactivity deadline on the SSE byte read in
    /// `LlamaChatClient::stream_openai`. If the model emits no bytes
    /// for this long, the call errors out instead of hanging. Default: 30s.
    #[serde(default = "default_stream_inactivity_secs")]
    pub stream_inactivity_secs: u64,
}

impl Default for TimeoutsConfig {
    fn default() -> Self {
        Self {
            presence_sleep_secs: default_presence_sleep_secs(),
            presence_drowse_secs: default_presence_drowse_secs(),
            dispatch_envelope_secs: default_dispatch_envelope_secs(),
            stream_inactivity_secs: default_stream_inactivity_secs(),
        }
    }
}

fn default_presence_sleep_secs() -> u64 {
    DEFAULT_TIMEOUT_PRESENCE_SLEEP_SECS
}

fn default_presence_drowse_secs() -> u64 {
    DEFAULT_TIMEOUT_PRESENCE_DROWSE_SECS
}

fn default_dispatch_envelope_secs() -> u64 {
    DEFAULT_TIMEOUT_DISPATCH_ENVELOPE_SECS
}

fn default_stream_inactivity_secs() -> u64 {
    DEFAULT_TIMEOUT_STREAM_INACTIVITY_SECS
}
