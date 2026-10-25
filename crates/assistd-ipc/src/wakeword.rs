//! Wire types for wake-phrase status and test sessions.

use serde::{Deserialize, Serialize};

/// One wake-word operation, carried by `Request::WakeWord`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum WakeWordOp {
    /// Report the configured phrase. Emits `WakeWordStatus`, then `Done`.
    Status,
    /// Open a test session on this connection, which streams `WakeWordTest`
    /// and `WakeWordSpeech` events until stopped or disconnected.
    TestStart,
    /// End the open test session.
    TestStop,
}
