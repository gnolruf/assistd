use crate::defaults::DEFAULT_DAEMON_SHUTDOWN_GRACE_SECS;
use serde::{Deserialize, Serialize};

/// Daemon process lifecycle settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct DaemonConfig {
    /// Seconds to wait for in-flight IPC connections (e.g. streaming LLM
    /// responses) to finish before aborting them on daemon shutdown. `0`
    /// aborts in-flight work immediately.
    pub shutdown_grace_secs: u64,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            shutdown_grace_secs: DEFAULT_DAEMON_SHUTDOWN_GRACE_SECS,
        }
    }
}
