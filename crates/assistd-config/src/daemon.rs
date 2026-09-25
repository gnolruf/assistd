use serde::{Deserialize, Serialize};

use crate::defaults::DEFAULT_DAEMON_SHUTDOWN_GRACE_SECS;

/// Daemon process lifecycle settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct DaemonConfig {
    /// Seconds in-flight IPC requests get to finish on shutdown. `0` aborts
    /// them immediately.
    pub shutdown_grace_secs: u64,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            shutdown_grace_secs: DEFAULT_DAEMON_SHUTDOWN_GRACE_SECS,
        }
    }
}
