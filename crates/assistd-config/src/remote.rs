use crate::defaults::{DEFAULT_REMOTE_BIND_ADDRESS, DEFAULT_REMOTE_PORT};
use serde::{Deserialize, Serialize};

/// Remote access API settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RemoteConfig {
    /// Whether the remote access API is enabled.
    pub enabled: bool,
    /// IP address to bind to.
    pub bind_address: String,
    /// TCP port to listen on.
    pub port: u16,
}

impl Default for RemoteConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bind_address: DEFAULT_REMOTE_BIND_ADDRESS.to_string(),
            port: DEFAULT_REMOTE_PORT,
        }
    }
}
