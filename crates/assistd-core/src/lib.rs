pub mod config;
pub mod presence;
pub mod socket;
pub mod state;

pub use assistd_ipc as ipc;
pub use assistd_ipc::PresenceState;
pub use assistd_tools::{CommandRegistry, ToolRegistry};
pub use config::{Config, DaemonConfig, PresenceConfig, SleepConfig};
pub use presence::{PresenceManager, RequestGuard};
pub use state::AppState;

/// Returns the version string of the assistd-core crate.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_not_empty() {
        assert!(!version().is_empty());
    }
}
