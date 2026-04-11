pub mod config;
pub mod socket;
pub mod state;

pub use assistd_ipc as ipc;
pub use config::Config;
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
