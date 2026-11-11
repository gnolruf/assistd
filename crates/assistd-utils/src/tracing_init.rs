//! Tracing subscriber setup shared by the binaries and integration tests.

use tracing_subscriber::EnvFilter;

/// The `RUST_LOG` filter when it is set and parses, else `default_directives`.
pub fn env_filter_or(default_directives: &str) -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_directives))
}

/// Install a test-writer subscriber filtered by [`env_filter_or`]; does
/// nothing when the process already has a subscriber.
#[cfg(feature = "test-support")]
pub fn init_test_tracing(default_directives: &str) {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(env_filter_or(default_directives))
        .with_test_writer()
        .try_init();
}
