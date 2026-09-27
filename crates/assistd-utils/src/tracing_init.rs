//! Tracing subscriber setup shared by the binaries and integration tests.

use std::sync::Once;

use tracing_subscriber::EnvFilter;

/// The `RUST_LOG` filter when it is set and parses, else `default_directives`.
pub fn env_filter_or(default_directives: &str) -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_directives))
}

/// Install a test-writer subscriber filtered by [`env_filter_or`] once per
/// process; later calls do nothing.
pub fn init_test_tracing(default_directives: &str) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(env_filter_or(default_directives))
            .with_test_writer()
            .try_init();
    });
}
