//! Test fixtures. A canonical minimal TOML plus helpers for building
//! [`Config`] instances that tests can mutate before running their
//! assertions. Available outside `#[cfg(test)]` so downstream crates
//! (`assistd-core`, `assistd-llm`) can reuse the same fixtures from
//! their own test modules.

use crate::top::Config;

/// Canonical minimal TOML covering every required section. Test modules
/// that need to exercise a specific field can either mutate the parsed
/// [`minimal`] instance, or `format!` additional `[section]` blocks onto
/// the end of this string.
pub fn minimal_toml() -> &'static str {
    include_str!("../tests/fixtures/minimal.toml")
}

/// Parse [`minimal_toml`] into a valid [`Config`]. Panics on failure; the
/// fixture is a code-controlled literal, so a failure is a bug, not an
/// environmental error.
pub fn minimal() -> Config {
    toml::from_str(minimal_toml()).expect("minimal fixture TOML must parse")
}
