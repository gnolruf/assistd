//! Test fixtures, exported so downstream crates' tests can share them.

use crate::top::Config;

/// Minimal TOML carrying only the overrides tests rely on.
pub fn minimal_toml() -> &'static str {
    include_str!("../tests/fixtures/minimal.toml")
}

/// Parse [`minimal_toml`] into a [`Config`].
pub fn minimal() -> Config {
    toml::from_str(minimal_toml()).expect("minimal fixture TOML must parse")
}
