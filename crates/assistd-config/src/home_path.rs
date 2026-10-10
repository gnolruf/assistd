//! Serde helpers that expand a leading `~` in path-valued config keys.

use std::path::PathBuf;

use assistd_utils::path::expand_tilde_from_env;
use serde::{Deserialize, Deserializer};

pub(crate) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<PathBuf, D::Error> {
    String::deserialize(deserializer).map(|raw| expand_tilde_from_env(&raw))
}

pub(crate) fn deserialize_option<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<PathBuf>, D::Error> {
    Option::<String>::deserialize(deserializer).map(|raw| raw.as_deref().map(expand_tilde_from_env))
}
