//! Serde helper that expands a leading `~` in a path-valued config key.

use std::path::PathBuf;

use assistd_utils::path::expand_tilde_from_env;
use serde::{Deserialize, Deserializer};

pub(crate) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<PathBuf, D::Error> {
    String::deserialize(deserializer).map(|raw| expand_tilde_from_env(&raw))
}
