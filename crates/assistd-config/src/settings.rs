//! Each config key's label, description and visibility in the settings
//! panel, declared beside its field with `#[setting(...)]`.

use std::collections::HashMap;
use std::sync::LazyLock;

pub use assistd_config_derive::DescribeSettings;

use crate::top::Config;

static SETTINGS: LazyLock<HashMap<String, SettingInfo>> = LazyLock::new(|| {
    let mut settings = Vec::new();
    Config::describe_settings("", &mut settings);
    settings
        .into_iter()
        .map(|setting| (setting.key.clone(), setting))
        .collect()
});

/// One config key as the settings panel presents it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingInfo {
    /// Dotted path, as written in `config.toml`.
    pub key: String,
    /// What the key is for, in a few words, listed in its place.
    pub label: &'static str,
    /// One line on what the key does, shown while it is selected.
    pub description: &'static str,
    /// Listed only while its section shows advanced settings.
    pub advanced: bool,
}

/// A config table whose fields each carry `#[setting(...)]`.
pub trait DescribeSettings {
    /// Appends every key under `prefix` in field order; the root's prefix is empty.
    fn describe_settings(prefix: &str, out: &mut Vec<SettingInfo>);
}

/// The info for `key`, or `None` when no config field has that path.
pub fn setting_info(key: &str) -> Option<&'static SettingInfo> {
    SETTINGS.get(key)
}

/// `name`'s dotted path under `prefix`; the root's prefix is empty.
pub fn child_key(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}
