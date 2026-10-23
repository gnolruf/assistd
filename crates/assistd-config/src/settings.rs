//! Each config key's label, description and visibility in the settings
//! panel, declared beside its table with `describe_fields!`.

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::top::Config;

static SETTINGS: LazyLock<HashMap<String, SettingInfo>> = LazyLock::new(|| {
    let mut settings = Vec::new();
    Config::default().describe_settings("", &mut settings);
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

/// A config table that can list the settings info of every key under it.
pub trait DescribeSettings {
    /// Appends every key under `prefix` in field order; the root's prefix is empty.
    fn describe_settings(&self, prefix: &str, out: &mut Vec<SettingInfo>);
}

/// The info for `key`, or `None` when no config field has that path.
pub fn setting_info(key: &str) -> Option<&'static SettingInfo> {
    SETTINGS.get(key)
}

/// `name`'s dotted path under `prefix`; the root's prefix is empty.
pub(crate) fn child_key(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

/// Implements [`DescribeSettings::describe_settings`] from one entry per field,
/// destructuring `Self` without `..` so a field added, removed or renamed
/// without its entry fails to compile. See the arms for the entry forms.
macro_rules! describe_fields {
    (@key $field:ident, ()) => {
        stringify!($field)
    };
    (@key $field:ident, ($key:literal)) => {
        $key
    };
    (@entry $config:tt, $prefix:tt, $out:tt, $field:ident, $key:tt, skip) => {};
    (@entry $config:tt, $prefix:tt, $out:tt, $field:ident, $key:tt, nested) => {
        $crate::settings::DescribeSettings::describe_settings(
            &$config.$field,
            &$crate::settings::child_key(
                $prefix,
                $crate::settings::describe_fields!(@key $field, $key),
            ),
            $out,
        );
    };
    (@entry $config:tt, $prefix:tt, $out:tt, $field:ident, $key:tt,
        ($label:literal, $description:literal)) => {
        $crate::settings::describe_fields!(
            @push $prefix, $out, $field, $key, $label, $description, false
        );
    };
    (@entry $config:tt, $prefix:tt, $out:tt, $field:ident, $key:tt,
        ($label:literal, $description:literal, advanced)) => {
        $crate::settings::describe_fields!(
            @push $prefix, $out, $field, $key, $label, $description, true
        );
    };
    (@push $prefix:tt, $out:tt, $field:ident, $key:tt,
        $label:literal, $description:literal, $advanced:literal) => {
        $out.push($crate::settings::SettingInfo {
            key: $crate::settings::child_key(
                $prefix,
                $crate::settings::describe_fields!(@key $field, $key),
            ),
            label: $label,
            description: $description,
            advanced: $advanced,
        });
    };
    ($($field:ident $(as $key:literal)? => $setting:tt),* $(,)?) => {
        fn describe_settings(
            &self,
            prefix: &str,
            out: &mut ::std::vec::Vec<$crate::settings::SettingInfo>,
        ) {
            let Self { $($field: _),* } = self;
            $(
                $crate::settings::describe_fields!(
                    @entry self, prefix, out, $field, ($($key)?), $setting
                );
            )*
        }
    };
}

pub(crate) use describe_fields;
