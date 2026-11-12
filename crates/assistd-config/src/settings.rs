//! Each config key's label, description, visibility and kind in the settings
//! panel, declared beside its table with `describe_fields!`.

use std::collections::{BTreeMap, HashMap};
use std::num::{NonZeroU16, NonZeroU32, NonZeroU64};
use std::sync::LazyLock;

use crate::custom_args::CustomArgs;
use crate::embedding::EmbeddingModel;
use crate::mcp::McpServerConfig;
use crate::search::{SearchCommand, SearchEngine};
use crate::theme::{ThemeColor, ThemeConfig, ThemeName};
use crate::tools::WritablePath;
use crate::top::Config;
use crate::values::{
    AbsoluteDir, Bounded, HfFile, LengthScale, LoopbackAddr, NonBlankString, NonEmpty,
    PresencePenalty, Temperature,
};
use crate::voice::{SttEncoder, UnitInterval, WakePhrase};

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
    /// How the key is edited and shown.
    pub kind: SettingKind,
}

/// How a setting is edited and shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingKind {
    Toggle,
    Integer,
    Decimal,
    Text,
    /// Free text long enough to need a wrapped, multi-line box.
    Prose,
    /// One of a fixed set of strings, cycled in place.
    Choice(&'static [&'static str]),
    /// A TOML array, edited entry by entry.
    List(ListItem),
    /// Too structured to edit in a line; shown as an entry count.
    External,
    /// A theme color, picked in the color picker and shown in itself.
    Color,
    /// The saved themes; opens the list that loads, saves and deletes themes.
    Themes,
}

/// What one entry of a [`SettingKind::List`] holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListItem {
    Text,
    /// A SearXNG engine name, with the file holding its API key when it
    /// needs one.
    SearchEngine,
}

/// A config field type whose values the settings panel edits one way.
pub trait SettingValue {
    const KIND: SettingKind;
}

impl<T: SettingValue> SettingValue for Option<T> {
    const KIND: SettingKind = T::KIND;
}

impl<S> SettingValue for CustomArgs<S> {
    const KIND: SettingKind = SettingKind::Text;
}

impl SettingValue for Vec<String> {
    const KIND: SettingKind = SettingKind::List(ListItem::Text);
}

impl SettingValue for NonEmpty<WritablePath> {
    const KIND: SettingKind = SettingKind::List(ListItem::Text);
}

impl SettingValue for NonEmpty<SearchEngine> {
    const KIND: SettingKind = SettingKind::List(ListItem::SearchEngine);
}

impl SettingValue for Vec<McpServerConfig> {
    const KIND: SettingKind = SettingKind::External;
}

impl SettingValue for ThemeColor {
    const KIND: SettingKind = SettingKind::Color;
}

impl<const MIN: u32, const MAX: u32> SettingValue for Bounded<MIN, MAX> {
    const KIND: SettingKind = SettingKind::Integer;
}

impl SettingValue for BTreeMap<ThemeName, ThemeConfig> {
    const KIND: SettingKind = SettingKind::Themes;
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

/// The kind `value`'s type is edited as.
pub(crate) fn kind_of<T: SettingValue>(_value: &T) -> SettingKind {
    T::KIND
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
/// without its entry fails to compile. A key's kind is its field type's, or
/// `Prose` for an entry marked `prose`. See the arms for the entry forms.
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
            @push $prefix, $out, $field, $key, $label, $description, false,
            $crate::settings::kind_of(&$config.$field)
        );
    };
    (@entry $config:tt, $prefix:tt, $out:tt, $field:ident, $key:tt,
        ($label:literal, $description:literal, advanced)) => {
        $crate::settings::describe_fields!(
            @push $prefix, $out, $field, $key, $label, $description, true,
            $crate::settings::kind_of(&$config.$field)
        );
    };
    (@entry $config:tt, $prefix:tt, $out:tt, $field:ident, $key:tt,
        ($label:literal, $description:literal, prose)) => {
        $crate::settings::describe_fields!(
            @push $prefix, $out, $field, $key, $label, $description, false,
            $crate::settings::SettingKind::Prose
        );
    };
    (@push $prefix:tt, $out:tt, $field:ident, $key:tt,
        $label:literal, $description:literal, $advanced:literal, $kind:expr) => {
        $out.push($crate::settings::SettingInfo {
            key: $crate::settings::child_key(
                $prefix,
                $crate::settings::describe_fields!(@key $field, $key),
            ),
            label: $label,
            description: $description,
            advanced: $advanced,
            kind: $kind,
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

/// Implements [`SettingValue`] with one kind for each listed type.
macro_rules! setting_values {
    ($kind:ident: $($value:ty),*) => {
        $(
            impl SettingValue for $value {
                const KIND: SettingKind = SettingKind::$kind;
            }
        )*
    };
}

setting_values!(Toggle: bool);
setting_values!(Integer: u32, u64, NonZeroU16, NonZeroU32, NonZeroU64);
setting_values!(Decimal: UnitInterval, Temperature, PresencePenalty, LengthScale);
setting_values!(Text: String, SearchCommand, NonBlankString, LoopbackAddr, AbsoluteDir);
setting_values!(Text: HfFile, SttEncoder, EmbeddingModel, WakePhrase);

/// Implements [`SettingValue`] for an enum as a choice between its variants'
/// TOML names; a variant left out fails to compile.
macro_rules! choice_setting {
    ($choice:ident { $($variant:ident => $name:literal),* $(,)? }) => {
        impl $crate::settings::SettingValue for $choice {
            const KIND: $crate::settings::SettingKind =
                $crate::settings::SettingKind::Choice(&[$($name),*]);
        }

        const _: fn($choice) -> &'static str = |choice| match choice {
            $($choice::$variant => $name),*
        };
    };
}

pub(crate) use {choice_setting, describe_fields};
