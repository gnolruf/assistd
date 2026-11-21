//! The settings the panel lists, grouped into sections in display order and
//! sections into tabs. A feature's section heading is the checkbox that enables it.

use assistd_config::{SettingInfo, SettingKind, setting_info};

pub(crate) const SECTIONS: &[Section] = &[
    section(
        Tab::Model,
        "Model",
        &[
            setting("model.name"),
            setting("model.context_length"),
            setting("model.gpu_layers"),
            setting("model.server_binary"),
            setting("model.port"),
            setting("model.ready_timeout_secs"),
            setting("model.custom_args"),
        ],
    ),
    section(
        Tab::Model,
        "Chat",
        &[
            setting("chat.system_prompt"),
            setting("chat.temperature"),
            setting("chat.max_response_tokens"),
            setting("chat.max_history_tokens"),
            setting("chat.summary_target_tokens"),
            setting("chat.max_reasoning_tokens"),
            setting("chat.reasoning_effort"),
            setting("chat.request_timeout_secs"),
            setting("chat.top_p"),
            setting("chat.top_k"),
            setting("chat.min_p"),
            setting("chat.presence_penalty"),
        ],
    ),
    feature(
        Tab::Voice,
        "Voice input",
        enable("voice.enabled"),
        &[],
        &[
            setting("voice.mic_device"),
            setting("voice.session_hold_secs"),
            setting("voice.transcription.model"),
            setting("voice.transcription.prefer_gpu"),
            setting("voice.transcription.threads"),
            setting("voice.transcription.vad_enabled"),
            setting("voice.transcription.model_cache_dir"),
        ],
    ),
    feature(
        Tab::Voice,
        "Wake word",
        enable("voice.wakeword.enabled"),
        &["voice.enabled"],
        &[
            setting("voice.wakeword.phrase"),
            setting("voice.wakeword.start_on_launch"),
            setting("voice.wakeword.barge_in"),
            setting("voice.wakeword.follow_up_ms"),
            setting("voice.wakeword.silence_ms"),
            setting("voice.wakeword.max_utterance_secs"),
            setting("voice.wakeword.command_timeout_ms"),
            setting("voice.wakeword.threshold"),
        ],
    ),
    feature(
        Tab::Voice,
        "Spoken replies",
        enable("voice.synthesis.enabled"),
        &[],
        &[
            setting("voice.synthesis.voice"),
            setting("voice.synthesis.family"),
            setting("voice.synthesis.speaker_id"),
            setting("voice.synthesis.conversational"),
            setting("voice.synthesis.output_device"),
            setting("voice.synthesis.length_scale"),
            setting("voice.synthesis.model_cache_dir"),
            setting("voice.synthesis.max_sentence_chars"),
            setting("voice.synthesis.partial_flush_ms"),
        ],
    ),
    section(Tab::General, "Compositor", &[setting("compositor.type")]),
    section(
        Tab::General,
        "Sleep",
        &[
            setting("sleep.idle_to_drowsy_mins"),
            setting("sleep.idle_to_sleep_mins"),
        ],
    ),
    feature(
        Tab::General,
        "GPU monitor",
        enable("sleep.gpu_monitor_enabled"),
        &[],
        &[
            setting("sleep.gpu_headroom_mb"),
            setting("sleep.gpu_auto_wake"),
        ],
    ),
    section(
        Tab::Tools,
        "Tools",
        &[
            setting("tools.bash.sandbox"),
            setting("tools.bash.timeout_secs"),
            setting("tools.allow_auto_mode"),
            setting("tools.default_permission_mode"),
            setting("tools.write.writable_paths"),
            setting("tools.bash.bwrap_extra_args"),
            setting("tools.scratch.dir"),
            setting("tools.scratch.retention_days"),
            setting("tools.output.max_lines"),
            setting("tools.output.max_kb"),
            setting("tools.output.overflow_dir"),
        ],
    ),
    feature(
        Tab::Model,
        "Embedding",
        enable("embedding.enabled"),
        &[],
        &[
            setting("embedding.model"),
            setting("embedding.document_prefix"),
            setting("embedding.query_prefix"),
            setting("embedding.top_k"),
            setting("embedding.auto_inject"),
            setting("embedding.min_similarity"),
            setting("embedding.port"),
            setting("embedding.gpu_layers"),
            setting("embedding.ready_timeout_secs"),
            setting("embedding.custom_args"),
        ],
    ),
    feature(
        Tab::Tools,
        "Documents",
        enable("documents.enabled"),
        &["embedding.enabled"],
        &[
            setting("documents.roots"),
            setting("documents.exclude"),
            setting("documents.max_file_mb"),
            setting("documents.embed_during_interaction"),
            setting("documents.embed_during_tasks"),
            setting("documents.embed_during_gpu_sleep"),
            setting("documents.ocr.model"),
            setting("documents.ocr.prompt"),
        ],
    ),
    feature(
        Tab::Tools,
        "Search",
        enable("search.enabled"),
        &[],
        &[
            setting("search.engines"),
            setting("search.command"),
            setting("search.port"),
            setting("search.ready_timeout_secs"),
        ],
    ),
    feature(
        Tab::Tools,
        "MCP",
        enable("mcp.enabled"),
        &[],
        &[setting("mcp.servers")],
    ),
    section(Tab::General, "Memory", &[setting("memory.enabled")]),
    feature(
        Tab::General,
        "Notifications",
        enable("notifications.enabled"),
        &[],
        &[setting("notifications.auto_hide_ms")],
    ),
    section(
        Tab::General,
        "Tasks",
        &[
            setting("tasks.interruption"),
            setting("tasks.resume_after_idle_secs"),
            setting("tasks.kept_runs"),
        ],
    ),
    section(Tab::Theme, "Themes", &[setting("chat.themes")]),
    section(
        Tab::Theme,
        "Colors",
        &[
            setting("chat.theme.accent"),
            setting("chat.theme.user_message"),
            setting("chat.theme.assistant_text"),
            setting("chat.theme.status_text"),
            setting("chat.theme.code"),
            setting("chat.theme.border"),
            setting("chat.theme.success"),
            setting("chat.theme.error"),
            setting("chat.theme.warning"),
            setting("chat.theme.link"),
            setting("chat.theme.selection"),
        ],
    ),
];

/// One config key, by its dotted path. Its label, description, kind and
/// whether it is advanced are declared on the config field.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Setting {
    pub key: &'static str,
}

impl Setting {
    /// What the key is for, in words, shown in its place; the key itself if
    /// no config field has its path.
    pub(crate) fn label(&self) -> &'static str {
        self.info().map_or(self.key, |info| info.label)
    }

    /// One line on what the key does, shown while it is selected.
    pub(crate) fn description(&self) -> &'static str {
        self.info().map_or("", |info| info.description)
    }

    /// Listed only while its section shows advanced settings.
    pub(crate) fn advanced(&self) -> bool {
        self.info().is_some_and(|info| info.advanced)
    }

    /// How the key is edited and shown; `Text` if no config field has its path.
    pub(crate) fn kind(&self) -> SettingKind {
        self.info().map_or(SettingKind::Text, |info| info.kind)
    }

    fn info(&self) -> Option<&'static SettingInfo> {
        setting_info(self.key)
    }
}

/// A page of the panel, listing the sections filed under it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tab {
    General,
    Model,
    Voice,
    Tools,
    Theme,
}

impl Tab {
    /// Every tab, in the order the tab bar shows them.
    pub(crate) const ALL: [Self; 5] = [
        Self::General,
        Self::Model,
        Self::Voice,
        Self::Tools,
        Self::Theme,
    ];

    pub(crate) fn title(self) -> &'static str {
        match self {
            Self::General => "General",
            Self::Model => "Model",
            Self::Voice => "Voice",
            Self::Tools => "Tools",
            Self::Theme => "Theme",
        }
    }
}

#[derive(Debug)]
pub(crate) struct Section {
    pub tab: Tab,
    pub title: &'static str,
    /// The checkbox, drawn in the heading, that turns the section's feature
    /// on; the rest of the section is locked while it is off.
    pub enable: Option<Setting>,
    /// Checkboxes elsewhere that must be on before any of the section,
    /// [`enable`](Self::enable) included, can be edited.
    pub requires: &'static [&'static str],
    pub settings: &'static [Setting],
}

impl Section {
    pub(crate) fn advanced_count(&self) -> usize {
        self.settings.iter().filter(|s| s.advanced()).count()
    }
}

const fn section(tab: Tab, title: &'static str, settings: &'static [Setting]) -> Section {
    Section {
        tab,
        title,
        enable: None,
        requires: &[],
        settings,
    }
}

const fn feature(
    tab: Tab,
    title: &'static str,
    enable: Setting,
    requires: &'static [&'static str],
    settings: &'static [Setting],
) -> Section {
    Section {
        tab,
        title,
        enable: Some(enable),
        requires,
        settings,
    }
}

/// A section's checkbox; its heading shows the title in place of a label.
const fn enable(key: &'static str) -> Setting {
    setting(key)
}

const fn setting(key: &'static str) -> Setting {
    Setting { key }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use assistd_config::{Config, ConfigDocument, SettingKind};

    use super::*;

    fn collect_leaf_keys(table: &toml::Table, prefix: &str, keys: &mut Vec<String>) {
        for (name, value) in table {
            let key = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{prefix}.{name}")
            };
            match value {
                toml::Value::Table(child) => collect_leaf_keys(child, &key, keys),
                _ => keys.push(key),
            }
        }
    }

    fn sample(kind: SettingKind) -> toml::Value {
        match kind {
            SettingKind::Toggle => toml::Value::Boolean(false),
            SettingKind::Integer => toml::Value::Integer(1),
            SettingKind::Decimal => toml::Value::Float(0.5),
            SettingKind::Text | SettingKind::Prose => toml::Value::from("/x"),
            SettingKind::Color => toml::Value::from("red"),
            SettingKind::Themes => toml::Value::Table(toml::Table::new()),
            SettingKind::Choice(options) => toml::Value::from(options[0]),
            SettingKind::List(_) | SettingKind::McpServers => toml::Value::Array(Vec::new()),
        }
    }

    #[test]
    fn catalog_lists_every_schema_key_and_every_value_kind_is_accepted() {
        let defaults = toml::Table::try_from(Config::default()).unwrap();
        let mut schema_keys = Vec::new();
        collect_leaf_keys(&defaults, "", &mut schema_keys);
        let settings = || {
            SECTIONS
                .iter()
                .flat_map(|section| section.enable.iter().chain(section.settings))
        };
        let listed: HashSet<&str> = settings().map(|setting| setting.key).collect();
        for key in &schema_keys {
            assert!(listed.contains(key.as_str()), "{key} is not in the catalog");
        }

        let root = tempfile::tempdir().unwrap();
        let mut document = ConfigDocument::open(&root.path().join("config.toml")).unwrap();
        for setting in settings() {
            assert!(
                setting_info(setting.key).is_some(),
                "{} has no describe_fields! entry",
                setting.key
            );
            let value = document
                .default_value(setting.key)
                .cloned()
                .unwrap_or_else(|| sample(setting.kind()));
            let choices = match setting.kind() {
                SettingKind::Choice(options) => options,
                _ => &[],
            };
            let values = choices.iter().map(|option| toml::Value::from(*option));
            for value in values.chain([value]) {
                if let Err(e) = document.set(setting.key, &value) {
                    panic!("{} = {value}: {e}", setting.key);
                }
            }
        }
    }
}
