//! The settings the panel lists, grouped into sections in display order and
//! sections into tabs. A feature's section heading is the checkbox that enables it.

use assistd_config::{SettingInfo, setting_info};

const COMPOSITOR_TYPES: &[&str] = &["auto", "i3", "sway", "hyprland"];
const SANDBOX_MODES: &[&str] = &["auto", "bwrap", "none"];
const VOICE_FAMILIES: &[&str] = &["vits", "kokoro"];

pub(crate) const SECTIONS: &[Section] = &[
    section(
        Tab::General,
        "Hotkeys",
        &[
            setting("hotkeys.push_to_talk", SettingKind::Hotkey),
            setting("hotkeys.toggle_listening", SettingKind::Hotkey),
            setting("hotkeys.toggle_speech", SettingKind::Hotkey),
            setting("hotkeys.skip_speech", SettingKind::Hotkey),
        ],
    ),
    section(
        Tab::Model,
        "Model",
        &[
            setting("model.name", SettingKind::Text),
            setting("model.context_length", SettingKind::Integer),
            setting("model.gpu_layers", SettingKind::Integer),
            setting("model.server_binary", SettingKind::Text),
            setting("model.host", SettingKind::Text),
            setting("model.port", SettingKind::Integer),
            setting("model.ready_timeout_secs", SettingKind::Integer),
            setting("model.custom_args", SettingKind::Text),
        ],
    ),
    section(
        Tab::Model,
        "Chat",
        &[
            setting("chat.system_prompt", SettingKind::Prose),
            setting("chat.temperature", SettingKind::Decimal),
            setting("chat.max_response_tokens", SettingKind::Integer),
            setting("chat.max_history_tokens", SettingKind::Integer),
            setting("chat.summary_target_tokens", SettingKind::Integer),
            setting("chat.max_reasoning_tokens", SettingKind::Integer),
            setting("chat.reasoning_effort", SettingKind::Text),
            setting("chat.request_timeout_secs", SettingKind::Integer),
            setting("chat.summary_temperature", SettingKind::Decimal),
            setting("chat.top_p", SettingKind::Decimal),
            setting("chat.top_k", SettingKind::Integer),
            setting("chat.min_p", SettingKind::Decimal),
            setting("chat.presence_penalty", SettingKind::Decimal),
        ],
    ),
    feature(
        Tab::Voice,
        "Voice input",
        enable("voice.enabled"),
        &[],
        &[
            setting("voice.mic_device", SettingKind::Text),
            setting("voice.transcription.model", SettingKind::Text),
            setting("voice.transcription.prefer_gpu", SettingKind::Toggle),
            setting("voice.transcription.threads", SettingKind::Integer),
            setting("voice.transcription.vad_enabled", SettingKind::Toggle),
            setting("voice.transcription.vad_model", SettingKind::Text),
            setting("voice.transcription.model_cache_dir", SettingKind::Text),
        ],
    ),
    feature(
        Tab::Voice,
        "Wake word",
        enable("voice.wakeword.enabled"),
        &["voice.enabled"],
        &[
            setting("voice.wakeword.phrase", SettingKind::Text),
            setting("voice.wakeword.start_on_launch", SettingKind::Toggle),
            setting("voice.wakeword.barge_in", SettingKind::Toggle),
            setting("voice.wakeword.follow_up_ms", SettingKind::Integer),
            setting("voice.wakeword.silence_ms", SettingKind::Integer),
            setting("voice.wakeword.max_utterance_secs", SettingKind::Integer),
            setting("voice.wakeword.command_timeout_ms", SettingKind::Integer),
            setting("voice.wakeword.threshold", SettingKind::Decimal),
        ],
    ),
    feature(
        Tab::Voice,
        "Spoken replies",
        enable("voice.synthesis.enabled"),
        &[],
        &[
            setting("voice.synthesis.voice", SettingKind::Text),
            setting(
                "voice.synthesis.family",
                SettingKind::Choice(VOICE_FAMILIES),
            ),
            setting("voice.synthesis.speaker_id", SettingKind::Integer),
            setting("voice.synthesis.output_device", SettingKind::Text),
            setting("voice.synthesis.length_scale", SettingKind::Decimal),
            setting("voice.synthesis.model_cache_dir", SettingKind::Text),
            setting("voice.synthesis.max_sentence_chars", SettingKind::Integer),
            setting("voice.synthesis.partial_flush_ms", SettingKind::Integer),
        ],
    ),
    section(
        Tab::General,
        "Compositor",
        &[setting(
            "compositor.type",
            SettingKind::Choice(COMPOSITOR_TYPES),
        )],
    ),
    section(
        Tab::General,
        "Sleep",
        &[
            setting("sleep.idle_to_drowsy_mins", SettingKind::Integer),
            setting("sleep.idle_to_sleep_mins", SettingKind::Integer),
        ],
    ),
    feature(
        Tab::General,
        "GPU monitor",
        enable("sleep.gpu_monitor_enabled"),
        &[],
        &[
            setting("sleep.gpu_poll_secs", SettingKind::Integer),
            setting("sleep.gpu_headroom_mb", SettingKind::Integer),
            setting("sleep.gpu_auto_wake", SettingKind::Toggle),
        ],
    ),
    section(
        Tab::Tools,
        "Tools",
        &[
            setting("tools.bash.sandbox", SettingKind::Choice(SANDBOX_MODES)),
            setting("tools.bash.timeout_secs", SettingKind::Integer),
            setting(
                "tools.write.writable_paths",
                SettingKind::List(ListItem::Text),
            ),
            setting("tools.bash.denylist", SettingKind::List(ListItem::Text)),
            setting(
                "tools.bash.allowed_programs",
                SettingKind::List(ListItem::Text),
            ),
            setting(
                "tools.bash.destructive_patterns",
                SettingKind::List(ListItem::Text),
            ),
            setting(
                "tools.bash.bwrap_extra_args",
                SettingKind::List(ListItem::Text),
            ),
            setting("tools.scratch.dir", SettingKind::Text),
            setting("tools.scratch.retention_days", SettingKind::Integer),
            setting("tools.output.max_lines", SettingKind::Integer),
            setting("tools.output.max_kb", SettingKind::Integer),
            setting("tools.output.overflow_dir", SettingKind::Text),
        ],
    ),
    feature(
        Tab::Model,
        "Embedding",
        enable("embedding.enabled"),
        &[],
        &[
            setting("embedding.model", SettingKind::Text),
            setting("embedding.top_k", SettingKind::Integer),
            setting("embedding.auto_inject", SettingKind::Toggle),
            setting("embedding.server_binary", SettingKind::Text),
            setting("embedding.host", SettingKind::Text),
            setting("embedding.port", SettingKind::Integer),
            setting("embedding.gpu_layers", SettingKind::Integer),
            setting("embedding.ready_timeout_secs", SettingKind::Integer),
            setting("embedding.custom_args", SettingKind::Text),
        ],
    ),
    feature(
        Tab::Tools,
        "Search",
        enable("search.enabled"),
        &[],
        &[
            setting("search.engines", SettingKind::List(ListItem::SearchEngine)),
            setting("search.command", SettingKind::Text),
            setting("search.host", SettingKind::Text),
            setting("search.port", SettingKind::Integer),
            setting("search.ready_timeout_secs", SettingKind::Integer),
        ],
    ),
    feature(
        Tab::Tools,
        "MCP",
        enable("mcp.enabled"),
        &[],
        &[setting("mcp.servers", SettingKind::External)],
    ),
    section(
        Tab::General,
        "Memory",
        &[setting("memory.enabled", SettingKind::Toggle)],
    ),
    feature(
        Tab::General,
        "Notifications",
        enable("notifications.enabled"),
        &[],
        &[
            setting("notifications.auto_hide_ms", SettingKind::Integer),
            setting("notifications.brief_when_away", SettingKind::Toggle),
        ],
    ),
    section(
        Tab::General,
        "Tray icons",
        &[
            setting("notifications.icons.active", SettingKind::Text),
            setting("notifications.icons.sleeping", SettingKind::Text),
            setting("notifications.icons.listening", SettingKind::Text),
            setting("notifications.icons.generating", SettingKind::Text),
            setting("notifications.icons.disconnected", SettingKind::Text),
        ],
    ),
];

/// How a setting is edited and shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SettingKind {
    Toggle,
    Integer,
    Decimal,
    Text,
    /// Free text long enough to need a wrapped, multi-line box.
    Prose,
    /// A key combination, set by pressing it.
    Hotkey,
    /// One of a fixed set of strings, cycled in place.
    Choice(&'static [&'static str]),
    /// A TOML array, edited entry by entry.
    List(ListItem),
    /// Too structured to edit in a line; shown as an entry count.
    External,
}

/// What one entry of a [`SettingKind::List`] holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ListItem {
    Text,
    /// A SearXNG engine name, with the file holding its API key when it
    /// needs one.
    SearchEngine,
}

/// One config key, by its dotted path. Its label, description and whether
/// it is advanced are declared on the config field.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Setting {
    pub key: &'static str,
    pub kind: SettingKind,
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
}

impl Tab {
    /// Every tab, in the order the tab bar shows them.
    pub(crate) const ALL: [Self; 4] = [Self::General, Self::Model, Self::Voice, Self::Tools];

    pub(crate) fn title(self) -> &'static str {
        match self {
            Self::General => "General",
            Self::Model => "Model",
            Self::Voice => "Voice",
            Self::Tools => "Tools",
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
    setting(key, SettingKind::Toggle)
}

const fn setting(key: &'static str, kind: SettingKind) -> Setting {
    Setting { key, kind }
}

#[cfg(test)]
mod tests;
