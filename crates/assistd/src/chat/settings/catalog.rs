//! The settings the panel lists, grouped into sections in display order,
//! hotkeys first. A feature's section heading is the checkbox that enables it.

const COMPOSITOR_TYPES: &[&str] = &["auto", "i3", "sway", "hyprland"];
const SANDBOX_MODES: &[&str] = &["auto", "bwrap", "none"];

pub(crate) const SECTIONS: &[Section] = &[
    section(
        "Hotkeys",
        "hotkeys",
        &[
            basic("hotkeys.push_to_talk", SettingKind::Text),
            basic("hotkeys.toggle_listening", SettingKind::Text),
            basic("hotkeys.toggle_speech", SettingKind::Text),
            basic("hotkeys.skip_speech", SettingKind::Text),
        ],
    ),
    section(
        "Model",
        "model",
        &[
            basic("model.name", SettingKind::Text),
            basic("model.context_length", SettingKind::Integer),
            basic("model.gpu_layers", SettingKind::Integer),
            advanced("model.server_binary", SettingKind::Text),
            advanced("model.host", SettingKind::Text),
            advanced("model.port", SettingKind::Integer),
            advanced("model.ready_timeout_secs", SettingKind::Integer),
            advanced("model.custom_args", SettingKind::Text),
        ],
    ),
    section(
        "Chat",
        "chat",
        &[
            basic("chat.system_prompt", SettingKind::Prose),
            basic("chat.temperature", SettingKind::Decimal),
            basic("chat.max_response_tokens", SettingKind::Integer),
            advanced("chat.max_history_tokens", SettingKind::Integer),
            advanced("chat.summary_target_tokens", SettingKind::Integer),
            advanced("chat.max_reasoning_tokens", SettingKind::Integer),
            advanced("chat.reasoning_effort", SettingKind::Text),
            advanced("chat.request_timeout_secs", SettingKind::Integer),
            advanced("chat.summary_temperature", SettingKind::Decimal),
            advanced("chat.top_p", SettingKind::Decimal),
            advanced("chat.top_k", SettingKind::Integer),
            advanced("chat.min_p", SettingKind::Decimal),
            advanced("chat.presence_penalty", SettingKind::Decimal),
        ],
    ),
    feature(
        "Voice input",
        "voice",
        enable("voice.enabled"),
        &[],
        &[
            basic("voice.mic_device", SettingKind::Text),
            basic("voice.transcription.model", SettingKind::Text),
            basic("voice.transcription.prefer_gpu", SettingKind::Toggle),
            advanced("voice.transcription.threads", SettingKind::Integer),
            advanced("voice.transcription.beams", SettingKind::Integer),
            advanced("voice.transcription.vad_enabled", SettingKind::Toggle)
                .labeled("enable silence trimming"),
            advanced("voice.transcription.vad_model", SettingKind::Text),
            advanced("voice.transcription.model_cache_dir", SettingKind::Text),
        ],
    ),
    feature(
        "Wake word",
        "voice.wakeword",
        enable("voice.wakeword.enabled"),
        &["voice.enabled"],
        &[
            basic("voice.wakeword.start_on_launch", SettingKind::Toggle),
            basic("voice.wakeword.barge_in", SettingKind::Toggle),
            basic("voice.wakeword.follow_up_ms", SettingKind::Integer),
            advanced("voice.wakeword.silence_ms", SettingKind::Integer),
            advanced("voice.wakeword.max_utterance_secs", SettingKind::Integer),
            advanced("voice.wakeword.command_timeout_ms", SettingKind::Integer),
            advanced("voice.wakeword.match_similarity", SettingKind::Decimal),
        ],
    ),
    feature(
        "Spoken replies",
        "voice.synthesis",
        enable("voice.synthesis.enabled"),
        &[],
        &[
            basic("voice.synthesis.voice", SettingKind::Text),
            basic("voice.synthesis.length_scale", SettingKind::Decimal),
            basic("voice.synthesis.output_device", SettingKind::Text),
            advanced("voice.synthesis.binary_path", SettingKind::Text),
            advanced("voice.synthesis.model_cache_dir", SettingKind::Text),
            advanced("voice.synthesis.espeak_data_dir", SettingKind::Text),
            advanced("voice.synthesis.deadline_secs", SettingKind::Integer),
            advanced("voice.synthesis.max_sentence_chars", SettingKind::Integer),
            advanced("voice.synthesis.partial_flush_ms", SettingKind::Integer),
            advanced("voice.synthesis.use_cuda", SettingKind::Toggle),
        ],
    ),
    section(
        "Compositor",
        "compositor",
        &[basic(
            "compositor.type",
            SettingKind::Choice(COMPOSITOR_TYPES),
        )],
    ),
    section(
        "Sleep",
        "sleep",
        &[
            basic("sleep.idle_to_drowsy_mins", SettingKind::Integer),
            basic("sleep.idle_to_sleep_mins", SettingKind::Integer),
        ],
    ),
    feature(
        "GPU monitor",
        "sleep",
        enable("sleep.gpu_monitor_enabled"),
        &[],
        &[
            basic("sleep.gpu_poll_secs", SettingKind::Integer),
            basic("sleep.gpu_headroom_mb", SettingKind::Integer),
            basic("sleep.gpu_auto_wake", SettingKind::Toggle),
        ],
    ),
    section(
        "Tools",
        "tools",
        &[
            basic("tools.bash.sandbox", SettingKind::Choice(SANDBOX_MODES)),
            basic("tools.bash.timeout_secs", SettingKind::Integer),
            basic("tools.write.writable_paths", SettingKind::List),
            advanced("tools.bash.denylist", SettingKind::List),
            advanced("tools.bash.allowed_programs", SettingKind::List),
            advanced("tools.bash.destructive_patterns", SettingKind::List),
            advanced("tools.bash.bwrap_extra_args", SettingKind::List),
            advanced("tools.scratch.dir", SettingKind::Text),
            advanced("tools.scratch.retention_days", SettingKind::Integer),
            advanced("tools.output.max_lines", SettingKind::Integer),
            advanced("tools.output.max_kb", SettingKind::Integer),
            advanced("tools.output.overflow_dir", SettingKind::Text),
        ],
    ),
    feature(
        "Memory",
        "memory",
        enable("memory.enabled"),
        &[],
        &[basic("memory.db_path", SettingKind::Text)],
    ),
    feature(
        "Embedding",
        "embedding",
        enable("embedding.enabled"),
        &[],
        &[
            basic("embedding.model", SettingKind::Text),
            basic("embedding.top_k", SettingKind::Integer),
            basic("embedding.auto_inject", SettingKind::Toggle),
            advanced("embedding.server_binary", SettingKind::Text),
            advanced("embedding.host", SettingKind::Text),
            advanced("embedding.port", SettingKind::Integer),
            advanced("embedding.gpu_layers", SettingKind::Integer),
            advanced("embedding.ready_timeout_secs", SettingKind::Integer),
            advanced("embedding.custom_args", SettingKind::Text),
        ],
    ),
    feature(
        "Search",
        "search",
        enable("search.enabled"),
        &[],
        &[
            basic("search.engines", SettingKind::List),
            basic("search.command", SettingKind::Text),
            advanced("search.host", SettingKind::Text),
            advanced("search.port", SettingKind::Integer),
            advanced("search.ready_timeout_secs", SettingKind::Integer),
        ],
    ),
    feature(
        "MCP",
        "mcp",
        enable("mcp.enabled"),
        &[],
        &[basic("mcp.servers", SettingKind::External)],
    ),
    feature(
        "Notifications",
        "notifications",
        enable("notifications.enabled"),
        &[],
        &[
            basic("notifications.auto_hide_ms", SettingKind::Integer),
            basic("notifications.brief_when_away", SettingKind::Toggle),
        ],
    ),
    section(
        "Tray icons",
        "notifications.icons",
        &[
            basic("notifications.icons.active", SettingKind::Text),
            basic("notifications.icons.sleeping", SettingKind::Text),
            basic("notifications.icons.listening", SettingKind::Text),
            advanced("notifications.icons.generating", SettingKind::Text),
            advanced("notifications.icons.disconnected", SettingKind::Text),
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
    /// One of a fixed set of strings, cycled in place.
    Choice(&'static [&'static str]),
    /// A TOML array, edited as its literal.
    List,
    /// Too structured to edit in a line; shown as an entry count.
    External,
}

/// One config key, by its dotted path.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Setting {
    pub key: &'static str,
    pub kind: SettingKind,
    /// Listed only while its section shows advanced settings.
    pub advanced: bool,
    /// Shown in place of the key's name.
    label: Option<&'static str>,
}

impl Setting {
    const fn labeled(self, label: &'static str) -> Self {
        Self {
            label: Some(label),
            ..self
        }
    }
}

#[derive(Debug)]
pub(crate) struct Section {
    pub title: &'static str,
    /// The table every key in the section sits under.
    pub table: &'static str,
    /// The checkbox, drawn in the heading, that turns the section's feature
    /// on; the rest of the section is locked while it is off.
    pub enable: Option<Setting>,
    /// Checkboxes elsewhere that must be on before any of the section,
    /// [`enable`](Self::enable) included, can be edited.
    pub requires: &'static [&'static str],
    pub settings: &'static [Setting],
}

impl Section {
    /// `setting`'s label, else its key relative to the section's table.
    pub(crate) fn label(&self, setting: &Setting) -> &'static str {
        setting.label.unwrap_or_else(|| {
            setting
                .key
                .strip_prefix(self.table)
                .and_then(|rest| rest.strip_prefix('.'))
                .unwrap_or(setting.key)
        })
    }

    pub(crate) fn advanced_count(&self) -> usize {
        self.settings.iter().filter(|s| s.advanced).count()
    }
}

const fn section(
    title: &'static str,
    table: &'static str,
    settings: &'static [Setting],
) -> Section {
    Section {
        title,
        table,
        enable: None,
        requires: &[],
        settings,
    }
}

const fn feature(
    title: &'static str,
    table: &'static str,
    enable: Setting,
    requires: &'static [&'static str],
    settings: &'static [Setting],
) -> Section {
    Section {
        title,
        table,
        enable: Some(enable),
        requires,
        settings,
    }
}

const fn enable(key: &'static str) -> Setting {
    basic(key, SettingKind::Toggle)
}

const fn basic(key: &'static str, kind: SettingKind) -> Setting {
    Setting {
        key,
        kind,
        advanced: false,
        label: None,
    }
}

const fn advanced(key: &'static str, kind: SettingKind) -> Setting {
    Setting {
        key,
        kind,
        advanced: true,
        label: None,
    }
}

#[cfg(test)]
mod tests;
