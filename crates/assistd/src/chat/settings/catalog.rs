//! The settings the panel lists, grouped into sections in display order,
//! hotkeys first.

const COMPOSITOR_TYPES: &[&str] = &["auto", "i3", "sway", "hyprland"];
const SANDBOX_MODES: &[&str] = &["auto", "bwrap", "none"];

pub(crate) const SECTIONS: &[Section] = &[
    Section {
        title: "Hotkeys",
        table: "hotkeys",
        settings: &[
            basic("hotkeys.push_to_talk", SettingKind::Text),
            basic("hotkeys.toggle_listening", SettingKind::Text),
            basic("hotkeys.toggle_speech", SettingKind::Text),
            basic("hotkeys.skip_speech", SettingKind::Text),
        ],
    },
    Section {
        title: "Model",
        table: "model",
        settings: &[
            basic("model.name", SettingKind::Text),
            basic("model.context_length", SettingKind::Integer),
            basic("model.gpu_layers", SettingKind::Integer),
            advanced("model.server_binary", SettingKind::Text),
            advanced("model.host", SettingKind::Text),
            advanced("model.port", SettingKind::Integer),
            advanced("model.ready_timeout_secs", SettingKind::Integer),
            advanced("model.custom_args", SettingKind::Text),
        ],
    },
    Section {
        title: "Chat",
        table: "chat",
        settings: &[
            basic("chat.system_prompt", SettingKind::Text),
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
    },
    Section {
        title: "Voice",
        table: "voice",
        settings: &[
            basic("voice.enabled", SettingKind::Toggle),
            basic("voice.wakeword.enabled", SettingKind::Toggle),
            basic("voice.synthesis.enabled", SettingKind::Toggle),
            advanced("voice.mic_device", SettingKind::Text),
            advanced("voice.transcription.model", SettingKind::Text),
            advanced("voice.transcription.prefer_gpu", SettingKind::Toggle),
            advanced("voice.transcription.threads", SettingKind::Integer),
            advanced("voice.transcription.beams", SettingKind::Integer),
            advanced("voice.transcription.vad_enabled", SettingKind::Toggle),
            advanced("voice.transcription.vad_model", SettingKind::Text),
            advanced("voice.transcription.model_cache_dir", SettingKind::Text),
            advanced("voice.wakeword.start_on_launch", SettingKind::Toggle),
            advanced("voice.wakeword.silence_ms", SettingKind::Integer),
            advanced("voice.wakeword.max_utterance_secs", SettingKind::Integer),
            advanced("voice.wakeword.barge_in", SettingKind::Toggle),
            advanced("voice.wakeword.command_timeout_ms", SettingKind::Integer),
            advanced("voice.wakeword.follow_up_ms", SettingKind::Integer),
            advanced("voice.wakeword.match_similarity", SettingKind::Decimal),
            advanced("voice.synthesis.binary_path", SettingKind::Text),
            advanced("voice.synthesis.voice", SettingKind::Text),
            advanced("voice.synthesis.model_cache_dir", SettingKind::Text),
            advanced("voice.synthesis.length_scale", SettingKind::Decimal),
            advanced("voice.synthesis.espeak_data_dir", SettingKind::Text),
            advanced("voice.synthesis.deadline_secs", SettingKind::Integer),
            advanced("voice.synthesis.max_sentence_chars", SettingKind::Integer),
            advanced("voice.synthesis.partial_flush_ms", SettingKind::Integer),
            advanced("voice.synthesis.use_cuda", SettingKind::Toggle),
            advanced("voice.synthesis.output_device", SettingKind::Text),
        ],
    },
    Section {
        title: "Compositor",
        table: "compositor",
        settings: &[basic(
            "compositor.type",
            SettingKind::Choice(COMPOSITOR_TYPES),
        )],
    },
    Section {
        title: "Sleep",
        table: "sleep",
        settings: &[
            basic("sleep.idle_to_drowsy_mins", SettingKind::Integer),
            basic("sleep.idle_to_sleep_mins", SettingKind::Integer),
            basic("sleep.gpu_monitor_enabled", SettingKind::Toggle),
            advanced("sleep.gpu_poll_secs", SettingKind::Integer),
            advanced("sleep.gpu_headroom_mb", SettingKind::Integer),
            advanced("sleep.gpu_auto_wake", SettingKind::Toggle),
        ],
    },
    Section {
        title: "Tools",
        table: "tools",
        settings: &[
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
    },
    Section {
        title: "Memory",
        table: "memory",
        settings: &[
            basic("memory.enabled", SettingKind::Toggle),
            basic("memory.db_path", SettingKind::Text),
        ],
    },
    Section {
        title: "Embedding",
        table: "embedding",
        settings: &[
            basic("embedding.enabled", SettingKind::Toggle),
            basic("embedding.model", SettingKind::Text),
            basic("embedding.top_k", SettingKind::Integer),
            advanced("embedding.auto_inject", SettingKind::Toggle),
            advanced("embedding.server_binary", SettingKind::Text),
            advanced("embedding.host", SettingKind::Text),
            advanced("embedding.port", SettingKind::Integer),
            advanced("embedding.gpu_layers", SettingKind::Integer),
            advanced("embedding.ready_timeout_secs", SettingKind::Integer),
            advanced("embedding.custom_args", SettingKind::Text),
        ],
    },
    Section {
        title: "Search",
        table: "search",
        settings: &[
            basic("search.enabled", SettingKind::Toggle),
            basic("search.engines", SettingKind::List),
            advanced("search.command", SettingKind::Text),
            advanced("search.host", SettingKind::Text),
            advanced("search.port", SettingKind::Integer),
            advanced("search.ready_timeout_secs", SettingKind::Integer),
        ],
    },
    Section {
        title: "MCP",
        table: "mcp",
        settings: &[
            basic("mcp.enabled", SettingKind::Toggle),
            basic("mcp.servers", SettingKind::External),
        ],
    },
    Section {
        title: "Tray",
        table: "tray",
        settings: &[
            basic("tray.notifications.enabled", SettingKind::Toggle),
            basic("tray.notifications.auto_hide_ms", SettingKind::Integer),
            basic("tray.notifications.brief_when_away", SettingKind::Toggle),
            advanced("tray.icons.active", SettingKind::Text),
            advanced("tray.icons.sleeping", SettingKind::Text),
            advanced("tray.icons.listening", SettingKind::Text),
            advanced("tray.icons.generating", SettingKind::Text),
            advanced("tray.icons.disconnected", SettingKind::Text),
        ],
    },
];

/// How a setting is edited and shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SettingKind {
    Toggle,
    Integer,
    Decimal,
    Text,
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
}

#[derive(Debug)]
pub(crate) struct Section {
    pub title: &'static str,
    /// The table every key in the section sits under.
    pub table: &'static str,
    pub settings: &'static [Setting],
}

impl Section {
    /// `setting`'s key relative to the section's table.
    pub(crate) fn label(&self, setting: &Setting) -> &'static str {
        setting
            .key
            .strip_prefix(self.table)
            .and_then(|rest| rest.strip_prefix('.'))
            .unwrap_or(setting.key)
    }

    pub(crate) fn advanced_count(&self) -> usize {
        self.settings.iter().filter(|s| s.advanced).count()
    }
}

const fn basic(key: &'static str, kind: SettingKind) -> Setting {
    Setting {
        key,
        kind,
        advanced: false,
    }
}

const fn advanced(key: &'static str, kind: SettingKind) -> Setting {
    Setting {
        key,
        kind,
        advanced: true,
    }
}

#[cfg(test)]
mod tests;
