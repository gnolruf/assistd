//! The settings the panel lists, grouped into sections in display order and
//! sections into tabs. A feature's section heading is the checkbox that enables it.

const COMPOSITOR_TYPES: &[&str] = &["auto", "i3", "sway", "hyprland"];
const SANDBOX_MODES: &[&str] = &["auto", "bwrap", "none"];

pub(crate) const SECTIONS: &[Section] = &[
    section(
        Tab::General,
        "Hotkeys",
        &[
            basic("hotkeys.push_to_talk", "push-to-talk", SettingKind::Text),
            basic(
                "hotkeys.toggle_listening",
                "toggle wake-word listening",
                SettingKind::Text,
            ),
            basic(
                "hotkeys.toggle_speech",
                "toggle spoken replies",
                SettingKind::Text,
            ),
            basic(
                "hotkeys.skip_speech",
                "skip the current reply",
                SettingKind::Text,
            ),
        ],
    ),
    section(
        Tab::Model,
        "Model",
        &[
            basic("model.name", "model", SettingKind::Text),
            basic(
                "model.context_length",
                "context window (tokens)",
                SettingKind::Integer,
            ),
            basic(
                "model.gpu_layers",
                "layers offloaded to GPU",
                SettingKind::Integer,
            ),
            advanced(
                "model.server_binary",
                "llama-server binary",
                SettingKind::Text,
            ),
            advanced("model.host", "server address", SettingKind::Text),
            advanced("model.port", "server port", SettingKind::Integer),
            advanced(
                "model.ready_timeout_secs",
                "startup timeout (s)",
                SettingKind::Integer,
            ),
            advanced(
                "model.custom_args",
                "extra server arguments",
                SettingKind::Text,
            ),
        ],
    ),
    section(
        Tab::Model,
        "Chat",
        &[
            basic("chat.system_prompt", "system prompt", SettingKind::Prose),
            basic("chat.temperature", "temperature", SettingKind::Decimal),
            basic(
                "chat.max_response_tokens",
                "max reply length (tokens)",
                SettingKind::Integer,
            ),
            advanced(
                "chat.max_history_tokens",
                "history budget (tokens)",
                SettingKind::Integer,
            ),
            advanced(
                "chat.summary_target_tokens",
                "summary length (tokens)",
                SettingKind::Integer,
            ),
            advanced(
                "chat.max_reasoning_tokens",
                "max thinking (tokens)",
                SettingKind::Integer,
            ),
            advanced(
                "chat.reasoning_effort",
                "reasoning effort",
                SettingKind::Text,
            ),
            advanced(
                "chat.request_timeout_secs",
                "first-token timeout (s)",
                SettingKind::Integer,
            ),
            advanced(
                "chat.summary_temperature",
                "summary temperature",
                SettingKind::Decimal,
            ),
            advanced("chat.top_p", "top-p cutoff", SettingKind::Decimal),
            advanced("chat.top_k", "top-k cutoff", SettingKind::Integer),
            advanced("chat.min_p", "min-p cutoff", SettingKind::Decimal),
            advanced(
                "chat.presence_penalty",
                "presence penalty",
                SettingKind::Decimal,
            ),
        ],
    ),
    feature(
        Tab::Voice,
        "Voice input",
        enable("voice.enabled"),
        &[],
        &[
            basic("voice.mic_device", "microphone", SettingKind::Text),
            basic(
                "voice.transcription.model",
                "speech-to-text model",
                SettingKind::Text,
            ),
            basic(
                "voice.transcription.prefer_gpu",
                "prefer GPU if available",
                SettingKind::Toggle,
            ),
            advanced(
                "voice.transcription.threads",
                "CPU threads",
                SettingKind::Integer,
            ),
            advanced(
                "voice.transcription.beams",
                "decoding beams",
                SettingKind::Integer,
            ),
            advanced(
                "voice.transcription.vad_enabled",
                "enable silence trimming",
                SettingKind::Toggle,
            ),
            advanced(
                "voice.transcription.vad_model",
                "voice-detection model",
                SettingKind::Text,
            ),
            advanced(
                "voice.transcription.model_cache_dir",
                "model cache folder",
                SettingKind::Text,
            ),
        ],
    ),
    feature(
        Tab::Voice,
        "Wake word",
        enable("voice.wakeword.enabled"),
        &["voice.enabled"],
        &[
            basic(
                "voice.wakeword.start_on_launch",
                "listen from launch",
                SettingKind::Toggle,
            ),
            basic(
                "voice.wakeword.barge_in",
                "interrupt replies by talking",
                SettingKind::Toggle,
            ),
            basic(
                "voice.wakeword.follow_up_ms",
                "follow-up window (ms)",
                SettingKind::Integer,
            ),
            advanced(
                "voice.wakeword.silence_ms",
                "end-of-speech silence (ms)",
                SettingKind::Integer,
            ),
            advanced(
                "voice.wakeword.max_utterance_secs",
                "max utterance (s)",
                SettingKind::Integer,
            ),
            advanced(
                "voice.wakeword.command_timeout_ms",
                "wait for command (ms)",
                SettingKind::Integer,
            ),
            advanced(
                "voice.wakeword.match_similarity",
                "phrase match strictness",
                SettingKind::Decimal,
            ),
        ],
    ),
    feature(
        Tab::Voice,
        "Spoken replies",
        enable("voice.synthesis.enabled"),
        &[],
        &[
            basic("voice.synthesis.voice", "voice", SettingKind::Text),
            basic(
                "voice.synthesis.length_scale",
                "speaking pace",
                SettingKind::Decimal,
            ),
            basic(
                "voice.synthesis.output_device",
                "speaker",
                SettingKind::Text,
            ),
            advanced(
                "voice.synthesis.binary_path",
                "piper binary",
                SettingKind::Text,
            ),
            advanced(
                "voice.synthesis.model_cache_dir",
                "voice cache folder",
                SettingKind::Text,
            ),
            advanced(
                "voice.synthesis.espeak_data_dir",
                "espeak-ng data folder",
                SettingKind::Text,
            ),
            advanced(
                "voice.synthesis.deadline_secs",
                "per-sentence timeout (s)",
                SettingKind::Integer,
            ),
            advanced(
                "voice.synthesis.max_sentence_chars",
                "max sentence length (chars)",
                SettingKind::Integer,
            ),
            advanced(
                "voice.synthesis.partial_flush_ms",
                "speak partial sentence after (ms)",
                SettingKind::Integer,
            ),
            advanced("voice.synthesis.use_cuda", "use CUDA", SettingKind::Toggle),
        ],
    ),
    section(
        Tab::General,
        "Compositor",
        &[basic(
            "compositor.type",
            "compositor",
            SettingKind::Choice(COMPOSITOR_TYPES),
        )],
    ),
    section(
        Tab::General,
        "Sleep",
        &[
            basic(
                "sleep.idle_to_drowsy_mins",
                "drowse after idle (min)",
                SettingKind::Integer,
            ),
            basic(
                "sleep.idle_to_sleep_mins",
                "sleep after idle (min)",
                SettingKind::Integer,
            ),
        ],
    ),
    feature(
        Tab::General,
        "GPU monitor",
        enable("sleep.gpu_monitor_enabled"),
        &[],
        &[
            basic(
                "sleep.gpu_poll_secs",
                "check interval (s)",
                SettingKind::Integer,
            ),
            basic(
                "sleep.gpu_headroom_mb",
                "free VRAM to keep (MiB)",
                SettingKind::Integer,
            ),
            basic(
                "sleep.gpu_auto_wake",
                "wake when VRAM frees up",
                SettingKind::Toggle,
            ),
        ],
    ),
    section(
        Tab::Tools,
        "Tools",
        &[
            basic(
                "tools.bash.sandbox",
                "sandbox",
                SettingKind::Choice(SANDBOX_MODES),
            ),
            basic(
                "tools.bash.timeout_secs",
                "command timeout (s)",
                SettingKind::Integer,
            ),
            basic(
                "tools.write.writable_paths",
                "writable folders",
                SettingKind::List(ListItem::Text),
            ),
            advanced(
                "tools.bash.denylist",
                "blocked commands",
                SettingKind::List(ListItem::Text),
            ),
            advanced(
                "tools.bash.allowed_programs",
                "programs run without asking",
                SettingKind::List(ListItem::Text),
            ),
            advanced(
                "tools.bash.destructive_patterns",
                "commands that always ask",
                SettingKind::List(ListItem::Text),
            ),
            advanced(
                "tools.bash.bwrap_extra_args",
                "extra bubblewrap arguments",
                SettingKind::List(ListItem::Text),
            ),
            advanced("tools.scratch.dir", "scratch folder", SettingKind::Text),
            advanced(
                "tools.scratch.retention_days",
                "keep scratch files (days)",
                SettingKind::Integer,
            ),
            advanced(
                "tools.output.max_lines",
                "max result lines",
                SettingKind::Integer,
            ),
            advanced(
                "tools.output.max_kb",
                "max result size (KB)",
                SettingKind::Integer,
            ),
            advanced(
                "tools.output.overflow_dir",
                "overflow folder",
                SettingKind::Text,
            ),
        ],
    ),
    feature(
        Tab::Model,
        "Embedding",
        enable("embedding.enabled"),
        &[],
        &[
            basic("embedding.model", "embedding model", SettingKind::Text),
            basic(
                "embedding.top_k",
                "memories recalled per query",
                SettingKind::Integer,
            ),
            basic(
                "embedding.auto_inject",
                "add recalled memories to queries",
                SettingKind::Toggle,
            ),
            advanced(
                "embedding.server_binary",
                "llama-server binary",
                SettingKind::Text,
            ),
            advanced("embedding.host", "server address", SettingKind::Text),
            advanced("embedding.port", "server port", SettingKind::Integer),
            advanced(
                "embedding.gpu_layers",
                "layers offloaded to GPU",
                SettingKind::Integer,
            ),
            advanced(
                "embedding.ready_timeout_secs",
                "startup timeout (s)",
                SettingKind::Integer,
            ),
            advanced(
                "embedding.custom_args",
                "extra server arguments",
                SettingKind::Text,
            ),
        ],
    ),
    feature(
        Tab::Tools,
        "Search",
        enable("search.enabled"),
        &[],
        &[
            basic(
                "search.engines",
                "search engines",
                SettingKind::List(ListItem::SearchEngine),
            ),
            basic("search.command", "SearXNG command", SettingKind::Text),
            advanced("search.host", "server address", SettingKind::Text),
            advanced("search.port", "server port", SettingKind::Integer),
            advanced(
                "search.ready_timeout_secs",
                "startup timeout (s)",
                SettingKind::Integer,
            ),
        ],
    ),
    feature(
        Tab::Tools,
        "MCP",
        enable("mcp.enabled"),
        &[],
        &[basic("mcp.servers", "servers", SettingKind::External)],
    ),
    section(
        Tab::General,
        "Memory",
        &[basic(
            "memory.enabled",
            "remember facts and past chats",
            SettingKind::Toggle,
        )],
    ),
    feature(
        Tab::General,
        "Notifications",
        enable("notifications.enabled"),
        &[],
        &[
            basic(
                "notifications.auto_hide_ms",
                "hide after idle (ms)",
                SettingKind::Integer,
            ),
            basic(
                "notifications.brief_when_away",
                "brief voice replies when away",
                SettingKind::Toggle,
            ),
        ],
    ),
    section(
        Tab::General,
        "Tray icons",
        &[
            basic("notifications.icons.active", "awake", SettingKind::Text),
            basic(
                "notifications.icons.sleeping",
                "sleeping",
                SettingKind::Text,
            ),
            basic(
                "notifications.icons.listening",
                "listening",
                SettingKind::Text,
            ),
            advanced(
                "notifications.icons.generating",
                "generating",
                SettingKind::Text,
            ),
            advanced(
                "notifications.icons.disconnected",
                "disconnected",
                SettingKind::Text,
            ),
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

/// One config key, by its dotted path.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Setting {
    pub key: &'static str,
    /// What the key is for, in words, shown in its place.
    pub label: &'static str,
    pub kind: SettingKind,
    /// Listed only while its section shows advanced settings.
    pub advanced: bool,
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
        self.settings.iter().filter(|s| s.advanced).count()
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
    basic(key, "enabled", SettingKind::Toggle)
}

const fn basic(key: &'static str, label: &'static str, kind: SettingKind) -> Setting {
    Setting {
        key,
        label,
        kind,
        advanced: false,
    }
}

const fn advanced(key: &'static str, label: &'static str, kind: SettingKind) -> Setting {
    Setting {
        key,
        label,
        kind,
        advanced: true,
    }
}

#[cfg(test)]
mod tests;
