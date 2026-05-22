use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::agent::AgentConfig;
use crate::chat::ChatConfig;
use crate::compositor::CompositorConfig;
use crate::daemon::DaemonConfig;
use crate::embedding::EmbeddingConfig;
use crate::errors::ConfigError;
use crate::llama::LlamaServerConfig;
use crate::mcp::{McpConfig, McpTransport};
use crate::memory::MemoryConfig;
use crate::model::ModelConfig;
use crate::presence::PresenceConfig;
use crate::remote::RemoteConfig;
use crate::sleep::SleepConfig;
use crate::timeouts::TimeoutsConfig;
use crate::tools::ToolsConfig;
use crate::tray::TrayConfig;
use crate::voice::VoiceConfig;

/// Top-level assistd configuration, deserialized from `config.toml`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Config {
    pub model: ModelConfig,
    pub llama_server: LlamaServerConfig,
    pub chat: ChatConfig,
    pub voice: VoiceConfig,
    pub compositor: CompositorConfig,
    pub sleep: SleepConfig,
    pub remote: RemoteConfig,
    #[serde(default)]
    pub presence: PresenceConfig,
    #[serde(default)]
    pub daemon: DaemonConfig,
    #[serde(default)]
    pub tools: ToolsConfig,
    #[serde(default)]
    pub agent: AgentConfig,
    #[serde(default)]
    pub memory: MemoryConfig,
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    #[serde(default)]
    pub mcp: McpConfig,
    #[serde(default)]
    pub timeouts: TimeoutsConfig,
    #[serde(default)]
    pub tray: TrayConfig,
}

impl Config {
    /// Validates all config fields. Returns every problem found, not just the first.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let mut errors = Vec::new();

        if self.model.name.is_empty() {
            errors.push("model.name must not be empty".into());
        }
        if self.model.context_length == 0 {
            errors.push("model.context_length must be greater than 0".into());
        }

        if self.llama_server.binary_path.is_empty() {
            errors.push("llama_server.binary_path must not be empty".into());
        }
        if self.llama_server.host.is_empty() {
            errors.push("llama_server.host must not be empty".into());
        }
        if self.llama_server.port == 0 {
            errors.push("llama_server.port must not be 0".into());
        }

        if self.chat.max_history_tokens == 0 {
            errors.push("chat.max_history_tokens must be greater than 0".into());
        }
        if self.model.context_length > 0
            && self.chat.max_history_tokens >= self.model.context_length
        {
            errors.push(
                "chat.max_history_tokens must be strictly less than model.context_length".into(),
            );
        }
        if self.chat.summary_target_tokens == 0 {
            errors.push("chat.summary_target_tokens must be greater than 0".into());
        }
        if self.chat.summary_target_tokens >= self.chat.max_history_tokens
            && self.chat.max_history_tokens > 0
        {
            errors.push(
                "chat.summary_target_tokens must be strictly less than chat.max_history_tokens"
                    .into(),
            );
        }
        if self.chat.max_response_tokens == 0 {
            errors.push("chat.max_response_tokens must be greater than 0".into());
        }
        if self.model.context_length > 0
            && self.chat.max_response_tokens >= self.model.context_length
        {
            errors.push(
                "chat.max_response_tokens must be strictly less than model.context_length".into(),
            );
        }
        if self.chat.max_summary_tokens == 0 {
            errors.push("chat.max_summary_tokens must be greater than 0".into());
        }
        if self.model.context_length > 0 && self.chat.max_summary_tokens > self.model.context_length
        {
            errors.push("chat.max_summary_tokens must not exceed model.context_length".into());
        }
        if self.chat.request_timeout_secs == 0 {
            errors.push("chat.request_timeout_secs must be greater than 0".into());
        }
        if !(0.0..=2.0).contains(&self.chat.temperature) || self.chat.temperature.is_nan() {
            errors.push("chat.temperature must be in the range 0.0..=2.0".into());
        }
        if !(0.0..=2.0).contains(&self.chat.summary_temperature)
            || self.chat.summary_temperature.is_nan()
        {
            errors.push("chat.summary_temperature must be in the range 0.0..=2.0".into());
        }
        if self.chat.preserve_recent_turns == 0 {
            errors.push("chat.preserve_recent_turns must be at least 1".into());
        }
        if let Some(tp) = self.chat.top_p
            && (!(0.0..=1.0).contains(&tp) || tp.is_nan())
        {
            errors.push("chat.top_p must be in the range 0.0..=1.0".into());
        }
        if let Some(tk) = self.chat.top_k
            && tk == 0
        {
            errors.push("chat.top_k must be greater than 0".into());
        }
        if let Some(mp) = self.chat.min_p
            && (!(0.0..=1.0).contains(&mp) || mp.is_nan())
        {
            errors.push("chat.min_p must be in the range 0.0..=1.0".into());
        }
        if let Some(pp) = self.chat.presence_penalty
            && (!(-2.0..=2.0).contains(&pp) || pp.is_nan())
        {
            errors.push("chat.presence_penalty must be in the range -2.0..=2.0".into());
        }

        if self.voice.enabled {
            // hotkey may be empty; the user might prefer the IPC-only
            // PTT pathway (i3 bindsym → `assistd ptt-start/stop`).
            if self.voice.max_recording_secs == 0 {
                errors.push(
                    "voice.max_recording_secs must be greater than 0 when voice is enabled".into(),
                );
            }
            let t = &self.voice.transcription;
            if t.model.trim().is_empty() {
                errors.push("voice.transcription.model must not be empty".into());
            } else if !is_valid_hf_id(&t.model) {
                errors.push(
                    "voice.transcription.model must be of the form \
                     '<owner>/<repo>:<file>'"
                        .into(),
                );
            }
            if t.vad_enabled {
                if t.vad_model.trim().is_empty() {
                    errors.push(
                        "voice.transcription.vad_model must not be empty when vad_enabled".into(),
                    );
                } else if !is_valid_hf_id(&t.vad_model) {
                    errors.push(
                        "voice.transcription.vad_model must be of the form \
                         '<owner>/<repo>:<file>'"
                            .into(),
                    );
                }
            }
            if t.beams == 0 {
                errors.push("voice.transcription.beams must be at least 1".into());
            }
            if !t.vad_silence_secs.is_finite() || t.vad_silence_secs < 0.0 {
                errors.push(
                    "voice.transcription.vad_silence_secs must be a non-negative, finite number"
                        .into(),
                );
            }
            if let Some(th) = t.threads
                && th == 0
            {
                errors.push("voice.transcription.threads must be greater than 0 when set".into());
            }
            if t.gpu_busy_timeout_ms > 10_000 {
                errors.push(
                    "voice.transcription.gpu_busy_timeout_ms must not exceed 10000 (10 s)".into(),
                );
            }

            let c = &self.voice.continuous;
            if c.enabled {
                if c.silence_ms == 0 {
                    errors.push(
                        "voice.continuous.silence_ms must be greater than 0 when enabled".into(),
                    );
                }
                if c.max_utterance_secs == 0 {
                    errors.push(
                        "voice.continuous.max_utterance_secs must be greater than 0 when enabled"
                            .into(),
                    );
                }
                if c.min_utterance_ms >= c.max_utterance_secs.saturating_mul(1000) {
                    errors.push(
                        "voice.continuous.min_utterance_ms must be less than max_utterance_secs * 1000"
                            .into(),
                    );
                }
                if c.aggressiveness > 3 {
                    errors
                        .push("voice.continuous.aggressiveness must be in the range 0..=3".into());
                }
            }
        }

        // Synthesis (Piper TTS) is gated independently of voice.enabled:
        // a user might want LLM responses spoken aloud even with no
        // microphone available.
        if self.voice.synthesis.enabled {
            let s = &self.voice.synthesis;
            if s.binary_path.trim().is_empty() {
                errors.push(
                    "voice.synthesis.binary_path must not be empty when synthesis is enabled"
                        .into(),
                );
            }
            if s.voice.trim().is_empty() {
                errors.push("voice.synthesis.voice must not be empty".into());
            } else if !is_valid_hf_id(&s.voice) {
                errors.push(
                    "voice.synthesis.voice must be of the form '<owner>/<repo>:<file>'".into(),
                );
            }
            if !s.length_scale.is_finite() || s.length_scale <= 0.0 {
                errors
                    .push("voice.synthesis.length_scale must be a positive, finite number".into());
            }
            if !s.noise_scale.is_finite() || !(0.0..=5.0).contains(&s.noise_scale) {
                errors.push("voice.synthesis.noise_scale must be in the range 0.0..=5.0".into());
            }
            if !s.noise_w.is_finite() || !(0.0..=5.0).contains(&s.noise_w) {
                errors.push("voice.synthesis.noise_w must be in the range 0.0..=5.0".into());
            }
            if !s.sentence_silence_secs.is_finite() || s.sentence_silence_secs < 0.0 {
                errors.push(
                    "voice.synthesis.sentence_silence_secs must be a non-negative, finite number"
                        .into(),
                );
            }
            if s.deadline_secs == 0 {
                errors.push("voice.synthesis.deadline_secs must be greater than 0".into());
            }
            if s.max_sentence_chars < 50 {
                errors.push("voice.synthesis.max_sentence_chars must be at least 50".into());
            }
        }

        if self.sleep.idle_to_drowsy_mins > 0
            && self.sleep.idle_to_sleep_mins > 0
            && self.sleep.idle_to_sleep_mins <= self.sleep.idle_to_drowsy_mins
        {
            errors.push(
                "sleep.idle_to_sleep_mins must be greater than sleep.idle_to_drowsy_mins \
                 (set either to 0 to disable that transition)"
                    .into(),
            );
        }

        if self.sleep.gpu_monitor_enabled {
            if self.sleep.gpu_poll_secs == 0 {
                errors.push(
                    "sleep.gpu_poll_secs must be greater than 0 when gpu_monitor_enabled".into(),
                );
            }
            if self.sleep.gpu_vram_threshold_mb == 0 {
                errors.push(
                    "sleep.gpu_vram_threshold_mb must be greater than 0 when gpu_monitor_enabled"
                        .into(),
                );
            }
        }

        if self.remote.enabled {
            if self.remote.port == 0 {
                errors.push("remote.port must not be 0 when remote access is enabled".into());
            }
            if self.remote.bind_address.is_empty() {
                errors.push(
                    "remote.bind_address must not be empty when remote access is enabled".into(),
                );
            }
        }

        if self.tools.output.max_lines == 0 {
            errors.push("tools.output.max_lines must be greater than 0".into());
        }
        if self.tools.output.max_kb == 0 {
            errors.push("tools.output.max_kb must be greater than 0".into());
        }
        if self.tools.output.overflow_dir.is_empty() {
            errors.push("tools.output.overflow_dir must not be empty".into());
        }

        if self.tools.bash.timeout_secs == 0 {
            errors.push("tools.bash.timeout_secs must be greater than 0".into());
        }
        if self.tools.write.writable_paths.is_empty() {
            errors.push(
                "tools.write.writable_paths must not be empty (the write command would be unusable)"
                    .into(),
            );
        }
        if self.tools.screenshot.timeout_secs == 0 {
            errors.push("tools.screenshot.timeout_secs must be greater than 0".into());
        }

        if self.agent.max_iterations == 0 {
            errors.push("agent.max_iterations must be greater than 0".into());
        }

        if self.memory.enabled && self.memory.db_path.is_empty() {
            errors.push("memory.db_path must not be empty when memory.enabled".into());
        }
        if self.memory.retention_days > 36500 {
            errors.push(
                "memory.retention_days exceeds 100 years; check for a typo (use 0 for forever)"
                    .into(),
            );
        }

        if self.embedding.enabled {
            if self.embedding.model.trim().is_empty() {
                errors.push("embedding.model must not be empty when embedding.enabled".into());
            } else if !is_valid_hf_id(&self.embedding.model) {
                errors.push("embedding.model must be of the form '<owner>/<repo>:<file>'".into());
            }
            if self.embedding.host.is_empty() {
                errors.push("embedding.host must not be empty when embedding.enabled".into());
            }
            if self.embedding.port == 0 {
                errors.push("embedding.port must not be 0".into());
            }
            if self.embedding.port == self.llama_server.port {
                errors.push(
                    "embedding.port must differ from llama_server.port (the chat server)".into(),
                );
            }
            if self.remote.enabled && self.embedding.port == self.remote.port {
                errors.push(
                    "embedding.port must differ from remote.port when remote access is enabled"
                        .into(),
                );
            }
            if self.embedding.top_k == 0 {
                errors.push("embedding.top_k must be greater than 0".into());
            }
            if self.embedding.chunk_chars == 0 {
                errors.push("embedding.chunk_chars must be greater than 0".into());
            }
            if self.embedding.chunk_overlap_chars >= self.embedding.chunk_chars {
                errors.push(
                    "embedding.chunk_overlap_chars must be strictly less than embedding.chunk_chars"
                        .into(),
                );
            }
            if self.embedding.ready_timeout_secs == 0 {
                errors.push("embedding.ready_timeout_secs must be greater than 0".into());
            }
            if self.embedding.request_timeout_secs == 0 {
                errors.push("embedding.request_timeout_secs must be greater than 0".into());
            }
        }

        if self.mcp.enabled {
            use std::collections::HashSet;
            let mut seen: HashSet<&str> = HashSet::new();
            for (i, s) in self.mcp.servers.iter().enumerate() {
                let trimmed = s.name.trim();
                if trimmed.is_empty() {
                    errors.push(format!("mcp.servers[{i}].name must not be empty"));
                } else if !seen.insert(trimmed) {
                    errors.push(format!(
                        "mcp.servers[{i}].name '{trimmed}' is duplicated; names must be unique"
                    ));
                }
                if !trimmed
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
                {
                    errors.push(format!(
                        "mcp.servers[{i}].name must use only ASCII letters, digits, '_' or '-' \
                         (becomes part of the LLM-visible tool name `mcp__<name>__<tool>`)"
                    ));
                }
                match s.transport {
                    McpTransport::Stdio => {
                        if s.command.as_deref().is_none_or(|c| c.trim().is_empty()) {
                            errors.push(format!(
                                "mcp.servers[{i}] (stdio) must set non-empty `command`"
                            ));
                        }
                        if s.url.is_some() {
                            errors.push(format!("mcp.servers[{i}] (stdio) must not set `url`"));
                        }
                    }
                    McpTransport::Sse => {
                        match s.url.as_deref() {
                            None => errors.push(format!("mcp.servers[{i}] (sse) must set `url`")),
                            Some(u) => {
                                if u.trim().is_empty() {
                                    errors.push(format!(
                                        "mcp.servers[{i}] (sse) `url` must not be empty"
                                    ));
                                } else if !(u.starts_with("http://") || u.starts_with("https://")) {
                                    errors.push(format!(
                                        "mcp.servers[{i}].url must start with http:// or https://"
                                    ));
                                }
                            }
                        }
                        if s.command.is_some() {
                            errors.push(format!("mcp.servers[{i}] (sse) must not set `command`"));
                        }
                    }
                }
                if s.request_timeout_secs == 0 {
                    errors.push(format!("mcp.servers[{i}].request_timeout_secs must be > 0"));
                }
                if matches!(s.transport, McpTransport::Sse) {
                    if s.sse_read_timeout_secs == 0 {
                        errors.push(format!(
                            "mcp.servers[{i}].sse_read_timeout_secs must be > 0"
                        ));
                    }
                    if s.sse_ping_interval_secs == 0 {
                        errors.push(format!(
                            "mcp.servers[{i}].sse_ping_interval_secs must be > 0"
                        ));
                    }
                }
            }
        }

        if self.tray.popup.enabled {
            let p = &self.tray.popup;
            if !(100..=1200).contains(&p.width) {
                errors.push("tray.popup.width must be in the range 100..=1200".into());
            }
            if !(60..=800).contains(&p.height) {
                errors.push("tray.popup.height must be in the range 60..=800".into());
            }
            if !(500..=60_000).contains(&p.auto_hide_ms) {
                errors.push("tray.popup.auto_hide_ms must be in the range 500..=60000".into());
            }
            if !(500..=60_000).contains(&p.listen_auto_hide_ms) {
                errors
                    .push("tray.popup.listen_auto_hide_ms must be in the range 500..=60000".into());
            }
            if !(1..=10_000).contains(&p.truncate_chars) {
                errors.push("tray.popup.truncate_chars must be in the range 1..=10000".into());
            }
        }

        let t = &self.timeouts;
        if t.presence_sleep_secs == 0 {
            errors.push("timeouts.presence_sleep_secs must be greater than 0".into());
        }
        if t.presence_drowse_secs == 0 {
            errors.push("timeouts.presence_drowse_secs must be greater than 0".into());
        }
        if t.dispatch_envelope_secs == 0 {
            errors.push("timeouts.dispatch_envelope_secs must be greater than 0".into());
        }
        if t.stream_inactivity_secs == 0 {
            errors.push("timeouts.stream_inactivity_secs must be greater than 0".into());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(ConfigError::Validation(errors))
        }
    }
}

impl Config {
    /// Default config file path, respecting `$XDG_CONFIG_HOME`:
    /// `$XDG_CONFIG_HOME/assistd/config.toml` or `$HOME/.config/assistd/config.toml`.
    pub fn default_path() -> Result<PathBuf, ConfigError> {
        let config_dir = match std::env::var_os("XDG_CONFIG_HOME") {
            Some(dir) => PathBuf::from(dir),
            None => {
                let home = std::env::var("HOME").map_err(|_| ConfigError::HomeNotSet)?;
                PathBuf::from(home).join(".config")
            }
        };
        Ok(config_dir.join("assistd/config.toml"))
    }

    /// Loads and deserializes a config from the given TOML file.
    pub fn load_from_file(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        toml::from_str(&content).map_err(|source| ConfigError::Parse {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Writes the default config to `path`. Errors if the file already exists.
    pub fn write_default(path: &Path) -> Result<(), ConfigError> {
        if path.exists() {
            return Err(ConfigError::AlreadyExists(path.to_path_buf()));
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| ConfigError::CreateDir {
                path: parent.to_path_buf(),
                source,
            })?;
        }

        let toml_string = toml::to_string_pretty(&Config::default())?;

        let content = format!(
            "# assistd configuration file\n\
             # Generated by `assistd init-config`\n\n\
             {toml_string}"
        );

        std::fs::write(path, content).map_err(|source| ConfigError::Write {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(())
    }
}

fn is_valid_hf_id(s: &str) -> bool {
    let Some((repo, file)) = s.split_once(':') else {
        return false;
    };
    if file.is_empty() || file.contains(':') {
        return false;
    }
    let Some((owner, name)) = repo.split_once('/') else {
        return false;
    };
    !owner.is_empty() && !name.is_empty()
}
