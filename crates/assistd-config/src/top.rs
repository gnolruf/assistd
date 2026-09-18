use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::chat::ChatConfig;
use crate::compositor::CompositorConfig;
use crate::daemon::DaemonConfig;
use crate::embedding::EmbeddingConfig;
use crate::errors::ConfigError;
use crate::llama::LlamaServerConfig;
use crate::mcp::McpConfig;
use crate::memory::MemoryConfig;
use crate::model::ModelConfig;
use crate::presence::PresenceConfig;
use crate::sleep::SleepConfig;
use crate::timeouts::TimeoutsConfig;
use crate::tools::ToolsConfig;
use crate::tray::TrayConfig;
use crate::voice::VoiceConfig;

/// Top-level assistd configuration, deserialized from `config.toml`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub model: ModelConfig,
    pub llama_server: LlamaServerConfig,
    pub chat: ChatConfig,
    pub voice: VoiceConfig,
    pub compositor: CompositorConfig,
    pub sleep: SleepConfig,
    pub presence: PresenceConfig,
    pub daemon: DaemonConfig,
    pub tools: ToolsConfig,
    pub memory: MemoryConfig,
    pub embedding: EmbeddingConfig,
    pub mcp: McpConfig,
    #[serde(skip)]
    pub timeouts: TimeoutsConfig,
    pub tray: TrayConfig,
}

impl Config {
    /// Validates the cross-field and format constraints that the field
    /// types can't express on their own. Returns every problem found,
    /// not just the first.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let mut errors = Vec::new();

        if self.model.name.is_empty() {
            errors.push("model.name must not be empty".into());
        }
        if self.llama_server.binary_path.as_os_str().is_empty() {
            errors.push("llama_server.binary_path must not be empty".into());
        }

        if self.chat.max_history_tokens >= self.model.context_length {
            errors.push(
                "chat.max_history_tokens must be strictly less than model.context_length".into(),
            );
        }
        if self.chat.summary_target_tokens >= self.chat.max_history_tokens {
            errors.push(
                "chat.summary_target_tokens must be strictly less than chat.max_history_tokens"
                    .into(),
            );
        }
        if self.chat.max_response_tokens >= self.model.context_length {
            errors.push(
                "chat.max_response_tokens must be strictly less than model.context_length".into(),
            );
        }
        if !(0.0..=2.0).contains(&self.chat.temperature) || self.chat.temperature.is_nan() {
            errors.push("chat.temperature must be in the range 0.0..=2.0".into());
        }
        if !(0.0..=2.0).contains(&self.chat.summary_temperature)
            || self.chat.summary_temperature.is_nan()
        {
            errors.push("chat.summary_temperature must be in the range 0.0..=2.0".into());
        }
        if let Some(tp) = self.chat.top_p
            && (!(0.0..=1.0).contains(&tp) || tp.is_nan())
        {
            errors.push("chat.top_p must be in the range 0.0..=1.0".into());
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
            let t = &self.voice.transcription;
            if !is_valid_hf_id(&t.model) {
                errors.push(
                    "voice.transcription.model must be of the form \
                     '<owner>/<repo>:<file>'"
                        .into(),
                );
            }
            if t.vad_enabled && !is_valid_hf_id(&t.vad_model) {
                errors.push(
                    "voice.transcription.vad_model must be of the form \
                     '<owner>/<repo>:<file>'"
                        .into(),
                );
            }
        }

        if self.voice.synthesis.enabled {
            let s = &self.voice.synthesis;
            if s.binary_path.as_os_str().is_empty() {
                errors.push(
                    "voice.synthesis.binary_path must not be empty when synthesis is enabled"
                        .into(),
                );
            }
            if !is_valid_hf_id(&s.voice) {
                errors.push(
                    "voice.synthesis.voice must be of the form '<owner>/<repo>:<file>'".into(),
                );
            }
            if !s.length_scale.is_finite() || s.length_scale <= 0.0 {
                errors
                    .push("voice.synthesis.length_scale must be a positive, finite number".into());
            }
            if s.max_sentence_chars.get() < 50 {
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

        if self.tools.output.overflow_dir.as_os_str().is_empty() {
            errors.push("tools.output.overflow_dir must not be empty".into());
        }
        if self.tools.write.writable_paths.is_empty() {
            errors.push(
                "tools.write.writable_paths must not be empty (the write command would be unusable)"
                    .into(),
            );
        }

        if self.memory.enabled && self.memory.db_path.as_os_str().is_empty() {
            errors.push("memory.db_path must not be empty when memory.enabled".into());
        }

        if self.embedding.enabled {
            if !is_valid_hf_id(&self.embedding.model) {
                errors.push("embedding.model must be of the form '<owner>/<repo>:<file>'".into());
            }
            if self.embedding.port == self.llama_server.port {
                errors.push(
                    "embedding.port must differ from llama_server.port (the chat server)".into(),
                );
            }
        }

        if self.mcp.enabled {
            use std::collections::HashSet;
            let mut seen: HashSet<&str> = HashSet::new();
            for (i, s) in self.mcp.servers.iter().enumerate() {
                let name = s.name();
                if name.is_empty() {
                    errors.push(format!("mcp.servers[{i}].name must not be empty"));
                } else if !seen.insert(name) {
                    errors.push(format!(
                        "mcp.servers[{i}].name '{name}' is duplicated; names must be unique"
                    ));
                }
                if !name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
                {
                    errors.push(format!(
                        "mcp.servers[{i}].name must use only ASCII letters, digits, '_' or '-' \
                         (becomes part of the LLM-visible tool name `mcp__<name>__<tool>`)"
                    ));
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
