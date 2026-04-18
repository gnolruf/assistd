use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::agent::AgentConfig;
use crate::chat::ChatConfig;
use crate::compositor::CompositorConfig;
use crate::daemon::DaemonConfig;
use crate::errors::ConfigError;
use crate::llama::LlamaServerConfig;
use crate::model::ModelConfig;
use crate::presence::PresenceConfig;
use crate::remote::RemoteConfig;
use crate::sleep::SleepConfig;
use crate::tools::ToolsConfig;
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

        if self.voice.enabled && self.voice.hotkey.is_empty() {
            errors.push("voice.hotkey must not be empty when voice is enabled".into());
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

        if self.agent.max_iterations == 0 {
            errors.push("agent.max_iterations must be greater than 0".into());
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
