use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Top-level assistd configuration, deserialized from `config.toml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Config {
    pub model: ModelConfig,
    pub voice: VoiceConfig,
    pub compositor: CompositorConfig,
    pub sleep: SleepConfig,
    pub remote: RemoteConfig,
}

/// Local model settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    /// Filesystem path to the GGUF model file.
    pub path: String,
    /// VRAM budget in megabytes.
    pub vram_budget_mb: u64,
    /// Context window length in tokens.
    pub context_length: u32,
}

/// Voice input settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VoiceConfig {
    /// Whether voice input is enabled.
    pub enabled: bool,
    /// ALSA/PulseAudio device name. `None` = system default.
    #[serde(default)]
    pub mic_device: Option<String>,
    /// Hotkey to activate voice input (e.g. "Super+V").
    pub hotkey: String,
}

/// Supported tiling compositors.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum CompositorType {
    I3,
    Sway,
    Hyprland,
}

/// Compositor integration settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompositorConfig {
    /// Which compositor to integrate with.
    #[serde(rename = "type")]
    pub compositor_type: CompositorType,
}

/// Sleep/idle policy settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SleepConfig {
    /// Idle timeout in seconds before the sleep action fires.
    pub idle_timeout_secs: u64,
    /// Whether to suspend the machine (true) or just deactivate (false).
    pub suspend: bool,
}

/// Remote access API settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RemoteConfig {
    /// Whether the remote access API is enabled.
    pub enabled: bool,
    /// IP address to bind to.
    pub bind_address: String,
    /// TCP port to listen on.
    pub port: u16,
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

impl Default for Config {
    fn default() -> Self {
        Self {
            model: ModelConfig {
                path: "/usr/share/models/default.gguf".to_string(),
                vram_budget_mb: 4096,
                context_length: 8192,
            },
            voice: VoiceConfig {
                enabled: false,
                mic_device: None,
                hotkey: "Super+V".to_string(),
            },
            compositor: CompositorConfig {
                compositor_type: CompositorType::Sway,
            },
            sleep: SleepConfig {
                idle_timeout_secs: 300,
                suspend: false,
            },
            remote: RemoteConfig {
                enabled: false,
                bind_address: "127.0.0.1".to_string(),
                port: 8384,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced while loading, writing, or validating configuration.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("HOME environment variable not set")]
    HomeNotSet,

    #[error("failed to read config file {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse config file {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },

    #[error("failed to serialize default config: {0}")]
    Serialize(#[from] toml::ser::Error),

    #[error("failed to create directory {path}: {source}")]
    CreateDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to write config file {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("config file already exists at {0} (not overwriting)")]
    AlreadyExists(PathBuf),

    #[error("{}", format_validation_errors(.0))]
    Validation(Vec<String>),
}

fn format_validation_errors(errors: &[String]) -> String {
    let mut s = format!("configuration has {} error(s):", errors.len());
    for (i, e) in errors.iter().enumerate() {
        s.push_str(&format!("\n  {}: {}", i + 1, e));
    }
    s
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

impl Config {
    /// Validates all config fields. Returns every problem found, not just the first.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let mut errors = Vec::new();

        if self.model.path.is_empty() {
            errors.push("model.path must not be empty".into());
        }
        if self.model.vram_budget_mb == 0 {
            errors.push("model.vram_budget_mb must be greater than 0".into());
        }
        if self.model.context_length == 0 {
            errors.push("model.context_length must be greater than 0".into());
        }

        if self.voice.enabled && self.voice.hotkey.is_empty() {
            errors.push("voice.hotkey must not be empty when voice is enabled".into());
        }

        if self.sleep.idle_timeout_secs == 0 {
            errors.push("sleep.idle_timeout_secs must be greater than 0".into());
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

        if errors.is_empty() {
            Ok(())
        } else {
            Err(ConfigError::Validation(errors))
        }
    }
}

// ---------------------------------------------------------------------------
// Load / Save
// ---------------------------------------------------------------------------

impl Config {
    /// Default config file path: `$HOME/.config/assistd/config.toml`.
    pub fn default_path() -> Result<PathBuf, ConfigError> {
        let home = std::env::var("HOME").map_err(|_| ConfigError::HomeNotSet)?;
        Ok(PathBuf::from(home).join(".config/assistd/config.toml"))
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_round_trips() {
        let config = Config::default();
        let serialized = toml::to_string_pretty(&config).expect("serialize");
        let deserialized: Config = toml::from_str(&serialized).expect("deserialize");
        assert_eq!(config, deserialized);
    }

    #[test]
    fn default_config_validates() {
        Config::default()
            .validate()
            .expect("default config should be valid");
    }

    fn validation_errors(err: ConfigError) -> Vec<String> {
        match err {
            ConfigError::Validation(errs) => errs,
            other => panic!("expected ConfigError::Validation, got {other:?}"),
        }
    }

    #[test]
    fn validation_catches_zero_port() {
        let mut config = Config::default();
        config.remote.enabled = true;
        config.remote.port = 0;
        let errs = validation_errors(config.validate().unwrap_err());
        assert!(errs.iter().any(|e| e.contains("port")));
    }

    #[test]
    fn validation_catches_empty_model_path() {
        let mut config = Config::default();
        config.model.path = String::new();
        let errs = validation_errors(config.validate().unwrap_err());
        assert!(errs.iter().any(|e| e.contains("model.path")));
    }

    #[test]
    fn validation_collects_multiple_errors() {
        let mut config = Config::default();
        config.model.path = String::new();
        config.model.vram_budget_mb = 0;
        config.sleep.idle_timeout_secs = 0;
        let errs = validation_errors(config.validate().unwrap_err());
        assert_eq!(errs.len(), 3);
    }

    #[test]
    fn unknown_compositor_type_rejected() {
        let toml_str = r#"
[model]
path = "/some/model.gguf"
vram_budget_mb = 4096
context_length = 8192

[voice]
enabled = false
hotkey = "Super+V"

[compositor]
type = "kde"

[sleep]
idle_timeout_secs = 300
suspend = false

[remote]
enabled = false
bind_address = "127.0.0.1"
port = 8384
"#;
        let result: Result<Config, _> = toml::from_str(toml_str);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("unknown variant"));
    }
}
