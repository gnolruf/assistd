use assistd_llm::{ChatSpec, ModelSpec, ServerSpec};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Top-level assistd configuration, deserialized from `config.toml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
}

/// Local model settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    /// Model identifier passed to llama-server's `--hf` flag.
    /// Format: `owner/repo:quant` (e.g. `"bartowski/Qwen3-14B-GGUF:Q4_K_M"`).
    pub name: String,
    /// Context window length in tokens.
    pub context_length: u32,
}

/// llama-server process lifecycle settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlamaServerConfig {
    /// Path to the llama-server binary. Absolute path or a name resolvable via `$PATH`.
    pub binary_path: String,
    /// Host the managed llama-server binds to. Should be a loopback address.
    pub host: String,
    /// TCP port the managed llama-server binds to.
    pub port: u16,
    /// GPU layer count passed as `-ngl`. Default `9999` offloads all layers;
    /// llama.cpp clamps to the model's actual layer count.
    #[serde(default = "default_gpu_layers")]
    pub gpu_layers: u32,
    /// Timeout in seconds for the health check to succeed after spawning
    /// llama-server. First-time HuggingFace downloads may need several minutes.
    #[serde(default = "default_ready_timeout_secs")]
    pub ready_timeout_secs: u64,
}

/// Chat backend behaviour: system prompt, history window, sampling.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatConfig {
    /// System prompt injected as the first message of every request.
    /// An empty string disables system-prompt injection.
    pub system_prompt: String,
    /// Approximate token budget for the full request messages (system + history).
    /// Must be strictly less than `model.context_length`. When exceeded, older
    /// messages are summarized and replaced by a single summary message.
    pub max_history_tokens: u32,
    /// Approximate target length of a history summary, in tokens.
    pub summary_target_tokens: u32,
    /// Number of recent user/assistant exchanges to preserve verbatim when
    /// summarizing. Must be at least 1.
    pub preserve_recent_turns: u32,
    /// Sampling temperature in the range `0.0..=2.0`.
    pub temperature: f32,
    /// Maximum tokens the model may emit in a single streamed response.
    pub max_response_tokens: u32,
    /// Maximum tokens for the summarization (non-streaming) call. Typically a
    /// bit above `summary_target_tokens`.
    pub max_summary_tokens: u32,
    /// HTTP request timeout for a single chat call, in seconds.
    pub request_timeout_secs: u64,
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
    /// Minutes of no user interaction after which the daemon drops
    /// `Active → Drowsy` (model weights unloaded, server stays alive).
    /// `0` disables this transition. Independent of `idle_to_sleep_mins`.
    #[serde(default = "default_idle_to_drowsy_mins")]
    pub idle_to_drowsy_mins: u64,
    /// Minutes of no user interaction after which the daemon drops to
    /// `Sleeping` (llama-server stopped, all VRAM freed). Must be greater
    /// than `idle_to_drowsy_mins` when both are non-zero. `0` disables
    /// this transition.
    #[serde(default = "default_idle_to_sleep_mins")]
    pub idle_to_sleep_mins: u64,
    /// Whether to suspend the machine (true) or just deactivate (false).
    pub suspend: bool,

    /// Enable the automatic GPU contention monitor. When true, a background
    /// task polls NVML for other processes using VRAM and transitions the
    /// daemon to Sleeping when a configurable threshold is exceeded.
    #[serde(default = "default_gpu_monitor_enabled")]
    pub gpu_monitor_enabled: bool,
    /// NVML poll interval in seconds.
    #[serde(default = "default_gpu_poll_secs")]
    pub gpu_poll_secs: u64,
    /// Per-process VRAM threshold in MiB. A non-assistd process holding at
    /// least this much VRAM triggers a transition to Sleeping. 2048 = 2 GiB.
    #[serde(default = "default_gpu_vram_threshold_mb")]
    pub gpu_vram_threshold_mb: u64,
    /// Automatically transition back to Active when the contending process
    /// exits.
    #[serde(default)]
    pub gpu_auto_wake: bool,
    /// Process basenames (matched against `/proc/<pid>/comm`, kernel-truncated
    /// at 16 bytes) that never trigger contention sleep, even above the
    /// threshold.
    #[serde(default = "default_gpu_allowlist")]
    pub gpu_allowlist: Vec<String>,
    /// Process basenames that always trigger sleep when present, regardless
    /// of their VRAM usage.
    #[serde(default)]
    pub gpu_denylist: Vec<String>,
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

/// Manual presence-control settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PresenceConfig {
    /// Global hotkey that cycles `Active → Drowsy → Sleeping → Active`.
    /// Empty string disables the in-daemon global hotkey listener.
    #[serde(default = "default_presence_hotkey")]
    pub hotkey: String,
}

impl Default for PresenceConfig {
    fn default() -> Self {
        Self {
            hotkey: default_presence_hotkey(),
        }
    }
}

fn default_presence_hotkey() -> String {
    "Super+Escape".to_string()
}

/// Daemon process lifecycle settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DaemonConfig {
    /// Seconds to wait for in-flight IPC connections (e.g. streaming LLM
    /// responses) to finish before aborting them on daemon shutdown. `0`
    /// aborts in-flight work immediately.
    #[serde(default = "default_shutdown_grace_secs")]
    pub shutdown_grace_secs: u64,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            shutdown_grace_secs: default_shutdown_grace_secs(),
        }
    }
}

fn default_shutdown_grace_secs() -> u64 {
    5
}

/// Tools subsystem configuration. Nested container so future tool-related
/// knobs (sandboxing, per-command timeouts, etc.) can slot in alongside the
/// output-presentation limits.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ToolsConfig {
    #[serde(default)]
    pub output: ToolsOutputConfig,
}

/// Layer-2 presentation limits applied to the final output of `run` before
/// it is handed to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolsOutputConfig {
    /// Max lines of stdout surfaced to the LLM before overflow spill.
    #[serde(default = "default_tools_max_lines")]
    pub max_lines: u32,
    /// Max bytes of the truncated head, in KB.
    #[serde(default = "default_tools_max_kb")]
    pub max_kb: u32,
    /// Directory where overflow output is spilled as `cmd-<n>.txt`.
    /// Cleared + recreated on daemon startup.
    #[serde(default = "default_tools_overflow_dir")]
    pub overflow_dir: String,
}

impl Default for ToolsOutputConfig {
    fn default() -> Self {
        Self {
            max_lines: default_tools_max_lines(),
            max_kb: default_tools_max_kb(),
            overflow_dir: default_tools_overflow_dir(),
        }
    }
}

fn default_tools_max_lines() -> u32 {
    200
}

fn default_tools_max_kb() -> u32 {
    50
}

fn default_tools_overflow_dir() -> String {
    "/tmp/assistd-output".to_string()
}

fn default_gpu_layers() -> u32 {
    9999
}

fn default_ready_timeout_secs() -> u64 {
    300
}

fn default_idle_to_drowsy_mins() -> u64 {
    30
}

fn default_idle_to_sleep_mins() -> u64 {
    120
}

fn default_gpu_monitor_enabled() -> bool {
    true
}

fn default_gpu_poll_secs() -> u64 {
    5
}

fn default_gpu_vram_threshold_mb() -> u64 {
    2048
}

fn default_gpu_allowlist() -> Vec<String> {
    vec![
        "Xorg".into(),
        "Xwayland".into(),
        "gnome-shell".into(),
        "kwin_x11".into(),
        "kwin_wayland".into(),
        "firefox".into(),
        "chromium".into(),
        "chrome".into(),
    ]
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

impl Default for Config {
    fn default() -> Self {
        Self {
            model: ModelConfig {
                name: "bartowski/Qwen3-14B-GGUF:Q4_K_M".to_string(),
                context_length: 8192,
            },
            llama_server: LlamaServerConfig {
                binary_path: "llama-server".to_string(),
                host: "127.0.0.1".to_string(),
                port: 8385,
                gpu_layers: default_gpu_layers(),
                ready_timeout_secs: default_ready_timeout_secs(),
            },
            chat: ChatConfig {
                system_prompt:
                    "You are assistd, a concise local desktop assistant running on a Linux \
                     workstation. Answer precisely and in a conversational tone."
                        .to_string(),
                max_history_tokens: 6000,
                summary_target_tokens: 1000,
                preserve_recent_turns: 4,
                temperature: 0.7,
                max_response_tokens: 1024,
                max_summary_tokens: 1200,
                request_timeout_secs: 120,
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
                idle_to_drowsy_mins: default_idle_to_drowsy_mins(),
                idle_to_sleep_mins: default_idle_to_sleep_mins(),
                suspend: false,
                gpu_monitor_enabled: default_gpu_monitor_enabled(),
                gpu_poll_secs: default_gpu_poll_secs(),
                gpu_vram_threshold_mb: default_gpu_vram_threshold_mb(),
                gpu_auto_wake: false,
                gpu_allowlist: default_gpu_allowlist(),
                gpu_denylist: Vec::new(),
            },
            remote: RemoteConfig {
                enabled: false,
                bind_address: "127.0.0.1".to_string(),
                port: 8384,
            },
            presence: PresenceConfig::default(),
            daemon: DaemonConfig::default(),
            tools: ToolsConfig::default(),
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

// ---------------------------------------------------------------------------
// Spec conversions
// ---------------------------------------------------------------------------

impl Config {
    pub fn to_server_spec(&self) -> ServerSpec {
        ServerSpec {
            binary_path: self.llama_server.binary_path.clone(),
            host: self.llama_server.host.clone(),
            port: self.llama_server.port,
            gpu_layers: self.llama_server.gpu_layers,
            ready_timeout_secs: self.llama_server.ready_timeout_secs,
        }
    }

    pub fn to_model_spec(&self) -> ModelSpec {
        ModelSpec {
            name: self.model.name.clone(),
            context_length: self.model.context_length,
        }
    }

    pub fn to_chat_spec(&self) -> ChatSpec {
        ChatSpec {
            host: self.llama_server.host.clone(),
            port: self.llama_server.port,
            system_prompt: self.chat.system_prompt.clone(),
            max_history_tokens: self.chat.max_history_tokens,
            summary_target_tokens: self.chat.summary_target_tokens,
            preserve_recent_turns: self.chat.preserve_recent_turns,
            temperature: self.chat.temperature,
            max_response_tokens: self.chat.max_response_tokens,
            max_summary_tokens: self.chat.max_summary_tokens,
            request_timeout_secs: self.chat.request_timeout_secs,
            model_context_length: self.model.context_length,
        }
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
    fn validation_catches_empty_model_name() {
        let mut config = Config::default();
        config.model.name = String::new();
        let errs = validation_errors(config.validate().unwrap_err());
        assert!(errs.iter().any(|e| e.contains("model.name")));
    }

    #[test]
    fn validation_collects_multiple_errors() {
        let mut config = Config::default();
        config.model.name = String::new();
        config.model.context_length = 0;
        config.chat.max_response_tokens = 0;
        let errs = validation_errors(config.validate().unwrap_err());
        assert_eq!(errs.len(), 3);
    }

    #[test]
    fn unknown_compositor_type_rejected() {
        let toml_str = r#"
[model]
name = "test/model-GGUF:Q4_K_M"
context_length = 8192

[llama_server]
binary_path = "llama-server"
host = "127.0.0.1"
port = 8385

[chat]
system_prompt = "hi"
max_history_tokens = 4000
summary_target_tokens = 500
preserve_recent_turns = 2
temperature = 0.5
max_response_tokens = 1024
max_summary_tokens = 800
request_timeout_secs = 60

[voice]
enabled = false
hotkey = "Super+V"

[compositor]
type = "kde"

[sleep]
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

    #[test]
    fn validation_catches_empty_binary_path() {
        let mut config = Config::default();
        config.llama_server.binary_path = String::new();
        let errs = validation_errors(config.validate().unwrap_err());
        assert!(errs.iter().any(|e| e.contains("llama_server.binary_path")));
    }

    #[test]
    fn validation_catches_zero_llama_server_port() {
        let mut config = Config::default();
        config.llama_server.port = 0;
        let errs = validation_errors(config.validate().unwrap_err());
        assert!(errs.iter().any(|e| e.contains("llama_server.port")));
    }

    #[test]
    fn validation_catches_empty_llama_server_host() {
        let mut config = Config::default();
        config.llama_server.host = String::new();
        let errs = validation_errors(config.validate().unwrap_err());
        assert!(errs.iter().any(|e| e.contains("llama_server.host")));
    }

    #[test]
    fn llama_server_gpu_layers_defaults_to_9999() {
        let toml_str = r#"
[model]
name = "test/model-GGUF:Q4_K_M"
context_length = 8192

[llama_server]
binary_path = "llama-server"
host = "127.0.0.1"
port = 8385

[chat]
system_prompt = "hi"
max_history_tokens = 4000
summary_target_tokens = 500
preserve_recent_turns = 2
temperature = 0.5
max_response_tokens = 1024
max_summary_tokens = 800
request_timeout_secs = 60

[voice]
enabled = false
hotkey = "Super+V"

[compositor]
type = "sway"

[sleep]
suspend = false

[remote]
enabled = false
bind_address = "127.0.0.1"
port = 8384
"#;
        let cfg: Config = toml::from_str(toml_str).expect("parse");
        assert_eq!(cfg.llama_server.gpu_layers, 9999);
        assert_eq!(cfg.llama_server.ready_timeout_secs, 300);
    }

    #[test]
    fn validation_catches_temperature_out_of_range() {
        let mut config = Config::default();
        config.chat.temperature = 2.5;
        let errs = validation_errors(config.validate().unwrap_err());
        assert!(errs.iter().any(|e| e.contains("chat.temperature")));
    }

    #[test]
    fn validation_catches_history_budget_exceeds_context() {
        let mut config = Config::default();
        config.chat.max_history_tokens = config.model.context_length;
        let errs = validation_errors(config.validate().unwrap_err());
        assert!(
            errs.iter().any(
                |e| e.contains("chat.max_history_tokens") && e.contains("model.context_length")
            )
        );
    }

    #[test]
    fn validation_catches_summary_target_exceeds_history_budget() {
        let mut config = Config::default();
        config.chat.summary_target_tokens = config.chat.max_history_tokens;
        let errs = validation_errors(config.validate().unwrap_err());
        assert!(
            errs.iter().any(|e| e.contains("chat.summary_target_tokens")
                && e.contains("chat.max_history_tokens"))
        );
    }

    #[test]
    fn validation_catches_zero_preserve_recent_turns() {
        let mut config = Config::default();
        config.chat.preserve_recent_turns = 0;
        let errs = validation_errors(config.validate().unwrap_err());
        assert!(
            errs.iter()
                .any(|e| e.contains("chat.preserve_recent_turns"))
        );
    }

    #[test]
    fn chat_config_empty_system_prompt_is_valid() {
        let mut config = Config::default();
        config.chat.system_prompt = String::new();
        config.validate().expect("empty system prompt allowed");
    }

    #[test]
    fn missing_presence_section_uses_defaults() {
        // Existing config.toml files won't have a [presence] section — they
        // must still parse and default to Super+Escape.
        let toml_str = r#"
[model]
name = "test/model-GGUF:Q4_K_M"
context_length = 8192

[llama_server]
binary_path = "llama-server"
host = "127.0.0.1"
port = 8385

[chat]
system_prompt = "hi"
max_history_tokens = 4000
summary_target_tokens = 500
preserve_recent_turns = 2
temperature = 0.5
max_response_tokens = 1024
max_summary_tokens = 800
request_timeout_secs = 60

[voice]
enabled = false
hotkey = "Super+V"

[compositor]
type = "sway"

[sleep]
suspend = false

[remote]
enabled = false
bind_address = "127.0.0.1"
port = 8384
"#;
        let cfg: Config = toml::from_str(toml_str).expect("parse without [presence]");
        assert_eq!(cfg.presence.hotkey, "Super+Escape");
    }

    #[test]
    fn presence_config_defaults_hotkey() {
        let cfg = PresenceConfig::default();
        assert_eq!(cfg.hotkey, "Super+Escape");
    }

    #[test]
    fn empty_presence_hotkey_is_valid() {
        let mut config = Config::default();
        config.presence.hotkey = String::new();
        config
            .validate()
            .expect("empty hotkey disables listener but is valid config");
    }

    #[test]
    fn old_sleep_block_without_gpu_keys_applies_defaults() {
        // An existing config.toml written before the GPU monitor feature
        // landed must keep parsing and receive the new defaults.
        let toml_str = r#"
[model]
name = "test/model-GGUF:Q4_K_M"
context_length = 8192

[llama_server]
binary_path = "llama-server"
host = "127.0.0.1"
port = 8385

[chat]
system_prompt = "hi"
max_history_tokens = 4000
summary_target_tokens = 500
preserve_recent_turns = 2
temperature = 0.5
max_response_tokens = 1024
max_summary_tokens = 800
request_timeout_secs = 60

[voice]
enabled = false
hotkey = "Super+V"

[compositor]
type = "sway"

[sleep]
suspend = false

[remote]
enabled = false
bind_address = "127.0.0.1"
port = 8384
"#;
        let cfg: Config = toml::from_str(toml_str).expect("parse without GPU keys");
        assert!(cfg.sleep.gpu_monitor_enabled);
        assert_eq!(cfg.sleep.gpu_poll_secs, 5);
        assert_eq!(cfg.sleep.gpu_vram_threshold_mb, 2048);
        assert!(!cfg.sleep.gpu_auto_wake);
        assert!(cfg.sleep.gpu_allowlist.iter().any(|s| s == "Xorg"));
        assert!(cfg.sleep.gpu_denylist.is_empty());
        cfg.validate().expect("defaults validate");
    }

    #[test]
    fn gpu_monitor_enabled_rejects_zero_poll_interval() {
        let mut config = Config::default();
        config.sleep.gpu_monitor_enabled = true;
        config.sleep.gpu_poll_secs = 0;
        let errs = validation_errors(config.validate().unwrap_err());
        assert!(errs.iter().any(|e| e.contains("sleep.gpu_poll_secs")));
    }

    #[test]
    fn gpu_monitor_enabled_rejects_zero_threshold() {
        let mut config = Config::default();
        config.sleep.gpu_monitor_enabled = true;
        config.sleep.gpu_vram_threshold_mb = 0;
        let errs = validation_errors(config.validate().unwrap_err());
        assert!(
            errs.iter()
                .any(|e| e.contains("sleep.gpu_vram_threshold_mb"))
        );
    }

    #[test]
    fn gpu_monitor_disabled_ignores_zero_threshold() {
        let mut config = Config::default();
        config.sleep.gpu_monitor_enabled = false;
        config.sleep.gpu_poll_secs = 0;
        config.sleep.gpu_vram_threshold_mb = 0;
        config
            .validate()
            .expect("zero poll/threshold is allowed when monitor disabled");
    }

    #[test]
    fn idle_timeouts_default_from_missing_keys() {
        let toml_str = r#"
[model]
name = "test/model-GGUF:Q4_K_M"
context_length = 8192

[llama_server]
binary_path = "llama-server"
host = "127.0.0.1"
port = 8385

[chat]
system_prompt = "hi"
max_history_tokens = 4000
summary_target_tokens = 500
preserve_recent_turns = 2
temperature = 0.5
max_response_tokens = 1024
max_summary_tokens = 800
request_timeout_secs = 60

[voice]
enabled = false
hotkey = "Super+V"

[compositor]
type = "sway"

[sleep]
suspend = false

[remote]
enabled = false
bind_address = "127.0.0.1"
port = 8384
"#;
        let cfg: Config = toml::from_str(toml_str).expect("parse without idle keys");
        assert_eq!(cfg.sleep.idle_to_drowsy_mins, 30);
        assert_eq!(cfg.sleep.idle_to_sleep_mins, 120);
    }

    #[test]
    fn validation_catches_idle_sleep_lte_drowsy() {
        let mut config = Config::default();
        config.sleep.idle_to_drowsy_mins = 60;
        config.sleep.idle_to_sleep_mins = 30;
        let errs = validation_errors(config.validate().unwrap_err());
        assert!(
            errs.iter()
                .any(|e| e.contains("idle_to_sleep_mins") && e.contains("idle_to_drowsy_mins"))
        );
    }

    #[test]
    fn validation_catches_idle_sleep_eq_drowsy() {
        let mut config = Config::default();
        config.sleep.idle_to_drowsy_mins = 60;
        config.sleep.idle_to_sleep_mins = 60;
        let errs = validation_errors(config.validate().unwrap_err());
        assert!(errs.iter().any(|e| e.contains("idle_to_sleep_mins")));
    }

    #[test]
    fn idle_both_zero_is_valid() {
        let mut config = Config::default();
        config.sleep.idle_to_drowsy_mins = 0;
        config.sleep.idle_to_sleep_mins = 0;
        config
            .validate()
            .expect("both zero disables idle monitoring");
    }

    #[test]
    fn idle_only_drowsy_set_is_valid() {
        let mut config = Config::default();
        config.sleep.idle_to_drowsy_mins = 30;
        config.sleep.idle_to_sleep_mins = 0;
        config.validate().expect("only drowsy configured is valid");
    }

    #[test]
    fn idle_only_sleep_set_is_valid() {
        let mut config = Config::default();
        config.sleep.idle_to_drowsy_mins = 0;
        config.sleep.idle_to_sleep_mins = 120;
        config.validate().expect("only sleep configured is valid");
    }

    #[test]
    fn tools_config_defaults_match_spec() {
        let cfg = ToolsOutputConfig::default();
        assert_eq!(cfg.max_lines, 200);
        assert_eq!(cfg.max_kb, 50);
        assert_eq!(cfg.overflow_dir, "/tmp/assistd-output");
    }

    #[test]
    fn missing_tools_section_uses_defaults() {
        // Existing config.toml files won't have a [tools] section — they
        // must still parse and default correctly.
        let toml_str = r#"
[model]
name = "test/model-GGUF:Q4_K_M"
context_length = 8192

[llama_server]
binary_path = "llama-server"
host = "127.0.0.1"
port = 8385

[chat]
system_prompt = "hi"
max_history_tokens = 4000
summary_target_tokens = 500
preserve_recent_turns = 2
temperature = 0.5
max_response_tokens = 1024
max_summary_tokens = 800
request_timeout_secs = 60

[voice]
enabled = false
hotkey = "Super+V"

[compositor]
type = "sway"

[sleep]
suspend = false

[remote]
enabled = false
bind_address = "127.0.0.1"
port = 8384
"#;
        let cfg: Config = toml::from_str(toml_str).expect("parse without [tools]");
        assert_eq!(cfg.tools.output.max_lines, 200);
        assert_eq!(cfg.tools.output.max_kb, 50);
        assert_eq!(cfg.tools.output.overflow_dir, "/tmp/assistd-output");
        cfg.validate().expect("defaults validate");
    }

    #[test]
    fn validation_catches_zero_tools_max_lines() {
        let mut config = Config::default();
        config.tools.output.max_lines = 0;
        let errs = validation_errors(config.validate().unwrap_err());
        assert!(errs.iter().any(|e| e.contains("tools.output.max_lines")));
    }

    #[test]
    fn validation_catches_zero_tools_max_kb() {
        let mut config = Config::default();
        config.tools.output.max_kb = 0;
        let errs = validation_errors(config.validate().unwrap_err());
        assert!(errs.iter().any(|e| e.contains("tools.output.max_kb")));
    }

    #[test]
    fn validation_catches_empty_tools_overflow_dir() {
        let mut config = Config::default();
        config.tools.output.overflow_dir = String::new();
        let errs = validation_errors(config.validate().unwrap_err());
        assert!(errs.iter().any(|e| e.contains("tools.output.overflow_dir")));
    }
}
