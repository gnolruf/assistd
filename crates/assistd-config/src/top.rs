//! The top-level `Config`: loading, writing and cross-key validation.

use std::collections::HashSet;
use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use assistd_utils::path::lexical_clean;
use serde::{Deserialize, Serialize};

use crate::chat::ChatConfig;
use crate::compositor::CompositorConfig;
use crate::defaults::CONFIRMATION_PROMPT_SECS;
use crate::embedding::EmbeddingConfig;
use crate::errors::{ConfigError, ConfigIssue};
use crate::mcp::McpConfig;
use crate::memory::MemoryConfig;
use crate::model::ModelConfig;
use crate::notifications::NotificationsConfig;
use crate::search::SearchConfig;
use crate::settings::{DescribeSettings, describe_fields};
use crate::sleep::SleepConfig;
use crate::tasks::TasksConfig;
use crate::timeouts::TimeoutsConfig;
use crate::tools::{ToolsConfig, ToolsOutputConfig};
use crate::voice::VoiceConfig;

const CONFIG_DIR_MODE: u32 = 0o700;
const CONFIG_FILE_MODE: u32 = 0o600;
const GROUP_OR_WORLD_READABLE: u32 = 0o044;
const SAMPLE_CONFIG: &str = include_str!("../../../config/config.sample.toml");
/// Keys a config may no longer set, each with what replaced it.
const REMOVED_KEYS: &[(&str, &str)] = &[
    (
        "tools.bash.allowed_programs",
        "the command classifier now decides which programs run unasked; \
         answer \"always allow\" to approve others, kept in allowed_programs.toml",
    ),
    (
        "tools.bash.destructive_patterns",
        "the command classifier and path guards now decide which commands ask",
    ),
];

/// Top-level assistd configuration, deserialized from `config.toml`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Config {
    pub model: ModelConfig,
    pub chat: ChatConfig,
    pub voice: VoiceConfig,
    pub compositor: CompositorConfig,
    pub sleep: SleepConfig,
    pub tools: ToolsConfig,
    pub memory: MemoryConfig,
    pub embedding: EmbeddingConfig,
    pub search: SearchConfig,
    pub mcp: McpConfig,
    #[serde(skip)]
    pub timeouts: TimeoutsConfig,
    pub notifications: NotificationsConfig,
    pub tasks: TasksConfig,
}

impl DescribeSettings for Config {
    describe_fields! {
        model => nested,
        chat => nested,
        voice => nested,
        compositor => nested,
        sleep => nested,
        tools => nested,
        memory => nested,
        embedding => nested,
        search => nested,
        mcp => nested,
        timeouts => skip,
        notifications => nested,
        tasks => nested,
    }
}

impl Config {
    /// Fails with [`ConfigError::Validation`] listing [`Self::issues`] when
    /// there are any.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let issues = self.issues();
        if issues.is_empty() {
            Ok(())
        } else {
            Err(ConfigError::Validation(issues))
        }
    }

    /// Every rule between keys, and between a key and this machine, that
    /// this config breaks; the field types hold each key's own rules.
    pub fn issues(&self) -> Vec<ConfigIssue> {
        let mut issues = Vec::new();
        validate_chat(&mut issues, &self.chat, &self.model);
        validate_sleep(&mut issues, &self.sleep);
        validate_tool_dirs(&mut issues, &self.tools);
        validate_tool_output_share(&mut issues, &self.tools.output, &self.chat);
        validate_tool_timeouts(&mut issues, self);
        validate_ports(&mut issues, self);
        validate_mcp(&mut issues, &self.mcp);
        issues
    }

    /// `$XDG_CONFIG_HOME/assistd/config.toml`, or `~/.config/assistd/config.toml`.
    /// Errors when no home directory can be found.
    pub fn default_path() -> Result<PathBuf, ConfigError> {
        let config_dir = dirs::config_dir().ok_or(ConfigError::NoConfigDir)?;
        Ok(config_dir.join("assistd/config.toml"))
    }

    /// Loads and deserializes a config from the given TOML file. Keys the
    /// schema doesn't know are skipped, with a warning logged for each; a
    /// warning is also logged if other users can read a file that sets MCP env.
    /// Fails with [`ConfigError::Removed`] on a key assistd no longer reads.
    pub fn load_from_file(path: &Path) -> Result<Self, ConfigError> {
        let content = fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let parse_error = |source| ConfigError::parse(path, &content, &source);
        let config: Self = toml::from_str(&content).map_err(parse_error)?;
        let raw: toml::Table = toml::from_str(&content).map_err(parse_error)?;
        let unknown = config.unknown_keys(&raw)?;
        if let Some(&(key, replacement)) = REMOVED_KEYS
            .iter()
            .find(|(removed, _)| unknown.iter().any(|key| key == removed))
        {
            return Err(ConfigError::Removed {
                path: path.to_path_buf(),
                key,
                replacement,
            });
        }
        for key in unknown {
            tracing::warn!(
                target: "assistd::config",
                "ignoring `{key}` in {}: assistd has no such setting \
                 (see config.sample.toml for the ones it reads)",
                path.display()
            );
        }
        config.warn_if_credentials_exposed(path);
        Ok(config)
    }

    /// Dotted paths of the keys in `raw` that no field of `self` accounts
    /// for, given that `self` was deserialized from `raw`.
    pub fn unknown_keys(&self, raw: &toml::Table) -> Result<Vec<String>, ConfigError> {
        let known = toml::Table::try_from(self)?;
        let mut unknown = Vec::new();
        collect_unknown_keys(raw, &known, "", &mut unknown);
        Ok(unknown)
    }

    /// Writes the annotated sample config, which sets no key, to `path` as an
    /// owner-only file, creating missing parent directories owner-only.
    /// Errors if the file already exists.
    pub fn write_sample(path: &Path) -> Result<(), ConfigError> {
        if let Some(parent) = path.parent() {
            create_private_dir_all(parent)?;
        }
        create_private_file(path)?
            .write_all(SAMPLE_CONFIG.as_bytes())
            .map_err(|source| ConfigError::Write {
                path: path.to_path_buf(),
                source,
            })
    }

    fn exposes_mcp_secrets(&self, mode: u32) -> bool {
        mode & GROUP_OR_WORLD_READABLE != 0
            && self
                .mcp
                .servers
                .iter()
                .any(|server| server.transport.sets_secrets())
    }

    fn warn_if_credentials_exposed(&self, path: &Path) {
        let exposed = fs::metadata(path)
            .is_ok_and(|metadata| self.exposes_mcp_secrets(metadata.permissions().mode()));
        if exposed {
            tracing::warn!(
                target: "assistd::config",
                "{} is readable by other users and sets mcp.servers[].env, headers or client_secret; \
                 run `chmod 600 {}` to keep those values private",
                path.display(),
                path.display()
            );
        }
    }
}

pub(crate) fn create_private_dir_all(dir: &Path) -> Result<(), ConfigError> {
    DirBuilder::new()
        .recursive(true)
        .mode(CONFIG_DIR_MODE)
        .create(dir)
        .map_err(|source| ConfigError::CreateDir {
            path: dir.to_path_buf(),
            source,
        })
}

fn create_private_file(path: &Path) -> Result<File, ConfigError> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(CONFIG_FILE_MODE)
        .open(path)
        .map_err(|source| match source.kind() {
            io::ErrorKind::AlreadyExists => ConfigError::AlreadyExists(path.to_path_buf()),
            _ => ConfigError::Write {
                path: path.to_path_buf(),
                source,
            },
        })
}

/// Walks `raw` alongside `known`, the serialized parsed config, which holds
/// every schema key because defaults fill the ones `raw` omits.
fn collect_unknown_keys(
    raw: &toml::Table,
    known: &toml::Table,
    prefix: &str,
    unknown: &mut Vec<String>,
) {
    for (key, raw_value) in raw {
        let path = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        match (raw_value, known.get(key)) {
            (_, None) => unknown.push(path),
            (toml::Value::Table(raw_table), Some(toml::Value::Table(known_table))) => {
                collect_unknown_keys(raw_table, known_table, &path, unknown);
            }
            (toml::Value::Array(raw_items), Some(toml::Value::Array(known_items))) => {
                collect_unknown_array_keys(raw_items, known_items, &path, unknown);
            }
            _ => {}
        }
    }
}

fn collect_unknown_array_keys(
    raw_items: &[toml::Value],
    known_items: &[toml::Value],
    path: &str,
    unknown: &mut Vec<String>,
) {
    for (index, items) in raw_items.iter().zip(known_items).enumerate() {
        if let (toml::Value::Table(raw_table), toml::Value::Table(known_table)) = items {
            collect_unknown_keys(raw_table, known_table, &format!("{path}[{index}]"), unknown);
        }
    }
}

fn validate_chat(issues: &mut Vec<ConfigIssue>, chat: &ChatConfig, model: &ModelConfig) {
    let total =
        u64::from(chat.max_history_tokens.get()) + u64::from(chat.max_response_tokens.get());
    let budget = model.context_budget();
    if total > u64::from(budget) {
        issues.push(ConfigIssue::HistoryAndResponseOverBudget { total, budget });
    }
    if chat.max_reasoning_tokens >= chat.max_response_tokens {
        issues.push(ConfigIssue::ReasoningNotBelowResponse);
    }
    if chat.summary_target_tokens >= chat.max_history_tokens {
        issues.push(ConfigIssue::SummaryNotBelowHistory);
    }
}

fn validate_sleep(issues: &mut Vec<ConfigIssue>, sleep: &SleepConfig) {
    if sleep.idle_to_drowsy_mins > 0
        && sleep.idle_to_sleep_mins > 0
        && sleep.idle_to_sleep_mins <= sleep.idle_to_drowsy_mins
    {
        issues.push(ConfigIssue::SleepNotAfterDrowsy);
    }
}

fn validate_tool_dirs(issues: &mut Vec<ConfigIssue>, tools: &ToolsConfig) {
    let owned_by_others = owned_by_others();
    for (key, dir) in [
        ("tools.scratch.dir", tools.scratch.resolved_dir()),
        (
            "tools.output.overflow_dir",
            tools.output.resolved_overflow_dir(),
        ),
    ] {
        match dir.map(|dir| lexical_clean(&dir)) {
            Ok(dir) if holds_any(&dir, &owned_by_others) => {
                issues.push(ConfigIssue::SharedToolDir { key, dir });
            }
            Ok(_) => {}
            Err(no_default) => issues.push(no_default.into()),
        }
    }
}

/// One tool result's head must leave room for the rest of the history.
fn validate_tool_output_share(
    issues: &mut Vec<ConfigIssue>,
    output: &ToolsOutputConfig,
    chat: &ChatConfig,
) {
    let head_tokens = u64::from(output.max_kb.get()) * 1024 / 4;
    let allowed = u64::from(chat.max_history_tokens.get()) / 4;
    if head_tokens > allowed {
        issues.push(ConfigIssue::ToolOutputOverShare {
            max_kb: output.max_kb.get(),
            head_tokens,
            allowed,
        });
    }
}

/// A tool's own timeout must leave it time to finish before the tool-call
/// deadline cuts it off.
fn validate_tool_timeouts(issues: &mut Vec<ConfigIssue>, config: &Config) {
    let mcp_timeouts = config
        .mcp
        .servers
        .iter()
        .enumerate()
        .filter(|_| config.mcp.enabled)
        .map(|(index, server)| {
            (
                format!("mcp.servers[{index}].request_timeout_secs"),
                server.request_timeout_secs,
            )
        });
    let bash_timeout = (
        "tools.bash.timeout_secs".to_string(),
        config.tools.bash.timeout_secs,
    );
    let max_secs = config.timeouts.max_tool_timeout_secs();
    for (key, timeout) in std::iter::once(bash_timeout).chain(mcp_timeouts) {
        if timeout.get() > max_secs {
            issues.push(ConfigIssue::ToolTimeoutOverValve {
                key,
                timeout_secs: timeout.get(),
                max_secs,
                valve_secs: config.timeouts.tool_call_secs,
                confirmation_secs: CONFIRMATION_PROMPT_SECS,
            });
        }
    }
}

fn validate_ports(issues: &mut Vec<ConfigIssue>, config: &Config) {
    let Config {
        model,
        embedding,
        search,
        ..
    } = config;
    if embedding.enabled && embedding.port == model.port {
        issues.push(ConfigIssue::PortClash {
            key: "embedding.port",
            other: "model.port (the chat server)",
        });
    }
    if search.enabled && search.port == model.port {
        issues.push(ConfigIssue::PortClash {
            key: "search.port",
            other: "model.port (the chat server)",
        });
    }
    if search.enabled && embedding.enabled && search.port == embedding.port {
        issues.push(ConfigIssue::PortClash {
            key: "search.port",
            other: "embedding.port (the embedding server)",
        });
    }
}

fn validate_mcp(issues: &mut Vec<ConfigIssue>, mcp: &McpConfig) {
    if !mcp.enabled {
        return;
    }
    let mut seen_names = HashSet::new();
    for (index, server) in mcp.servers.iter().enumerate() {
        if !seen_names.insert(&server.name) {
            issues.push(ConfigIssue::DuplicateMcpServer {
                index,
                name: server.name.as_str().to_owned(),
            });
        }
    }
}

/// Whether `dir` is `/`, or holds one of `owned`: the daemon restricts a tool
/// directory to its owner and removes old files from it at startup.
fn holds_any(dir: &Path, owned: &[PathBuf]) -> bool {
    dir.parent().is_none() || owned.iter().any(|owned| owned.starts_with(dir))
}

/// `$HOME` and assistd's config and data directories, which no tool directory may hold.
fn owned_by_others() -> Vec<PathBuf> {
    let assistd_dirs = [dirs::config_dir(), dirs::data_dir()]
        .into_iter()
        .flatten()
        .map(|dir| dir.join("assistd"));
    dirs::home_dir().into_iter().chain(assistd_dirs).collect()
}

#[cfg(test)]
mod tests;
