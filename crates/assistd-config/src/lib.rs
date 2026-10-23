//! assistd configuration schema. An omitted key takes its value from
//! [`defaults`]; an unknown key is ignored with a warning, and `0` in a
//! `NonZero*` field is a parse error.

extern crate self as assistd_config;

pub mod chat;
pub mod compositor;
pub mod custom_args;
pub mod defaults;
pub mod document;
pub mod embedding;
pub mod errors;
mod home_path;
pub mod hotkeys;
pub mod mcp;
pub mod memory;
pub mod model;
pub mod notifications;
pub mod search;
pub mod settings;
pub mod sleep;
pub mod timeouts;
pub mod tools;
pub mod top;
pub mod voice;

pub use chat::ChatConfig;
pub use compositor::{Compositor, CompositorConfig, CompositorType};
pub use custom_args::{ChatServer, CustomArgs, EmbeddingServer, ServerKind};
pub use document::ConfigDocument;
pub use embedding::EmbeddingConfig;
pub use errors::{ConfigError, CustomArgsError, SearchCommandError, UnitIntervalError};
pub use hotkeys::HotkeysConfig;
pub use mcp::{McpConfig, McpServerConfig};
pub use memory::MemoryConfig;
pub use model::ModelConfig;
pub use notifications::{NotificationsConfig, TrayIconsConfig};
pub use search::{SearchCommand, SearchConfig, SearchEngine};
pub use settings::{DescribeSettings, SettingInfo, setting_info};
pub use sleep::SleepConfig;
pub use timeouts::TimeoutsConfig;
pub use tools::{
    BashSandboxMode, ToolsBashConfig, ToolsConfig, ToolsOutputConfig, ToolsScratchConfig,
    ToolsWriteConfig,
};
pub use top::Config;
pub use voice::{SynthesisConfig, TranscriptionConfig, UnitInterval, VoiceConfig, WakeWordConfig};
