//! assistd configuration schema. An omitted key takes its value from
//! [`defaults`]; an unknown key is ignored with a warning, and `0` in a
//! `NonZero*` field is a parse error.

pub mod chat;
pub mod compositor;
pub mod custom_args;
pub mod daemon;
pub mod defaults;
pub mod embedding;
pub mod errors;
mod home_path;
pub mod mcp;
pub mod memory;
pub mod model;
pub mod presence;
pub mod sleep;
pub mod timeouts;
pub mod tools;
pub mod top;
pub mod tray;
pub mod voice;

pub use chat::ChatConfig;
pub use compositor::{CompositorConfig, CompositorType};
pub use custom_args::{ChatServer, CustomArgs, EmbeddingServer, ServerKind};
pub use daemon::DaemonConfig;
pub use embedding::EmbeddingConfig;
pub use errors::{ConfigError, CustomArgsError};
pub use mcp::{McpConfig, McpServerConfig};
pub use memory::MemoryConfig;
pub use model::ModelConfig;
pub use presence::PresenceConfig;
pub use sleep::SleepConfig;
pub use timeouts::TimeoutsConfig;
pub use tools::{
    BashSandboxMode, ScreenshotBackend, ToolsBashConfig, ToolsConfig, ToolsOutputConfig,
    ToolsScratchConfig, ToolsScreenshotConfig, ToolsWriteConfig,
};
pub use top::Config;
pub use tray::{TrayConfig, TrayIconsConfig, TrayNotificationsConfig, TrayNotificationsWakeConfig};
pub use voice::{
    CodeBlockMode, ContinuousListenConfig, SynthesisConfig, TranscriptionConfig, VoiceConfig,
};
