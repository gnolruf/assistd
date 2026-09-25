//! assistd configuration schema. An omitted key takes its value from
//! [`defaults`]; an unknown key is ignored with a warning, and `0` in a
//! `NonZero*` field is a parse error.

pub mod chat;
pub mod compositor;
pub mod daemon;
pub mod defaults;
pub mod embedding;
pub mod errors;
pub mod fixtures;
pub mod llama;
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
pub use daemon::DaemonConfig;
pub use embedding::EmbeddingConfig;
pub use errors::ConfigError;
pub use llama::LlamaServerConfig;
pub use mcp::{McpConfig, McpServerConfig};
pub use memory::MemoryConfig;
pub use model::ModelConfig;
pub use presence::PresenceConfig;
pub use sleep::SleepConfig;
pub use timeouts::TimeoutsConfig;
pub use tools::{
    BashSandboxMode, ScreenshotBackend, ToolsBashConfig, ToolsConfig, ToolsOutputConfig,
    ToolsScreenshotConfig, ToolsWriteConfig,
};
pub use top::Config;
pub use tray::{PopupAnchor, TrayConfig, TrayPopupConfig, TrayPopupWakeConfig};
pub use voice::{
    CodeBlockMode, ContinuousListenConfig, SynthesisConfig, TranscriptionConfig, VoiceConfig,
};
