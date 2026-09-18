#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::print_stdout,
        clippy::print_stderr
    )
)]

//! assistd configuration types.
//!
//! This is a leaf crate: it depends on no other internal crate, so both
//! `assistd-core` and `assistd-llm` can import it without creating the
//! `core → llm → core` cycle that previously forced us to mirror every
//! config field into parallel `Spec` structs in `assistd-llm`.
//!
//! Every default value lives as a single `pub const` in [`defaults`],
//! read by the owning section's `Default` impl, so tests and config
//! defaults can't drift apart.
//!
//! Every section carries `#[serde(default, deny_unknown_fields)]`: an
//! omitted key falls back to that `Default` impl, and a key the schema
//! doesn't know is a parse error.

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
