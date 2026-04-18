//! assistd configuration types.
//!
//! This is a leaf crate: it depends on no other internal crate, so both
//! `assistd-core` and `assistd-llm` can import it without creating the
//! `core → llm → core` cycle that previously forced us to mirror every
//! config field into parallel `Spec` structs in `assistd-llm`.
//!
//! Every default value lives as a single `pub const` in [`defaults`]; the
//! `#[serde(default = "…")]` helpers in each section module reference
//! those constants so tests and config defaults can't drift apart.

pub mod agent;
pub mod chat;
pub mod compositor;
pub mod daemon;
pub mod defaults;
pub mod errors;
pub mod fixtures;
pub mod llama;
pub mod model;
pub mod presence;
pub mod remote;
pub mod sleep;
pub mod tools;
pub mod top;
pub mod voice;

pub use agent::AgentConfig;
pub use chat::ChatConfig;
pub use compositor::{CompositorConfig, CompositorType};
pub use daemon::DaemonConfig;
pub use errors::ConfigError;
pub use llama::LlamaServerConfig;
pub use model::ModelConfig;
pub use presence::PresenceConfig;
pub use remote::RemoteConfig;
pub use sleep::SleepConfig;
pub use tools::{
    BashSandboxMode, ToolsBashConfig, ToolsConfig, ToolsOutputConfig, ToolsWriteConfig,
};
pub use top::Config;
pub use voice::VoiceConfig;
