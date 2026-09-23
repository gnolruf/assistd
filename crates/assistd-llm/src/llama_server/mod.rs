//! Out-of-process `llama-server` lifecycle: spawns the child, polls
//! `/health` until it reports ready, restarts it with exponential backoff
//! on unexpected exit, and enters a terminal `Degraded` state after
//! [`MAX_CONSECUTIVE_FAILURES`] consecutive pre-ready failures rather
//! than crash-looping forever.

pub mod backoff;
pub mod capabilities;
pub mod control;
pub mod error;
pub mod health;
pub mod process;
pub mod service;
pub mod supervisor;

pub use backoff::MAX_CONSECUTIVE_FAILURES;
pub use capabilities::{VisionState, probe_capabilities_routed};
pub use control::LlamaServerControl;
pub use error::LlamaServerError;
pub use service::{LlamaService, ReadyState};
