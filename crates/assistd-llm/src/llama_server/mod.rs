//! Out-of-process `llama-server` lifecycle: spawn, health-poll, restart with
//! backoff, and a terminal `Degraded` state once restart limits trip.

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
