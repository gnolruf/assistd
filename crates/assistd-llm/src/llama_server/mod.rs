//! Out-of-process `llama-server` lifecycle manager.
//!
//! Owns the child process: spawns it with the configured model + auto-derived
//! GPU layer count, polls `/health` until it reports ready, supervises the
//! child, and restarts it with exponential backoff on unexpected exit. After
//! [`MAX_CONSECUTIVE_FAILURES`] consecutive pre-ready failures the supervisor
//! enters a terminal `Degraded` state rather than crash-looping forever.
//!
//! The daemon holds a single [`LlamaService`] handle. This module does **not**
//! implement [`crate::LlmBackend`]; wiring the backend to talk HTTP to the
//! managed server is a follow-up task.

pub mod backoff;
pub mod capabilities;
pub mod control;
pub mod error;
pub mod health;
pub mod process;
pub mod service;
pub mod supervisor;

pub use backoff::MAX_CONSECUTIVE_FAILURES;
pub use capabilities::{VisionState, detect_vision_support, probe_capabilities};
pub use control::LlamaServerControl;
pub use error::LlamaServerError;
pub use service::{LlamaService, ReadyState};
