//! Out-of-process supervisor for the embedding llama-server.
//!
//! Mirrors the shape of `crates/assistd-llm/src/llama_server/`:
//! `Supervisor` runs in a `tokio::spawn`'d task, monitors the child's
//! `/health`, and restarts on crash with exponential backoff. Differences
//! from chat-server lifecycle:
//!
//! - Spawn args use `--embedding` and `--hf-repo <model>` directly: the
//!   embed model is held resident for the daemon's lifetime, so we don't
//!   need router-mode model load/unload.
//! - No `LlamaServerControl` analogue. Callers go straight to the HTTP
//!   `/v1/embeddings` endpoint via [`crate::LlamaEmbedder`].
//! - No vision capability probe.
//!
//! The daemon holds one [`EmbedService`].

pub mod backoff;
pub mod error;
pub mod health;
pub mod process;
pub mod service;
pub mod supervisor;

pub use error::EmbedServerError;
pub use service::{EmbedService, ReadyState};
