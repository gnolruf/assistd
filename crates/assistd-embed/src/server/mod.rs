//! Supervisor for the embedding llama-server child: spawn, health
//! check, restart on crash with backoff.

pub mod backoff;
pub mod error;
pub mod health;
pub mod process;
pub mod service;
pub mod supervisor;

pub use error::EmbedServerError;
pub use service::{EmbedService, ReadyState};
