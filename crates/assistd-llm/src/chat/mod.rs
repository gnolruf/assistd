//! Streaming chat client for the locally-managed llama-server process.
//!
//! The daemon holds a single `LlamaChatClient` as `Arc<dyn LlmBackend>`.
//! Each query contributes a turn to one daemon-wide conversation guarded by
//! a `tokio::sync::Mutex`. Tokens are streamed back to the caller as
//! `LlmEvent::Delta` events followed by a terminal `LlmEvent::Done`.

pub mod client;
pub mod conversation;
pub mod error;
pub mod sse;
pub mod wire;

pub use client::LlamaChatClient;
pub use error::ChatClientError;
