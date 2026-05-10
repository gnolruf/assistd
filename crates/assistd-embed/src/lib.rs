#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::print_stdout,
        clippy::print_stderr
    )
)]

//! Embedding subsystem: a dedicated llama-server instance that serves
//! `/v1/embeddings`, plus the [`Embedder`] trait callers depend on.
//!
//! Topology rationale (vs. swapping models on the chat router):
//! - The chat server runs in **router mode** with on-demand model load/unload
//!   driven by the presence state machine. Toggling `--embedding` per request
//!   would require server-level reconfiguration that llama.cpp doesn't
//!   support at runtime.
//! - The embedding model is small enough (~30-300 MB Q4) to keep resident
//!   on CPU 24/7 without VRAM pressure.
//! - Embedding stays available when the chat router is `Drowsy` /
//!   `Sleeping`, so semantic retrieval keeps working in low-power states.
//!
//! Public surface:
//! - [`Embedder`] - trait `AppState`/tools hold as `Arc<dyn Embedder>`.
//! - [`NoEmbedder`] - successful-no-op fallback used when the subsystem is
//!   disabled in config or fails to start (mirrors `NoMemoryStore` /
//!   `NoVoiceOutput`).
//! - [`LlamaEmbedder`] - concrete HTTP client for an `/v1/embeddings`
//!   endpoint.
//! - [`server::EmbedService`] - supervised process lifecycle.
//!
//! The background [`spawn_embedder_task`] worker is added in a follow-up
//! commit alongside the `assistd-memory` `WriteOp` variants it dispatches
//! into.

pub mod client;
pub mod embedder_task;
pub mod server;

pub use client::LlamaEmbedder;
pub use embedder_task::{EmbedJob, spawn_embedder_task};
pub use server::{EmbedServerError, EmbedService, ReadyState};

use anyhow::{Result, anyhow};
use async_trait::async_trait;

/// Generates embedding vectors for arbitrary text. Implementors must be
/// `Send + Sync + 'static` so callers can hold them as
/// `Arc<dyn Embedder>` and share across tasks.
#[async_trait]
pub trait Embedder: Send + Sync + 'static {
    /// Produce an L2-normalised embedding vector for `text`. Normalisation
    /// is the implementor's responsibility; callers compose retrieval as
    /// dot products and assume both sides are unit-length.
    async fn embed(&self, text: String) -> Result<Vec<f32>>;
    /// Model id this embedder serves. Used to filter the SQLite
    /// `embeddings` / `memory_embeddings` rows by `model` so a query
    /// against today's embedder never collides with vectors produced by
    /// a previous model.
    fn model(&self) -> &str;
    /// Vector dimensionality served by this embedder. Probed at
    /// startup; stable for the lifetime of the process.
    fn dim(&self) -> usize;
}

/// Successful-no-op fallback used when the embedding subsystem is
/// disabled in config or fails to start. `embed` returns an error so
/// callers can `Result<_>::ok()` to a "skip retrieval" branch; `model()`
/// is empty and `dim()` is `0`.
pub struct NoEmbedder;

#[async_trait]
impl Embedder for NoEmbedder {
    async fn embed(&self, _text: String) -> Result<Vec<f32>> {
        Err(anyhow!("embedder disabled"))
    }
    fn model(&self) -> &str {
        ""
    }
    fn dim(&self) -> usize {
        0
    }
}

/// Returns the crate version string from `Cargo.toml`.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn no_embedder_errors_on_embed() {
        let e = NoEmbedder;
        assert!(e.embed("hi".into()).await.is_err());
        assert_eq!(e.model(), "");
        assert_eq!(e.dim(), 0);
    }

    #[test]
    fn no_embedder_is_object_safe() {
        let _: std::sync::Arc<dyn Embedder> = std::sync::Arc::new(NoEmbedder);
    }

    #[test]
    fn version_is_not_empty() {
        assert!(!version().is_empty());
    }
}
