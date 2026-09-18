#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::print_stdout,
        clippy::print_stderr
    )
)]

//! Embedding subsystem: the [`Embedder`] trait, an HTTP client for a
//! dedicated embedding llama-server, its supervisor, and the
//! background task that embeds queued rows.

pub mod client;
pub mod embedder_task;
pub mod server;

/// Per-request HTTP deadline against `/v1/embeddings`.
pub const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

pub use client::LlamaEmbedder;
pub use embedder_task::{EmbedJob, spawn_embedder_task};
pub use server::{EmbedServerError, EmbedService, ReadyState};

use anyhow::{Result, anyhow};
use async_trait::async_trait;

/// Generates embedding vectors for text.
#[async_trait]
pub trait Embedder: Send + Sync + 'static {
    /// An L2-normalised embedding of `text`; callers compute cosine as
    /// a dot product.
    async fn embed(&self, text: String) -> Result<Vec<f32>>;
    /// Model id, stored alongside every vector so models never mix.
    fn model(&self) -> &str;
    /// Vector dimensionality, stable for the life of the embedder.
    fn dim(&self) -> usize;
}

/// Fallback when embedding is disabled: `embed` errors, `model` is
/// empty, `dim` is zero.
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
