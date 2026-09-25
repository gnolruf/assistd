//! Embedding subsystem: the [`Embedder`] trait, an HTTP client and supervisor for a
//! dedicated embedding llama-server, and the background task that embeds queued rows.

use std::time::Duration;

use async_trait::async_trait;

pub mod client;
pub mod embedder_task;
mod error;
pub mod server;

pub use client::LlamaEmbedder;
pub use embedder_task::{EmbedJob, spawn_embedder_task};
pub use error::EmbedError;
pub use server::{EmbedServerError, EmbedService, ReadyState};

/// Per-request HTTP deadline against `/v1/embeddings`.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Most inputs sent to the embedding server in one request.
pub const BATCH_SIZE: usize = 32;

/// Generates embedding vectors for text.
#[async_trait]
pub trait Embedder: Send + Sync + 'static {
    /// An L2-normalised embedding of `text`, so cosine similarity is a plain dot product.
    async fn embed(&self, text: String) -> Result<Vec<f32>, EmbedError>;

    /// L2-normalised embeddings of `texts`, in input order; fails as a whole if any
    /// input fails. The default embeds each text in turn.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let mut vectors = Vec::with_capacity(texts.len());
        for &text in texts {
            vectors.push(self.embed(text.to_owned()).await?);
        }
        Ok(vectors)
    }

    /// Model id, stored alongside every vector so models never mix.
    fn model(&self) -> &str;
    /// Vector dimensionality, stable for the life of the embedder.
    fn dim(&self) -> usize;
}

/// Embed `texts` in one [`Embedder::embed_batch`] call, returning one result per input.
/// If the batch fails, each text is retried alone so a bad input fails only itself.
pub async fn embed_each(
    embedder: &dyn Embedder,
    texts: &[&str],
) -> Vec<Result<Vec<f32>, EmbedError>> {
    match embedder.embed_batch(texts).await {
        Ok(vectors) => vectors.into_iter().map(Ok).collect(),
        Err(err) if texts.len() > 1 => {
            tracing::debug!(
                target: "assistd::embed",
                batch = texts.len(),
                error = %err,
                "batch embed failed; retrying inputs individually"
            );
            let mut results = Vec::with_capacity(texts.len());
            for &text in texts {
                results.push(embedder.embed(text.to_owned()).await);
            }
            results
        }
        Err(err) => vec![Err(err)],
    }
}

/// Fallback when embedding is disabled: `embed` errors, `model` is empty, `dim` is zero.
pub struct NoEmbedder;

#[async_trait]
impl Embedder for NoEmbedder {
    async fn embed(&self, _text: String) -> Result<Vec<f32>, EmbedError> {
        Err(EmbedError::Disabled)
    }
    fn model(&self) -> &str {
        ""
    }
    fn dim(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests;
