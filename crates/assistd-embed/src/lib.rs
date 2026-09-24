//! Embedding subsystem: the [`Embedder`] trait, an HTTP client for a
//! dedicated embedding llama-server, its supervisor, and the
//! background task that embeds queued rows.

pub mod client;
pub mod embedder_task;
mod error;
pub mod server;

/// Per-request HTTP deadline against `/v1/embeddings`.
pub const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Most inputs sent to the embedding server in one request.
pub const BATCH_SIZE: usize = 32;

pub use client::LlamaEmbedder;
pub use embedder_task::{EmbedJob, spawn_embedder_task};
pub use error::EmbedError;
pub use server::{EmbedServerError, EmbedService, ReadyState};

use async_trait::async_trait;

/// Generates embedding vectors for text.
#[async_trait]
pub trait Embedder: Send + Sync + 'static {
    /// An L2-normalised embedding of `text`; callers compute cosine as
    /// a dot product.
    async fn embed(&self, text: String) -> Result<Vec<f32>, EmbedError>;

    /// L2-normalised embeddings of `texts`, one per input and in input
    /// order. Fails as a whole if any input fails. The default embeds
    /// each text in turn.
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

/// Embed `texts` in one [`Embedder::embed_batch`] call, returning one
/// result per input in input order. If the batch fails, each text is
/// retried on its own so a bad input fails only itself.
pub async fn embed_each(
    embedder: &dyn Embedder,
    texts: &[&str],
) -> Vec<Result<Vec<f32>, EmbedError>> {
    match embedder.embed_batch(texts).await {
        Ok(vectors) => vectors.into_iter().map(Ok).collect(),
        Err(e) if texts.len() > 1 => {
            tracing::debug!(
                target: "assistd::embed",
                batch = texts.len(),
                error = %e,
                "batch embed failed; retrying inputs individually"
            );
            let mut results = Vec::with_capacity(texts.len());
            for &text in texts {
                results.push(embedder.embed(text.to_owned()).await);
            }
            results
        }
        Err(e) => vec![Err(e)],
    }
}

/// Fallback when embedding is disabled: `embed` errors, `model` is
/// empty, `dim` is zero.
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
