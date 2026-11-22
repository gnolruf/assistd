//! Embedding subsystem: the [`Embedder`] trait, an HTTP client and launch spec for a
//! dedicated embedding llama-server, and the background task that embeds rows missing one.

use std::fmt;
use std::time::Duration;

use async_trait::async_trait;

pub mod client;
pub mod embedder_task;
mod error;
mod handle;
pub mod server;

pub use client::{LlamaEmbedder, TaskPrefixes};
pub use embedder_task::spawn_embedder_task;
pub use error::EmbedError;
pub use handle::EmbedderHandle;
pub use server::EmbedServerSpec;

/// Per-request HTTP deadline against `/v1/embeddings`.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Most inputs sent to the embedding server in one request.
pub const BATCH_SIZE: usize = 32;

/// Generates embedding vectors, telling text that is stored apart from the query that
/// searches it so each gets the task prefix its model was trained with.
#[async_trait]
pub trait Embedder: fmt::Debug + Send + Sync + 'static {
    /// An L2-normalised embedding of a search `query`, comparable by dot product with
    /// the vectors [`Embedder::embed_documents`] returns.
    async fn embed_query(&self, query: &str) -> Result<Vec<f32>, EmbedError>;

    /// L2-normalised embeddings of `texts` that are stored to be searched, in input
    /// order; fails as a whole if any input fails.
    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError>;

    /// Model id, as `<owner>/<repo>:<quant>`.
    fn model(&self) -> &str;
    /// Vector dimensionality, stable for the life of the embedder.
    fn dim(&self) -> usize;
}
