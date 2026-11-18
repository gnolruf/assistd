//! Embedding subsystem: the [`Embedder`] trait, an HTTP client and launch spec for a
//! dedicated embedding llama-server, and the background task that embeds rows missing one.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use assistd_memory::{Embedding, EmbeddingTarget};

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

/// Embed `texts` in one [`Embedder::embed_documents`] call, returning one result per input.
/// If an input broke the batch ([`EmbedError::is_input_caused`]), each text is retried
/// alone; any other failure fails every input without a retry.
async fn embed_each(embedder: &dyn Embedder, texts: &[&str]) -> Vec<Result<Vec<f32>, EmbedError>> {
    match embedder.embed_documents(texts).await {
        Ok(vectors) => vectors.into_iter().map(Ok).collect(),
        Err(EmbedError::NotReady) => texts.iter().map(|_| Err(EmbedError::NotReady)).collect(),
        Err(err) if !err.is_input_caused() => {
            let shared = Arc::new(err);
            texts
                .iter()
                .map(|_| Err(EmbedError::BatchFailed(Arc::clone(&shared))))
                .collect()
        }
        Err(err) if texts.len() > 1 => {
            tracing::debug!(
                target: "assistd::embed",
                batch = texts.len(),
                error = %err,
                "batch embed failed; retrying inputs individually"
            );
            let mut results = Vec::with_capacity(texts.len());
            for &text in texts {
                results.push(embed_document(embedder, text).await);
            }
            results
        }
        Err(err) => vec![Err(err)],
    }
}

async fn embed_document(embedder: &dyn Embedder, text: &str) -> Result<Vec<f32>, EmbedError> {
    embedder
        .embed_documents(&[text])
        .await?
        .pop()
        .ok_or(EmbedError::CountMismatch {
            got: 0,
            expected: 1,
        })
}

/// What [`embed_rows`] made of one batch of rows.
#[derive(Debug)]
pub struct EmbeddedRows {
    /// A vector for each row embedded, and none for each row its own content failed.
    pub embeddings: Vec<Embedding>,
    /// Why the remaining rows were left to a later attempt, when any were.
    pub deferred: Option<EmbedError>,
}

/// Embed the rows' texts in one batch, retrying alone any input that fails it. A row
/// that fails for its own content ([`EmbedError::is_input_caused`]) comes back with no
/// vector; one that fails for any other reason is left out.
pub async fn embed_rows(
    embedder: &dyn Embedder,
    rows: Vec<(EmbeddingTarget, String)>,
) -> EmbeddedRows {
    let texts: Vec<&str> = rows.iter().map(|(_, text)| text.as_str()).collect();
    let results = embed_each(embedder, &texts).await;
    let mut embedded = EmbeddedRows {
        embeddings: Vec::with_capacity(rows.len()),
        deferred: None,
    };
    for ((target, text), result) in rows.into_iter().zip(results) {
        let vector = match result {
            Ok(vector) => Some(vector),
            Err(err) if err.is_input_caused() => {
                tracing::warn!(
                    target: "assistd::embed",
                    row = %target,
                    error = %err,
                    "row cannot be embedded; marking it so it is not retried"
                );
                None
            }
            Err(err) => {
                embedded.deferred.get_or_insert(err);
                continue;
            }
        };
        embedded.embeddings.push(Embedding {
            target,
            text,
            vector,
        });
    }
    embedded
}

#[cfg(test)]
mod tests;
