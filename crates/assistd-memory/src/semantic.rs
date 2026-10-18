//! Vector retrieval over embedded transcript chunks and saved memories.

use std::fmt;

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::{ChunkId, MemoryId, MessageId, PersistedRole, Result, SessionId};

/// One message hit through its best-scoring chunk. `content` is the full
/// message, not the chunk text, since chunks may cut mid-sentence.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingHit {
    pub message_id: MessageId,
    pub chunk_id: ChunkId,
    pub session_id: SessionId,
    pub timestamp: DateTime<Utc>,
    pub role: PersistedRole,
    pub content: String,
    pub similarity: f32,
}

/// One saved-memory hit, ranked by cosine similarity to the query.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryHit {
    pub memory_id: MemoryId,
    pub key: String,
    pub value: String,
    pub similarity: f32,
}

/// The row an embedding belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingTarget {
    Chunk(ChunkId),
    Memory(MemoryId),
}

impl fmt::Display for EmbeddingTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Chunk(id) => write!(f, "chunk {id}"),
            Self::Memory(id) => write!(f, "memory {id}"),
        }
    }
}

/// An L2-normalised embedding of `text`, the content of `target` when it was embedded.
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    pub target: EmbeddingTarget,
    pub text: String,
    pub vector: Vec<f32>,
}

/// Rows with no embedding, by table.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnindexedRows {
    pub chunks: u64,
    pub memories: u64,
}

/// How [`SemanticStore::adopt_model`] reconciled the stored vectors with a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelAdoption {
    /// The stored vectors already came from this model.
    Unchanged,
    /// No model was recorded; any vectors present were dropped and this model recorded.
    Recorded,
    /// `previous`'s `dropped` vectors were deleted and this model recorded in its place.
    Replaced { previous: String, dropped: u64 },
}

/// Top-K vector retrieval over embedded chunks and memories.
#[async_trait]
pub trait SemanticStore: fmt::Debug + Send + Sync + 'static {
    /// Top-K messages by their best chunk's cosine to the L2-normalised `query_vector`.
    /// `exclude_session`'s messages are dropped before ranking, so never count toward `top_k`.
    async fn nearest_chunks(
        &self,
        query_vector: Vec<f32>,
        top_k: usize,
        exclude_session: Option<SessionId>,
    ) -> Result<Vec<EmbeddingHit>>;

    /// Top-K saved memories by cosine to the L2-normalised `query_vector`.
    async fn nearest_memories(
        &self,
        query_vector: Vec<f32>,
        top_k: usize,
    ) -> Result<Vec<MemoryHit>>;

    /// Record `model` as the source of every stored vector, deleting all vectors in
    /// the same transaction unless they already came from it. Must complete before
    /// anything embedded by `model` is stored or searched.
    async fn adopt_model(&self, model: &str) -> Result<ModelAdoption>;

    /// Rows with no embedding.
    async fn count_missing(&self) -> Result<UnindexedRows>;

    /// Up to `limit` memories after `after` (from the start when `None`) with no
    /// embedding, in id order, as `(id, value)`.
    async fn memories_missing_embedding(
        &self,
        after: Option<MemoryId>,
        limit: usize,
    ) -> Result<Vec<(MemoryId, String)>>;

    /// Up to `limit` message chunks after `after` (from the start when `None`) with
    /// no embedding, in id order, as `(id, chunk text)`.
    async fn chunks_missing_embedding(
        &self,
        after: Option<ChunkId>,
        limit: usize,
    ) -> Result<Vec<(ChunkId, String)>>;

    /// Upsert `embeddings` in one transaction, skipping targets deleted since they
    /// were embedded and memories whose value no longer matches.
    async fn store_embeddings(&self, embeddings: Vec<Embedding>) -> Result<()>;
}

/// No-op store: writes are discarded and searches find nothing.
#[derive(Debug)]
pub struct NoSemanticStore;

#[async_trait]
impl SemanticStore for NoSemanticStore {
    async fn nearest_chunks(
        &self,
        _query_vector: Vec<f32>,
        _top_k: usize,
        _exclude_session: Option<SessionId>,
    ) -> Result<Vec<EmbeddingHit>> {
        Ok(Vec::new())
    }
    async fn nearest_memories(
        &self,
        _query_vector: Vec<f32>,
        _top_k: usize,
    ) -> Result<Vec<MemoryHit>> {
        Ok(Vec::new())
    }
    async fn adopt_model(&self, _model: &str) -> Result<ModelAdoption> {
        Ok(ModelAdoption::Unchanged)
    }
    async fn count_missing(&self) -> Result<UnindexedRows> {
        Ok(UnindexedRows::default())
    }
    async fn memories_missing_embedding(
        &self,
        _after: Option<MemoryId>,
        _limit: usize,
    ) -> Result<Vec<(MemoryId, String)>> {
        Ok(Vec::new())
    }
    async fn chunks_missing_embedding(
        &self,
        _after: Option<ChunkId>,
        _limit: usize,
    ) -> Result<Vec<(ChunkId, String)>> {
        Ok(Vec::new())
    }
    async fn store_embeddings(&self, _embeddings: Vec<Embedding>) -> Result<()> {
        Ok(())
    }
}
