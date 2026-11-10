//! Vector retrieval over embedded transcript and document chunks and saved memories.

use std::fmt;
use std::ops::Range;
use std::path::PathBuf;

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::{ChunkId, DocumentId, MemoryId, MessageId, PersistedRole, Result, SessionId};

const POINTER_DATE_FORMAT: &str = "%Y-%m-%d";

/// One matched chunk: its text, its cosine similarity to the query, and where it came from.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingHit {
    pub chunk_id: ChunkId,
    pub locator: ChunkLocator,
    pub text: String,
    pub similarity: f32,
}

/// The source a matched chunk was cut from.
#[derive(Debug, Clone, PartialEq)]
pub enum ChunkLocator {
    /// A stored transcript message.
    Message {
        session_id: SessionId,
        message_id: MessageId,
        role: PersistedRole,
        timestamp: DateTime<Utc>,
        /// The session's title, or its first user message cut short when untitled.
        session_label: String,
    },
    /// A byte range of an indexed local document's text.
    Document {
        document_id: DocumentId,
        path: PathBuf,
        bytes: Range<usize>,
        indexed_at: DateTime<Utc>,
    },
}

impl ChunkLocator {
    /// Whether both chunks were cut from the same source: one chat session, or one document.
    pub fn same_source(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Message { session_id, .. },
                Self::Message {
                    session_id: other, ..
                },
            ) => session_id == other,
            (
                Self::Document { document_id, .. },
                Self::Document {
                    document_id: other, ..
                },
            ) => document_id == other,
            _ => false,
        }
    }
}

/// A source matched by one or more chunks, listed once so it can be read on
/// demand instead of quoted. Displays as kind, short id, label, date and match count.
#[derive(Debug, Clone, PartialEq)]
pub struct SourcePointer {
    /// Where the source's best-ranked match sits.
    pub locator: ChunkLocator,
    pub matches: usize,
}

impl SourcePointer {
    /// One pointer per source among `hits`, in order of each source's first hit.
    pub fn group<'a>(hits: impl IntoIterator<Item = &'a EmbeddingHit>) -> Vec<Self> {
        let mut pointers: Vec<Self> = Vec::new();
        for hit in hits {
            match pointers
                .iter_mut()
                .find(|pointer| pointer.locator.same_source(&hit.locator))
            {
                Some(pointer) => pointer.matches += 1,
                None => pointers.push(Self {
                    locator: hit.locator.clone(),
                    matches: 1,
                }),
            }
        }
        pointers
    }
}

impl fmt::Display for SourcePointer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let plural = if self.matches == 1 { "" } else { "s" };
        match &self.locator {
            ChunkLocator::Message {
                session_id,
                timestamp,
                session_label,
                ..
            } => write!(
                f,
                "chat {} \"{}\" {}, {} matching message{plural}",
                session_id.short(),
                session_label.replace('\n', " "),
                timestamp.format(POINTER_DATE_FORMAT),
                self.matches,
            ),
            ChunkLocator::Document {
                path, indexed_at, ..
            } => write!(
                f,
                "document \"{}\" {}, {} matching section{plural}",
                path.display(),
                indexed_at.format(POINTER_DATE_FORMAT),
                self.matches,
            ),
        }
    }
}

/// A chunk shown beside a hit so the match reads in context.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextChunk {
    pub chunk_id: ChunkId,
    pub locator: ChunkLocator,
    pub text: String,
}

/// The chunks immediately before and after a hit in its source.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AdjacentChunks {
    pub before: Option<ContextChunk>,
    pub after: Option<ContextChunk>,
}

/// Which sessions a chunk search ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionScope {
    All,
    AllExcept(SessionId),
    /// Every session, plus only those messages of this one that its current
    /// branch's summary folded away (seq before `kept_from_seq`).
    AllExceptInContext(SessionId),
    Only(SessionId),
}

/// Which sources a chunk search ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkScope {
    /// Chat messages from the sessions in scope.
    Chats(SessionScope),
    /// Every indexed document.
    Documents,
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

/// The L2-normalised embedding of `text`, the content of `target` when it was embedded,
/// or no `vector` when the model cannot embed that text.
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    pub target: EmbeddingTarget,
    pub text: String,
    pub vector: Option<Vec<f32>>,
}

/// Rows neither embedded nor marked unembeddable, by table.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnindexedRows {
    pub chunks: u64,
    pub memories: u64,
}

/// Top-K vector retrieval over embedded chunks and memories.
#[async_trait]
pub trait SemanticStore: fmt::Debug + Send + Sync + 'static {
    /// Top-K messages or documents in `scope` by their best chunk's cosine to the
    /// L2-normalised `query_vector`, each hit carrying that chunk's text.
    /// Sources outside `scope` are dropped before ranking, so never count toward `top_k`.
    async fn nearest_chunks(
        &self,
        query_vector: Vec<f32>,
        top_k: usize,
        scope: ChunkScope,
    ) -> Result<Vec<EmbeddingHit>>;

    /// The chunks around `locator` in its source: for a message, the last chunk of
    /// the previous user or assistant message in its session and the first of the next;
    /// for a document, the chunks starting just before and just after it.
    async fn adjacent_chunks(&self, locator: &ChunkLocator) -> Result<AdjacentChunks>;

    /// Top-K saved memories by cosine to the L2-normalised `query_vector`.
    async fn nearest_memories(
        &self,
        query_vector: Vec<f32>,
        top_k: usize,
    ) -> Result<Vec<MemoryHit>>;

    /// Delete every vector not embedded by `model`, returning how many. Must complete
    /// before anything embedded by `model` is searched, so searches never compare
    /// vectors from two models.
    async fn drop_other_models(&self, model: &str) -> Result<u64>;

    /// Rows neither embedded nor marked unembeddable.
    async fn count_missing(&self) -> Result<UnindexedRows>;

    /// Up to `limit` memories after `after` (from the start when `None`) neither
    /// embedded nor marked unembeddable, in id order, as `(id, value)`.
    async fn memories_missing_embedding(
        &self,
        after: Option<MemoryId>,
        limit: usize,
    ) -> Result<Vec<(MemoryId, String)>>;

    /// Up to `limit` chunks after `after` (from the start when `None`) neither
    /// embedded nor marked unembeddable, in id order, as `(id, chunk text)`.
    async fn chunks_missing_embedding(
        &self,
        after: Option<ChunkId>,
        limit: usize,
    ) -> Result<Vec<(ChunkId, String)>>;

    /// Upsert `embeddings` from `model` in one transaction, skipping targets deleted
    /// since they were embedded and memories whose value no longer matches. One with
    /// no vector marks its target unembeddable by `model`.
    async fn store_embeddings(&self, model: &str, embeddings: Vec<Embedding>) -> Result<()>;

    /// Clear every unembeddable mark, returning how many, so those rows are missing again.
    async fn clear_unembeddable(&self) -> Result<u64>;
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
        _scope: ChunkScope,
    ) -> Result<Vec<EmbeddingHit>> {
        Ok(Vec::new())
    }
    async fn adjacent_chunks(&self, _locator: &ChunkLocator) -> Result<AdjacentChunks> {
        Ok(AdjacentChunks::default())
    }
    async fn nearest_memories(
        &self,
        _query_vector: Vec<f32>,
        _top_k: usize,
    ) -> Result<Vec<MemoryHit>> {
        Ok(Vec::new())
    }
    async fn drop_other_models(&self, _model: &str) -> Result<u64> {
        Ok(0)
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
    async fn store_embeddings(&self, _model: &str, _embeddings: Vec<Embedding>) -> Result<()> {
        Ok(())
    }
    async fn clear_unembeddable(&self) -> Result<u64> {
        Ok(0)
    }
}
