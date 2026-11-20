//! Retrieval over transcript and document chunks and saved memories: by vector, and
//! for documents by their words as well.

use std::fmt;
use std::fs::Metadata;
use std::ops::Range;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

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

/// One document a search found, shown through its best-ranked chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentHit {
    pub chunk_id: ChunkId,
    pub document_id: DocumentId,
    pub path: PathBuf,
    /// The page the chunk is on, counted from zero.
    pub page: usize,
    pub page_count: usize,
    /// The bytes of the page's text the chunk covers.
    pub bytes: Range<usize>,
    pub indexed_at: DateTime<Utc>,
    pub text: String,
    /// The chunk's cosine similarity to the query; `None` while it has no vector.
    pub similarity: Option<f32>,
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
    /// A byte range of one page of an indexed local document's text.
    Document {
        document_id: DocumentId,
        path: PathBuf,
        /// The page's position in the file, counted from zero.
        page: usize,
        /// Pages in the document.
        page_count: usize,
        /// The bytes of the page's text the chunk covers.
        bytes: Range<usize>,
        indexed_at: DateTime<Utc>,
    },
}

impl ChunkLocator {
    /// Whether both chunks were cut from the same source: one session, or one document.
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

/// Part of the stored text of one page of an indexed document.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentText {
    pub text: String,
    /// The page's position in the file, counted from zero.
    pub page: usize,
    /// Pages in the document.
    pub page_count: usize,
    /// The bytes of the page's text that `text` covers.
    pub bytes: Range<usize>,
    /// Length of the page's text in bytes.
    pub page_bytes: usize,
    pub indexed_at: DateTime<Utc>,
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

/// What a chunk's text was cut from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkSource {
    Message,
    Document,
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

/// What produced a set of vectors; only vectors from one space are comparable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingSpace {
    /// The embedding model, as `<owner>/<repo>:<quant>`.
    pub model: String,
    /// Components per vector.
    pub dim: usize,
    /// The extra arguments the embedding server was launched with.
    pub server_args: String,
    /// The text sent before each stored text; a query's own prefix marks no stored vector.
    pub document_prefix: String,
}

/// Rows neither embedded nor marked unembeddable, by what they hold.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnindexedRows {
    pub message_chunks: u64,
    pub document_chunks: u64,
    pub memories: u64,
}

/// What the index holds of one registered index root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexRootCounts {
    pub root: PathBuf,
    pub documents: u64,
    /// Chunks its documents are cut into.
    pub chunks: u64,
    /// Those of its chunks neither embedded nor marked unembeddable.
    pub unembedded_chunks: u64,
}

/// A file's modification time and size; a document whose stored stamp still
/// matches is taken as unchanged without being read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileStamp {
    /// Modification time in nanoseconds since the Unix epoch.
    pub mtime_ns: i64,
    /// Size in bytes.
    pub size: u64,
}

impl FileStamp {
    /// The stamp of the file `metadata` describes.
    pub fn of(metadata: &Metadata) -> Self {
        Self {
            mtime_ns: metadata
                .mtime()
                .saturating_mul(1_000_000_000)
                .saturating_add(metadata.mtime_nsec()),
            size: metadata.len(),
        }
    }
}

/// One walk over an index root's files or over part of them, from
/// [`SemanticStore::begin_index_pass`] to [`SemanticStore::finish_index_pass`] or
/// [`SemanticStore::finish_index_pass_under`]. At most one pass per root runs at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexPass {
    pub(crate) root: i64,
    pub(crate) pass: i64,
}

/// Top-K vector retrieval over embedded chunks and memories, and the index of
/// local documents with its search by vector and keyword.
#[async_trait]
pub trait SemanticStore: fmt::Debug + Send + Sync + 'static {
    /// Top-K messages in `scope` by their best chunk's cosine to the L2-normalised
    /// `query_vector`, each hit carrying that chunk's text. Messages outside `scope`
    /// are dropped before ranking, so never count toward `top_k`.
    async fn nearest_chunks(
        &self,
        query_vector: Vec<f32>,
        top_k: usize,
        scope: SessionScope,
    ) -> Result<Vec<EmbeddingHit>>;

    /// Top-K documents for `query`, best first: their ranking by best chunk's cosine
    /// to the L2-normalised `query_vector` and their ranking by best chunk's BM25
    /// against the words of `query`, fused by reciprocal rank.
    async fn search_documents(
        &self,
        query: String,
        query_vector: Vec<f32>,
        top_k: usize,
    ) -> Result<Vec<DocumentHit>>;

    /// The chunks around `locator` in its source: for a message, the last chunk of
    /// the previous user or assistant message in its session and the first of the next;
    /// for a document, those starting just before and after it, on any of its pages.
    async fn adjacent_chunks(&self, locator: &ChunkLocator) -> Result<AdjacentChunks>;

    /// Top-K saved memories by cosine to the L2-normalised `query_vector`.
    async fn nearest_memories(
        &self,
        query_vector: Vec<f32>,
        top_k: usize,
    ) -> Result<Vec<MemoryHit>>;

    /// Record `space` as the one every stored vector is in. When a different space was
    /// recorded, drop every vector and unembeddable mark and return that space. Must
    /// complete before anything from `space` is stored or searched.
    async fn adopt_embedding_space(&self, space: &EmbeddingSpace)
    -> Result<Option<EmbeddingSpace>>;

    /// Rows neither embedded nor marked unembeddable.
    async fn count_missing(&self) -> Result<UnindexedRows>;

    /// Up to `limit` memories after `after` (from the start when `None`) neither
    /// embedded nor marked unembeddable, in id order, as `(id, value)`.
    async fn memories_missing_embedding(
        &self,
        after: Option<MemoryId>,
        limit: usize,
    ) -> Result<Vec<(MemoryId, String)>>;

    /// Up to `limit` chunks cut from `source` after `after` (from the start when `None`)
    /// neither embedded nor marked unembeddable, in id order, as `(id, chunk text)`.
    async fn chunks_missing_embedding(
        &self,
        source: ChunkSource,
        after: Option<ChunkId>,
        limit: usize,
    ) -> Result<Vec<(ChunkId, String)>>;

    /// Upsert `embeddings` in one transaction, skipping targets deleted since they
    /// were embedded and memories whose value no longer matches. One with no vector
    /// marks its target unembeddable.
    async fn store_embeddings(&self, embeddings: Vec<Embedding>) -> Result<()>;

    /// Clear every unembeddable mark, returning how many, so those rows are missing again.
    async fn clear_unembeddable(&self) -> Result<u64>;

    /// Start a pass over the files under index root `root`, registering the root
    /// when it is new.
    async fn begin_index_pass(&self, root: &Path) -> Result<IndexPass>;

    /// The files among `files`, in their order, that must be read and indexed: those not
    /// stored at their path with their stamp and the current chunking. `pass` has seen
    /// the rest. One transaction covers them all.
    async fn changed_documents(
        &self,
        pass: IndexPass,
        files: Vec<(PathBuf, FileStamp)>,
    ) -> Result<Vec<(PathBuf, FileStamp)>>;

    /// Store `pages`, each page's text in file order, as the document at `path`, seen
    /// by `pass`. No chunk spans two pages. A chunk whose text is unchanged keeps its
    /// vector on whatever page it now sits; returns the new chunks, which have none.
    async fn index_document(
        &self,
        pass: IndexPass,
        path: PathBuf,
        stamp: FileStamp,
        pages: Vec<String>,
    ) -> Result<Vec<ChunkId>>;

    /// Remove every document of `pass`'s root that the pass did not see, returning
    /// how many.
    async fn finish_index_pass(&self, pass: IndexPass) -> Result<u64>;

    /// Remove the documents at or under any of `paths` that `pass` did not see,
    /// returning how many; for a pass that walked only those parts of its root.
    async fn finish_index_pass_under(&self, pass: IndexPass, paths: Vec<PathBuf>) -> Result<u64>;

    /// Remove the document at `path` with its chunks and vectors; `false` when
    /// none was indexed.
    async fn remove_document(&self, path: &Path) -> Result<bool>;

    /// Every registered index root.
    async fn index_roots(&self) -> Result<Vec<PathBuf>>;

    /// What the index holds of every registered index root.
    async fn index_root_counts(&self) -> Result<Vec<IndexRootCounts>>;

    /// Forget index root `root`, removing every document found under it and
    /// returning how many.
    async fn remove_index_root(&self, root: &Path) -> Result<u64>;

    /// How many chunks the stored documents are cut into.
    async fn count_document_chunks(&self) -> Result<u64>;

    /// The stored text of page `page`, counted from zero, of the document indexed at
    /// `path`, within `bytes` of that page: narrowed to whole characters, and empty
    /// past the page's end or the last page. `None` when none is indexed there.
    async fn document_text(
        &self,
        path: &Path,
        page: usize,
        bytes: Range<usize>,
    ) -> Result<Option<DocumentText>>;
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
        _scope: SessionScope,
    ) -> Result<Vec<EmbeddingHit>> {
        Ok(Vec::new())
    }
    async fn search_documents(
        &self,
        _query: String,
        _query_vector: Vec<f32>,
        _top_k: usize,
    ) -> Result<Vec<DocumentHit>> {
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
    async fn adopt_embedding_space(
        &self,
        _space: &EmbeddingSpace,
    ) -> Result<Option<EmbeddingSpace>> {
        Ok(None)
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
        _source: ChunkSource,
        _after: Option<ChunkId>,
        _limit: usize,
    ) -> Result<Vec<(ChunkId, String)>> {
        Ok(Vec::new())
    }
    async fn store_embeddings(&self, _embeddings: Vec<Embedding>) -> Result<()> {
        Ok(())
    }
    async fn clear_unembeddable(&self) -> Result<u64> {
        Ok(0)
    }
    async fn begin_index_pass(&self, _root: &Path) -> Result<IndexPass> {
        Ok(IndexPass { root: 0, pass: 0 })
    }
    async fn changed_documents(
        &self,
        _pass: IndexPass,
        files: Vec<(PathBuf, FileStamp)>,
    ) -> Result<Vec<(PathBuf, FileStamp)>> {
        Ok(files)
    }
    async fn index_document(
        &self,
        _pass: IndexPass,
        _path: PathBuf,
        _stamp: FileStamp,
        _pages: Vec<String>,
    ) -> Result<Vec<ChunkId>> {
        Ok(Vec::new())
    }
    async fn finish_index_pass(&self, _pass: IndexPass) -> Result<u64> {
        Ok(0)
    }
    async fn finish_index_pass_under(&self, _pass: IndexPass, _paths: Vec<PathBuf>) -> Result<u64> {
        Ok(0)
    }
    async fn remove_document(&self, _path: &Path) -> Result<bool> {
        Ok(false)
    }
    async fn index_roots(&self) -> Result<Vec<PathBuf>> {
        Ok(Vec::new())
    }
    async fn index_root_counts(&self) -> Result<Vec<IndexRootCounts>> {
        Ok(Vec::new())
    }
    async fn remove_index_root(&self, _root: &Path) -> Result<u64> {
        Ok(0)
    }
    async fn count_document_chunks(&self) -> Result<u64> {
        Ok(0)
    }
    async fn document_text(
        &self,
        _path: &Path,
        _page: usize,
        _bytes: Range<usize>,
    ) -> Result<Option<DocumentText>> {
        Ok(None)
    }
}
