//! Retrieval over the chunks cut from transcripts, saved memories and documents: by
//! vector, and for documents by their words as well.

use std::fmt;
use std::fs::Metadata;
use std::ops::Range;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use tokio::sync::Notify;

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
    /// A saved memory, whose whole value is one chunk.
    Memory,
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

/// The L2-normalised embedding of `text` for every chunk cut from `source` that holds
/// it, or no `vector` when the model cannot embed that text.
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    pub source: ChunkSource,
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

/// Chunks of each kind whose text is neither embedded nor marked unembeddable.
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
    /// Those of its chunks whose text is neither embedded nor marked unembeddable.
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

/// Why reading a file left nothing to index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkippedRead {
    Unsupported,
    Binary,
    TooLarge,
    NotDecodable,
    PasswordProtected,
    NoTextLayer,
    Unreadable,
    ExtractorFailed,
    TimedOut,
}

/// A file a pass found that is not stored as a document with the stamp it has.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnindexedFile {
    pub path: PathBuf,
    pub stamp: FileStamp,
    /// Why its read at this stamp left nothing to index; `None` when it has not
    /// been read at this stamp.
    pub skipped: Option<SkippedRead>,
}

/// What a document's pages without text are stored as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmptyPages {
    /// As they are: their file has no more to give.
    Final,
    /// As waiting for transcription: each is an image an OCR model may read.
    AwaitTranscription,
}

/// A page waiting for transcription.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaitingPage {
    pub document_id: DocumentId,
    pub path: PathBuf,
    /// The page's position in the file, counted from zero.
    pub page: usize,
    /// The stamp its file was indexed at.
    pub stamp: FileStamp,
}

/// What transcribing a waiting page came to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageReading {
    /// The text read from the page; empty for a page that shows none.
    Text(String),
    Failed,
}

/// How many pages wait for transcription, and how many it failed on.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TranscriptionCounts {
    pub waiting: u64,
    pub failed: u64,
}

/// One walk over an index root's files or over part of them, from
/// [`DocumentStore::begin_index_pass`] to [`DocumentStore::finish_index_pass`] or
/// [`DocumentStore::finish_index_pass_under`]. At most one pass per root runs at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexPass {
    pub(crate) root: i64,
    pub(crate) pass: i64,
}

/// Top-K retrieval over embedded chunks, memories and documents, and reads of the
/// text around a hit.
#[async_trait]
pub trait RecallStore: fmt::Debug + Send + Sync + 'static {
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

/// The chunk texts still to embed and the vectors stored for them, all in one embedding
/// space. A vector is stored once for its text, whichever chunks of its kind hold it.
#[async_trait]
pub trait EmbeddingQueue: fmt::Debug + Send + Sync + 'static {
    /// Record `space` as the one every stored vector is in. When a different space was
    /// recorded, drop every vector and unembeddable mark and return that space. Must
    /// complete before anything from `space` is stored or searched.
    async fn adopt_embedding_space(&self, space: &EmbeddingSpace)
    -> Result<Option<EmbeddingSpace>>;

    /// Signalled once a committed write has left a row to embed; a write that
    /// leaves none does not signal it.
    fn rows_to_embed(&self) -> Arc<Notify>;

    /// Chunks of each kind whose text is neither embedded nor marked unembeddable.
    async fn count_missing(&self) -> Result<UnindexedRows>;

    /// Up to `limit` chunks cut from `source` after `after` (from the start when `None`)
    /// whose text is neither embedded nor marked unembeddable, in id order, as
    /// `(id, chunk text)`: of the chunks holding one text, only the first.
    async fn chunks_missing_embedding(
        &self,
        source: ChunkSource,
        after: Option<ChunkId>,
        limit: usize,
    ) -> Result<Vec<(ChunkId, String)>>;

    /// Upsert `embeddings` in one transaction, skipping texts no chunk of their kind
    /// holds any more. One with no vector marks its text unembeddable.
    async fn store_embeddings(&self, embeddings: Vec<Embedding>) -> Result<()>;

    /// Clear every unembeddable mark, returning how many, so those rows are missing again.
    async fn clear_unembeddable(&self) -> Result<u64>;
}

/// The index of local documents: the passes over its roots that fill it, the pages
/// waiting for transcription, and what it holds.
#[async_trait]
pub trait DocumentStore: fmt::Debug + Send + Sync + 'static {
    /// Start a pass over the files under index root `root`, registering the root
    /// when it is new.
    async fn begin_index_pass(&self, root: &Path) -> Result<IndexPass>;

    /// The files among `files`, in their order, not stored as documents at their path
    /// with their stamp and the current chunking, each with why when it was skipped at
    /// that stamp. `pass` has seen the rest and the skipped. One transaction covers them all.
    async fn unindexed_files(
        &self,
        pass: IndexPass,
        files: Vec<(PathBuf, FileStamp)>,
    ) -> Result<Vec<UnindexedFile>>;

    /// Record that reading the file at `path` with `stamp` left nothing to index and
    /// why, seen by `pass`, removing the document stored at that path.
    async fn record_skipped_file(
        &self,
        pass: IndexPass,
        path: PathBuf,
        stamp: FileStamp,
        cause: SkippedRead,
    ) -> Result<()>;

    /// Forget every skipped file, returning how many, so the next pass reads each again.
    async fn forget_skipped_files(&self) -> Result<u64>;

    /// Store `pages`, in file order, as the document at `path`, seen by `pass`, replacing
    /// its chunks and dropping what was transcribed of it before and any skip recorded
    /// for it. No chunk spans two pages. Returns the chunks whose text has no vector.
    async fn index_document(
        &self,
        pass: IndexPass,
        path: PathBuf,
        stamp: FileStamp,
        pages: Vec<String>,
        empty_pages: EmptyPages,
    ) -> Result<Vec<ChunkId>>;

    /// Signalled once a committed write has left a page waiting for transcription;
    /// a write that leaves none does not signal it.
    fn pages_to_transcribe(&self) -> Arc<Notify>;

    /// The first page waiting for transcription, by document and then position;
    /// `None` when none waits.
    async fn next_waiting_page(&self) -> Result<Option<WaitingPage>>;

    /// Record `reading` of `page` with its text as the page's; ignored when the page
    /// no longer waits or its file was indexed again. Once no page of the document
    /// waits, its pages that have no chunk are cut, and the new chunks whose text has
    /// no vector returned.
    async fn store_page_reading(
        &self,
        page: WaitingPage,
        reading: PageReading,
    ) -> Result<Vec<ChunkId>>;

    async fn count_transcriptions(&self) -> Result<TranscriptionCounts>;

    /// Have every page whose transcription failed wait again, returning how many.
    async fn retry_failed_transcriptions(&self) -> Result<u64>;

    /// Remove every document of `pass`'s root that the pass did not see, returning
    /// how many, and every skipped file it did not see.
    async fn finish_index_pass(&self, pass: IndexPass) -> Result<u64>;

    /// Remove the documents at or under any of `paths` that `pass` did not see,
    /// returning how many, and the skipped files there it did not see; for a pass
    /// that walked only those parts of its root.
    async fn finish_index_pass_under(&self, pass: IndexPass, paths: Vec<PathBuf>) -> Result<u64>;

    /// Remove the document at `path` with its chunks, and the vectors of the texts
    /// only they held; `false` when none was indexed.
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
}

/// A store that is all three parts at once, as the daemon hands one to their consumers.
pub trait SemanticStore: RecallStore + EmbeddingQueue + DocumentStore {}

impl<S: RecallStore + EmbeddingQueue + DocumentStore> SemanticStore for S {}

/// No-op store: writes are discarded and searches find nothing.
#[derive(Debug)]
pub struct NoSemanticStore;

#[async_trait]
impl RecallStore for NoSemanticStore {
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
    async fn document_text(
        &self,
        _path: &Path,
        _page: usize,
        _bytes: Range<usize>,
    ) -> Result<Option<DocumentText>> {
        Ok(None)
    }
}

#[async_trait]
impl EmbeddingQueue for NoSemanticStore {
    async fn adopt_embedding_space(
        &self,
        _space: &EmbeddingSpace,
    ) -> Result<Option<EmbeddingSpace>> {
        Ok(None)
    }
    fn rows_to_embed(&self) -> Arc<Notify> {
        Arc::default()
    }
    async fn count_missing(&self) -> Result<UnindexedRows> {
        Ok(UnindexedRows::default())
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
}

#[async_trait]
impl DocumentStore for NoSemanticStore {
    async fn begin_index_pass(&self, _root: &Path) -> Result<IndexPass> {
        Ok(IndexPass { root: 0, pass: 0 })
    }
    async fn unindexed_files(
        &self,
        _pass: IndexPass,
        files: Vec<(PathBuf, FileStamp)>,
    ) -> Result<Vec<UnindexedFile>> {
        let unread = |(path, stamp)| UnindexedFile {
            path,
            stamp,
            skipped: None,
        };
        Ok(files.into_iter().map(unread).collect())
    }
    async fn record_skipped_file(
        &self,
        _pass: IndexPass,
        _path: PathBuf,
        _stamp: FileStamp,
        _cause: SkippedRead,
    ) -> Result<()> {
        Ok(())
    }
    async fn forget_skipped_files(&self) -> Result<u64> {
        Ok(0)
    }
    async fn index_document(
        &self,
        _pass: IndexPass,
        _path: PathBuf,
        _stamp: FileStamp,
        _pages: Vec<String>,
        _empty_pages: EmptyPages,
    ) -> Result<Vec<ChunkId>> {
        Ok(Vec::new())
    }
    fn pages_to_transcribe(&self) -> Arc<Notify> {
        Arc::default()
    }
    async fn next_waiting_page(&self) -> Result<Option<WaitingPage>> {
        Ok(None)
    }
    async fn store_page_reading(
        &self,
        _page: WaitingPage,
        _reading: PageReading,
    ) -> Result<Vec<ChunkId>> {
        Ok(Vec::new())
    }
    async fn count_transcriptions(&self) -> Result<TranscriptionCounts> {
        Ok(TranscriptionCounts::default())
    }
    async fn retry_failed_transcriptions(&self) -> Result<u64> {
        Ok(0)
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
}
