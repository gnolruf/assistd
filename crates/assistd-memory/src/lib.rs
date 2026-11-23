//! Persistent memory: the key/value, transcript and task stores over one SQLite
//! database, the semantic store over a second that can be built again from the first
//! and from disk, and that store's no-op fallback for when embedding is off.

mod chunking;
mod error;
mod ids;
mod index_schema;
mod memories;
mod migrations;
mod relay_traces;
mod semantic;
mod sqlite;
mod tasks;
mod transcripts;

pub use error::{BackendError, MemoryError, Result};
pub use ids::{
    BranchId, ChunkId, DocumentId, MemoryId, MessageId, SHORT_SESSION_ID_CHARS, Seq, SessionId,
    TurnId,
};
pub use memories::{MemoryRecord, SavedMemory};
pub use relay_traces::{
    PersistedRelayEnd, PersistedRelayEntry, PersistedRelayRun, PersistedRelayStep,
};
pub use semantic::{
    AdjacentChunks, ChunkLocator, ChunkSearch, ChunkSource, ContextChunk, DocumentStore,
    DocumentText, Embedding, EmbeddingQueue, EmbeddingSpace, EmptyPages, FileStamp, IndexPass,
    IndexRootCounts, NoSemanticStore, PageReading, RecallStore, SearchHit, SearchRanking,
    SemanticStore, SessionScope, SkippedRead, SourcePointer, TranscriptionCounts, UnindexedFile,
    UnindexedRows, WaitingPage,
};
pub use sqlite::{
    MemoryStore, MessageChunker, SqliteHandle, SqliteSemanticStore, SqliteTranscriptStore,
    TaskStore,
};
pub use tasks::{PrunedRuns, RecoveredQueue, TaskRecord, TaskRunEnd, UndecodableTask, WaitingRun};
pub use transcripts::{
    AppendedMessage, BranchHistory, BranchInfo, BranchSummary, HistoryRow, HistoryScope,
    MessageKind, OpenedTurn, PersistedMessage, PersistedRole, PersistedToolCall, Reasoning,
    ResumeCandidate, ToolOutcome, TranscriptStore, TurnStats, UndoOutcome,
};
