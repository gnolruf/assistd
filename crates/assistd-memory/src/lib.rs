//! Persistent memory: the key/value, transcript, task, and semantic stores over one
//! SQLite database, and the semantic store's no-op fallback for when embedding is off.

mod chunking;
mod error;
mod ids;
mod memories;
mod migrations;
mod relay_traces;
mod semantic;
mod sqlite;
mod tasks;
mod transcripts;

pub use error::{BackendError, MemoryError, Result};
pub use ids::{BranchId, ChunkId, DocumentId, MemoryId, MessageId, Seq, SessionId, TurnId};
pub use memories::{MemoryRecord, SavedMemory};
pub use relay_traces::{
    PersistedRelayEnd, PersistedRelayEntry, PersistedRelayRun, PersistedRelayStep,
};
pub use semantic::{
    AdjacentChunks, ChunkLocator, ChunkScope, ChunkSource, ContextChunk, DocumentText, Embedding,
    EmbeddingHit, EmbeddingSpace, EmbeddingTarget, FileStamp, IndexPass, IndexRootCounts,
    MemoryHit, NoSemanticStore, SemanticStore, SessionScope, SourcePointer, UnindexedRows,
};
pub use sqlite::{
    MemoryStore, SqliteHandle, SqliteSemanticStore, SqliteTranscriptStore, TaskStore,
};
pub use tasks::{PrunedRuns, RecoveredQueue, TaskRecord, TaskRunEnd, UndecodableTask, WaitingRun};
pub use transcripts::{
    AppendedMessage, BranchHistory, BranchInfo, BranchSummary, HistoryRow, HistoryScope,
    MessageKind, OpenedTurn, PersistedMessage, PersistedRole, PersistedToolCall, Reasoning,
    ResumeCandidate, ToolOutcome, TranscriptStore, TurnStats, UndoOutcome,
};
