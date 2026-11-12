//! Persistent memory: the key/value, transcript, and semantic stores, their
//! SQLite implementations, and no-op fallbacks for when memory is disabled.

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
pub use memories::{MemoryRecord, MemoryStore, NoMemoryStore, SavedMemory};
pub use relay_traces::{
    PersistedRelayEnd, PersistedRelayEntry, PersistedRelayRun, PersistedRelayStep,
};
pub use semantic::{
    AdjacentChunks, ChunkLocator, ChunkScope, ContextChunk, Embedding, EmbeddingHit,
    EmbeddingSpace, EmbeddingTarget, FileStamp, IndexPass, MemoryHit, NoSemanticStore,
    SemanticStore, SessionScope, SourcePointer, UnindexedRows,
};
pub use sqlite::{
    SqliteHandle, SqliteMemoryStore, SqliteSemanticStore, SqliteTaskStore, SqliteTranscriptStore,
};
pub use tasks::{
    NoTaskStore, PrunedRuns, RecoveredQueue, TaskRecord, TaskRunEnd, TaskStore, WaitingRun,
};
pub use transcripts::{
    AppendedMessage, BranchHistory, BranchInfo, BranchSummary, HistoryRow, HistoryScope,
    MessageKind, NoTranscriptStore, OpenedTurn, PersistedMessage, PersistedRole, PersistedToolCall,
    Reasoning, ResumeCandidate, ToolOutcome, TranscriptStore, TurnStats, UndoOutcome,
};
