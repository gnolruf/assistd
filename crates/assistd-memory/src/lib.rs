//! Persistent memory: the key/value, transcript, and semantic stores, their
//! SQLite implementations, and no-op fallbacks for when memory is disabled.

mod chunking;
mod error;
mod ids;
mod memories;
mod migrations;
mod semantic;
mod sqlite;
mod transcripts;

pub use error::{BackendError, MemoryError, Result};
pub use ids::{BranchId, ChunkId, MemoryId, MessageId, Seq, SessionId, TurnId};
pub use memories::{MemoryRecord, MemoryStore, NoMemoryStore, SavedMemory};
pub use semantic::{
    Embedding, EmbeddingHit, EmbeddingTarget, MemoryHit, ModelAdoption, NoSemanticStore,
    SemanticStore, UnindexedRows,
};
pub use sqlite::{SqliteHandle, SqliteMemoryStore, SqliteSemanticStore, SqliteTranscriptStore};
pub use transcripts::{
    AppendedMessage, BranchHistory, BranchInfo, BranchSummary, HistoryRow, HistoryScope,
    MessageKind, NoTranscriptStore, OpenedTurn, PersistedMessage, PersistedRole, PersistedToolCall,
    Reasoning, ResumeCandidate, SummaryCut, ToolOutcome, TranscriptStore, TurnStats, UndoOutcome,
};
