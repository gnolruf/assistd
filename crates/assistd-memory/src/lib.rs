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

pub use chunking::{ChunkingConfig, chunk_ranges};
pub use error::{MemoryError, Result};
pub use ids::{BranchId, ChunkId, MemoryId, MessageId, SessionId, TurnId};
pub use memories::{MemoryRecord, MemoryStore, NoMemoryStore};
pub use semantic::{
    Embedding, EmbeddingHit, EmbeddingTarget, MemoryHit, NoSemanticStore, SemanticStore,
    StaleEmbeddings, UnindexedRows,
};
pub use sqlite::{SqliteHandle, SqliteMemoryStore, SqliteSemanticStore, SqliteTranscriptStore};
pub use transcripts::{
    BranchInfo, BranchSummary, HistoryRow, MessageKind, NoTranscriptStore, PersistedMessage,
    PersistedRole, ResumeCandidate, ToolOutcome, TranscriptStore, UndoOutcome,
};
