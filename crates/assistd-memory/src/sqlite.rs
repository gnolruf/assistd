//! SQLite-backed stores sharing one [`SqliteHandle`]: reads go to the
//! connection directly, writes through a single writer task.

pub mod connection;
pub mod embeddings;
mod ids;
pub mod store;
mod timestamp;
pub mod transcripts;
pub mod writer;

pub use connection::SqliteHandle;
pub use embeddings::{
    EmbeddingHit, MemoryHit, NoSemanticStore, SemanticStore, SqliteSemanticStore, vector_to_blob,
};
pub use ids::{BranchId, ChunkId, MemoryId, MessageId, SessionId, TurnId};
pub use store::SqliteMemoryStore;
pub use transcripts::{
    BranchInfo, BranchSummary, HistoryRow, MessageKind, NoTranscriptStore, PersistedMessage,
    PersistedRole, ResumeCandidate, SqliteTranscriptStore, ToolOutcome, TranscriptStore,
    UndoOutcome,
};
pub use writer::WriteOp;
