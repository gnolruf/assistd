//! SQLite-backed stores sharing one [`SqliteHandle`]: reads go to the
//! connection directly, writes through a single writer task.

mod connection;
mod embeddings;
mod ids;
mod store;
mod timestamp;
mod transcripts;
mod writer;

pub use connection::SqliteHandle;
pub use embeddings::{
    Embedding, EmbeddingHit, EmbeddingTarget, MemoryHit, NoSemanticStore, SemanticStore,
    SqliteSemanticStore, StaleEmbeddings, UnindexedRows,
};
pub use ids::{BranchId, ChunkId, MemoryId, MessageId, SessionId, TurnId};
pub use store::SqliteMemoryStore;
pub use transcripts::{
    BranchInfo, BranchSummary, HistoryRow, MessageKind, NoTranscriptStore, PersistedMessage,
    PersistedRole, ResumeCandidate, SqliteTranscriptStore, ToolOutcome, TranscriptStore,
    UndoOutcome,
};
