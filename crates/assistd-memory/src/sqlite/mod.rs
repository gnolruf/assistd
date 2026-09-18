//! SQLite-backed stores sharing one [`SqliteHandle`]: reads go to the
//! connection directly, writes through a single writer task.

pub mod connection;
pub mod conversations;
pub mod embeddings;
pub mod store;
pub mod writer;

pub use connection::SqliteHandle;
pub use conversations::{
    BranchId, BranchInfo, ConversationStore, HistoryRow, NoConversationStore, PersistedMessage,
    PersistedRole, ResumeCandidate, SearchHit, SessionId, SqliteConversationStore, TurnId,
    TurnSummary, UndoOutcome,
};
pub use embeddings::{
    EmbeddingHit, MemoryHit, NoSemanticStore, SemanticStore, SqliteSemanticStore, vector_to_blob,
};
pub use store::SqliteMemoryStore;
pub use writer::WriteOp;
