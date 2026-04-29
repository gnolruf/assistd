//! SQLite-backed concrete implementations of [`crate::MemoryStore`] and
//! the new [`ConversationStore`] trait.
//!
//! Layout:
//! - [`connection`] — opens the DB, applies pragmas, runs migrations,
//!   spawns the writer task, and hands callers a cheaply-cloneable
//!   [`SqliteHandle`] that fans reads to `tokio_rusqlite::Connection`
//!   directly and writes through the dedicated writer task.
//! - [`writer`] — typed [`writer::WriteOp`] enum and the
//!   `tokio::select!` worker loop. Drains its queue on shutdown so an
//!   in-flight `append_message` is never lost across SIGTERM.
//! - [`store`] — [`SqliteMemoryStore`] (the flat KV impl).
//! - [`conversations`] — [`ConversationStore`] trait, the
//!   [`SqliteConversationStore`] impl, and the no-op
//!   [`NoConversationStore`] used when memory is disabled.

pub mod connection;
pub mod conversations;
pub mod embeddings;
pub mod store;
pub mod writer;

pub use connection::SqliteHandle;
pub use conversations::{
    ConversationStore, NoConversationStore, PersistedMessage, PersistedRole, SearchHit, SessionId,
    SqliteConversationStore, TurnId, TurnSummary,
};
pub use embeddings::{
    EmbeddingHit, MemoryHit, NoSemanticStore, SemanticStore, SqliteSemanticStore, vector_to_blob,
};
pub use store::SqliteMemoryStore;
pub use writer::WriteOp;
