//! The backlog of saved messages that `index.db` has not cut into chunks.

use std::sync::Arc;

use assistd_utils::backlog::{Backlog, DrainEnd, Page, Pass, Signals, drain, keep_drained};
use assistd_utils::backoff::backoff_delay;
use tokio::sync::watch;

use crate::{MemoryError, MessageId, Result};

use super::connection::SqliteHandle;
use super::reconcile;

/// Messages cut in one write to `index.db`.
const MESSAGES_PER_PAGE: usize = 64;

/// The saved user and assistant messages above the highest one `index.db` has
/// chunked, each with its content. Doing a page cuts their chunks and moves that mark,
/// which is the only way a message gets chunks.
#[derive(Debug, Clone)]
pub struct MessageChunker {
    handle: SqliteHandle,
    /// Whether, with every message chunked, the message vectors were held against
    /// the chunks since this chunker was made.
    unheld_vectors_removed: bool,
}

impl MessageChunker {
    /// Chunker over the shared database handle.
    pub fn new(handle: SqliteHandle) -> Self {
        Self {
            handle,
            unheld_vectors_removed: false,
        }
    }

    /// Cut the messages waiting now and again after each save of one, retrying with
    /// backoff a pass that failed, until `shutdown` reads `true`. The databases must
    /// outlive it.
    pub async fn keep_chunked(self, shutdown: watch::Receiver<bool>) {
        let signals = Signals {
            wake: Arc::clone(self.handle.messages_to_chunk()),
            shutdown,
        };
        keep_drained("message chunking", &signals, backoff_delay, self).await;
    }
}

impl Backlog for MessageChunker {
    type RowId = MessageId;
    type Row = String;
    type Error = MemoryError;

    async fn page_after(&self, after: Option<MessageId>) -> Result<Page<Self>> {
        self.handle
            .read("list unchunked messages", move |c| {
                reconcile::unchunked_messages(c, after, MESSAGES_PER_PAGE)
            })
            .await
    }

    async fn do_page(&mut self, page: Page<Self>) -> Result<usize> {
        let added = self
            .handle
            .write_index("chunk messages", move |tx| {
                reconcile::chunk_messages(tx, &page)
            })
            .await?;
        if added > 0 {
            self.handle.rows_to_embed().notify_one();
        }
        Ok(0)
    }
}

impl Pass for MessageChunker {
    type Error = MemoryError;

    /// Once the first pass to leave no message unchunked has ended, the message
    /// vectors no chunk holds the text of are removed.
    async fn run(&mut self, signals: &Signals) -> Result<DrainEnd> {
        let end = drain(self, signals).await?;
        if matches!(end, DrainEnd::Emptied { .. }) && !self.unheld_vectors_removed {
            self.unheld_vectors_removed = self.handle.remove_unheld_message_vectors().await?;
        }
        Ok(end)
    }
}
