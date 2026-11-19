//! Handlers for the `Documents*` variants of `Request`, and the broadcast of what the
//! document index is doing.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::watch;
use tracing::debug;

use assistd_config::Config;
use assistd_documents::Extractor;
use assistd_embed::EmbedderHandle;
use assistd_ipc::{
    DocumentActivity, DocumentPass, DocumentPassCounts, DocumentPassEnd, DocumentRootStatus,
    DocumentSkipReason, Event, EventBody, SkippedFiles,
};
use assistd_memory::{ChunkLocator, ChunkScope, EmbeddingHit, IndexRootCounts, SemanticStore};
use assistd_tools::policy::on_blocking_pool;
use assistd_utils::readiness::NotReady;

use super::memory_stack::MemoryStack;
use super::subsystems::Subsystems;
use super::{DispatchError, FailedStep};
use crate::bus::{EventBus, EventSink};
use crate::documents::{
    DocumentIndex, IndexProgress, MAX_DOCUMENT_CHUNKS, PassCounts, PassEnd, RootPass, SkipCause,
    root_unavailable,
};

const ACTIVITY_EVENT_ID: &str = "documents";

/// How often the chunks waiting for the embedder are counted again while a pass runs
/// or some wait.
const ACTIVITY_RECOUNT_INTERVAL: Duration = Duration::from_secs(5);

/// Handles the `Documents*` requests over the document index, and broadcasts what the
/// index is doing.
#[derive(Debug)]
pub struct DocumentsService {
    config: Arc<Config>,
    extractor: Arc<Extractor>,
    index: Arc<DocumentIndex>,
    embedder: Arc<EmbedderHandle>,
    semantic: Arc<dyn SemanticStore>,
    activity: watch::Sender<DocumentActivity>,
    bus: EventBus,
}

impl DocumentsService {
    pub(super) fn new(
        config: Arc<Config>,
        subsystems: &Subsystems,
        memory: &MemoryStack,
        bus: EventBus,
    ) -> Self {
        Self {
            config,
            extractor: Arc::clone(&subsystems.documents),
            index: Arc::clone(&subsystems.document_index),
            embedder: Arc::clone(&memory.embedder),
            semantic: Arc::clone(&memory.semantic),
            activity: watch::Sender::default(),
            bus,
        }
    }

    /// Follows what [`Self::broadcast_activity`] last broadcast.
    pub(super) fn subscribe_activity(&self) -> watch::Receiver<DocumentActivity> {
        self.activity.subscribe()
    }

    /// Broadcast what the document index is doing each time it changes, until
    /// `shutdown` flips. The chunks waiting for the embedder are counted when a pass
    /// starts or ends, and every few seconds while one runs or some wait.
    pub async fn broadcast_activity(&self, mut shutdown: watch::Receiver<bool>) {
        let mut progress = self.index.subscribe_progress();
        loop {
            let (rounds_ended, root) = {
                let now = progress.borrow_and_update();
                (now.rounds_ended, now.running_root().map(Path::to_path_buf))
            };
            let activity = DocumentActivity {
                indexing: root.as_deref().map(wire_path),
                unembedded_chunks: self.chunks_awaiting_embedder().await,
            };
            let recount_due = !activity.is_idle();
            self.broadcast_if_changed(activity);
            let moved_on = |now: &IndexProgress| {
                now.rounds_ended != rounds_ended || now.running_root() != root.as_deref()
            };
            tokio::select! {
                biased;
                _ = shutdown.wait_for(|stop| *stop) => return,
                _ = progress.wait_for(moved_on) => {}
                () = tokio::time::sleep(ACTIVITY_RECOUNT_INTERVAL), if recount_due => {}
            }
        }
    }

    /// Document chunks the embedder has yet to embed: none while there is no embedder,
    /// and the last count when they cannot be counted.
    async fn chunks_awaiting_embedder(&self) -> u64 {
        if matches!(self.embedder.readiness(), Err(NotReady::Unavailable(_))) {
            return 0;
        }
        match self.semantic.count_missing().await {
            Ok(missing) => missing.document_chunks,
            Err(error) => {
                debug!(
                    target: "assistd::documents",
                    "cannot count the document chunks left to embed: {error}"
                );
                self.activity.borrow().unembedded_chunks
            }
        }
    }

    fn broadcast_if_changed(&self, activity: DocumentActivity) {
        let changed = self.activity.send_if_modified(|current| {
            let changed = *current != activity;
            if changed {
                current.clone_from(&activity);
            }
            changed
        });
        if changed {
            self.bus.publish(&Event::new(
                ACTIVITY_EVENT_ID,
                EventBody::DocumentActivity { activity },
            ));
        }
    }

    pub(super) async fn handle_documents_status(
        &self,
        id: String,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let stored = self
            .semantic
            .index_root_counts()
            .await
            .context("document status failed")?;
        let running_pass = self
            .index
            .subscribe_progress()
            .borrow()
            .running_root()
            .map(wire_path);
        let index = EventBody::DocumentIndex {
            indexing: self.extractor.readiness().into(),
            embedding: self.embedder.readiness().into(),
            chunks: stored.iter().map(|counts| counts.chunks).sum(),
            chunk_limit: MAX_DOCUMENT_CHUNKS,
            running_pass,
        };
        let _ = tx.send(Event::new(id.clone(), index)).await;
        let last_passes = self.index.last_passes();
        for (root, unavailable) in self.roots_with_availability().await {
            let status = root_status(&root, unavailable, &stored, &last_passes);
            let _ = tx
                .send(Event::new(id.clone(), EventBody::DocumentRoot { status }))
                .await;
        }
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
        Ok(())
    }

    /// Every `documents.roots` entry with why its files cannot be listed now, if they
    /// cannot.
    async fn roots_with_availability(&self) -> Vec<(PathBuf, Option<String>)> {
        let roots: Vec<PathBuf> = self
            .config
            .documents
            .roots
            .iter()
            .map(|root| root.as_path().to_path_buf())
            .collect();
        on_blocking_pool(move || {
            roots
                .into_iter()
                .map(|root| {
                    let unavailable = root_unavailable(&root);
                    (root, unavailable)
                })
                .collect()
        })
        .await
    }

    /// Ask for a round of passes and stream the one that answers until it ends or the
    /// client goes away. Errors when no extraction worker will run.
    pub(super) async fn handle_documents_reindex(
        &self,
        id: String,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        if let Err(not_ready @ NotReady::Unavailable(_)) = self.extractor.readiness() {
            return Err(DispatchError::DocumentsUnavailable(not_ready));
        }
        let mut progress = self.index.subscribe_progress();
        let answered_at = self.index.request_pass();
        loop {
            let now = progress.borrow_and_update().clone();
            if now.rounds_ended >= answered_at {
                break;
            }
            if now.rounds_ended + 1 == answered_at
                && let Some(pass) = now.running
            {
                let body = EventBody::DocumentPassProgress {
                    root: wire_path(&pass.root),
                    counts: wire_counts(&pass.counts),
                };
                let _ = tx.send(Event::new(id.clone(), body)).await;
            }
            tokio::select! {
                () = tx.closed() => return Ok(()),
                changed = progress.changed() => {
                    if changed.is_err() {
                        break;
                    }
                }
            }
        }
        for pass in self.index.last_passes() {
            let body = EventBody::DocumentPassEnded {
                root: wire_path(&pass.root),
                pass: wire_pass(&pass),
            };
            let _ = tx.send(Event::new(id.clone(), body)).await;
        }
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
        Ok(())
    }

    pub(super) async fn handle_documents_search(
        &self,
        id: String,
        query: String,
        limit: u32,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let embedder = self.embedder.get().context("document search failed")?;
        let limit = match limit {
            0 => self.config.embedding.top_k.get(),
            limit => limit,
        };
        let query_vector = embedder.embed_query(&query).await.context("embed failed")?;
        let hits = self
            .semantic
            .nearest_chunks(query_vector, limit as usize, ChunkScope::Documents)
            .await
            .context("document search failed")?;
        for body in hits.into_iter().filter_map(wire_hit) {
            let _ = tx.send(Event::new(id.clone(), body)).await;
        }
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
        Ok(())
    }
}

/// Where `root` stands: what `stored` holds of it and what the last of `last_passes`
/// over it did.
fn root_status(
    root: &Path,
    unavailable: Option<String>,
    stored: &[IndexRootCounts],
    last_passes: &[RootPass],
) -> DocumentRootStatus {
    let stored = stored.iter().find(|counts| counts.root == root);
    DocumentRootStatus {
        root: wire_path(root),
        unavailable,
        documents: stored.map_or(0, |counts| counts.documents),
        chunks: stored.map_or(0, |counts| counts.chunks),
        unembedded_chunks: stored.map_or(0, |counts| counts.unembedded_chunks),
        last_pass: last_passes
            .iter()
            .find(|pass| pass.root == root)
            .map(wire_pass),
    }
}

fn wire_path(path: &Path) -> String {
    path.display().to_string()
}

fn wire_pass(pass: &RootPass) -> DocumentPass {
    DocumentPass {
        ended_at: pass.ended_at,
        end: match &pass.end {
            PassEnd::Finished => DocumentPassEnd::Finished,
            PassEnd::RootUnavailable(reason) => DocumentPassEnd::RootUnavailable {
                reason: reason.clone(),
            },
            PassEnd::Incomplete(reason) => DocumentPassEnd::Incomplete {
                reason: reason.clone(),
            },
        },
        counts: wire_counts(&pass.counts),
    }
}

fn wire_counts(counts: &PassCounts) -> DocumentPassCounts {
    DocumentPassCounts {
        indexed: counts.indexed,
        unchanged: counts.unchanged,
        removed: counts.removed,
        skipped: counts
            .skipped
            .iter()
            .map(|(cause, files)| SkippedFiles {
                reason: wire_skip_reason(*cause),
                files: *files,
            })
            .collect(),
    }
}

fn wire_skip_reason(cause: SkipCause) -> DocumentSkipReason {
    match cause {
        SkipCause::Sensitive => DocumentSkipReason::Sensitive,
        SkipCause::TooLarge => DocumentSkipReason::TooLarge,
        SkipCause::Unsupported => DocumentSkipReason::Unsupported,
        SkipCause::Binary => DocumentSkipReason::Binary,
        SkipCause::NotDecodable => DocumentSkipReason::NotDecodable,
        SkipCause::PasswordProtected => DocumentSkipReason::PasswordProtected,
        SkipCause::NoTextLayer => DocumentSkipReason::NoTextLayer,
        SkipCause::Unreadable => DocumentSkipReason::Unreadable,
        SkipCause::ExtractorFailed => DocumentSkipReason::ExtractorFailed,
        SkipCause::TimedOut => DocumentSkipReason::TimedOut,
        SkipCause::IndexFull => DocumentSkipReason::IndexFull,
    }
}

/// `hit` as a `DocumentHit` event; `None` for a hit that is not in a document.
fn wire_hit(hit: EmbeddingHit) -> Option<EventBody> {
    let ChunkLocator::Document {
        path,
        page,
        page_count,
        bytes,
        indexed_at,
        ..
    } = hit.locator
    else {
        return None;
    };
    Some(EventBody::DocumentHit {
        path: wire_path(&path),
        page: page as u64 + 1,
        page_count: page_count as u64,
        byte_start: bytes.start as u64,
        byte_end: bytes.end as u64,
        indexed_at,
        content: hit.text,
        similarity: hit.similarity,
    })
}
