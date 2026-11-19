//! One pass over one index root or over the changed parts of it: check the stamp of
//! every file the walk finds, extract and store the changed ones, and remove the
//! documents no longer found.

use std::collections::BTreeMap;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use assistd_documents::{Extraction, Extractor, ExtractorError, SkipReason};
use assistd_memory::{FileStamp, IndexPass, MemoryError, SemanticStore};
use assistd_tools::policy::{PathGuards, on_blocking_pool};
use assistd_utils::readiness::NotReady;
use chrono::{DateTime, Utc};
use thiserror::Error;
use tokio::sync::{Notify, mpsc, watch};
use tracing::debug;

use super::indexer::DocumentIndex;
use super::walk::{IndexRoot, WalkScope, Walked, root_unavailable};

/// Most chunks the stored documents are cut into. A search scans every document
/// vector, about 100 ms and 410 MB of them here; a pass that reaches this stores
/// nothing more.
pub const MAX_DOCUMENT_CHUNKS: u64 = 500_000;

/// Files whose stamps one store transaction checks.
const WALK_BATCH_FILES: usize = 256;

/// Batches the walk may run ahead of extraction.
const WALK_QUEUE_BATCHES: usize = 2;

/// Times the worker may exit over one file before that file is skipped.
const WORKER_EXITS_PER_FILE: u32 = 2;

/// Why a file found under an index root is not in the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SkipCause {
    /// Reading it would need the user's confirmation.
    Sensitive,
    TooLarge,
    Unsupported,
    Binary,
    NotDecodable,
    PasswordProtected,
    NoTextLayer,
    Unreadable,
    ExtractorFailed,
    TimedOut,
    /// The index already holds [`MAX_DOCUMENT_CHUNKS`].
    IndexFull,
}

impl From<&SkipReason> for SkipCause {
    fn from(reason: &SkipReason) -> Self {
        match reason {
            SkipReason::Unsupported { .. } => Self::Unsupported,
            SkipReason::Binary => Self::Binary,
            SkipReason::TooLarge => Self::TooLarge,
            SkipReason::NotDecodable => Self::NotDecodable,
            SkipReason::PasswordProtected => Self::PasswordProtected,
            SkipReason::NoTextLayer => Self::NoTextLayer,
            SkipReason::Unreadable { .. } => Self::Unreadable,
            SkipReason::ExtractorFailed { .. } => Self::ExtractorFailed,
            SkipReason::TimedOut => Self::TimedOut,
        }
    }
}

/// How many files one pass indexed, found unchanged, removed and skipped.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PassCounts {
    /// Files whose text was extracted and stored.
    pub indexed: u64,
    /// Files whose stored stamp still matched, so they were not read.
    pub unchanged: u64,
    /// Documents removed because the pass no longer found their file.
    pub removed: u64,
    pub skipped: BTreeMap<SkipCause, u64>,
}

impl PassCounts {
    fn skip(&mut self, cause: SkipCause) {
        *self.skipped.entry(cause).or_default() += 1;
    }
}

/// How a pass over a root ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PassEnd {
    /// The whole root was walked, and documents no longer found were removed.
    Finished,
    /// The root was missing, unreadable or empty, so its documents were kept;
    /// holds which.
    RootUnavailable(String),
    /// The pass stopped early or could not list part of the root, so no document
    /// was removed; holds why.
    Incomplete(String),
}

/// What the last pass over one index root did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootPass {
    pub root: PathBuf,
    pub ended_at: DateTime<Utc>,
    pub end: PassEnd,
    pub counts: PassCounts,
}

/// What a pass reads from and writes to.
#[derive(Debug, Clone)]
pub struct IndexDeps {
    pub store: Arc<dyn SemanticStore>,
    pub extractor: Arc<Extractor>,
    /// Wakes the embedding worker after a document leaves chunks to embed.
    pub embed_wake: Arc<Notify>,
}

/// Why a pass ended without removing the documents it did not see.
#[derive(Debug, Error)]
enum PassStopped {
    #[error("the daemon is shutting down")]
    Shutdown,
    #[error("document extraction is {0}")]
    Worker(NotReady),
    #[error(transparent)]
    Store(#[from] MemoryError),
    #[error("part of the folder could not be listed: {0}")]
    Unlisted(String),
    #[error("the folder went away during the pass: {0}")]
    RootGone(String),
}

struct RunningPass<'a> {
    deps: &'a IndexDeps,
    index: &'a DocumentIndex,
    root: &'a Path,
    pass: IndexPass,
    max_file_bytes: u64,
    counts: &'a mut PassCounts,
    /// The first part of the root the walk could not list.
    unlisted: Option<String>,
    shutdown: watch::Receiver<bool>,
    worker: watch::Receiver<Result<(), NotReady>>,
}

impl RunningPass<'_> {
    async fn store_batch(&mut self, batch: Vec<Walked>) -> Result<(), PassStopped> {
        if *self.shutdown.borrow() {
            return Err(PassStopped::Shutdown);
        }
        let mut files = Vec::with_capacity(batch.len());
        for walked in batch {
            match walked {
                Walked::Candidate { path, stamp } => files.push((path, stamp)),
                Walked::Skipped(cause) => self.counts.skip(cause),
                Walked::Unlisted(error) => {
                    self.unlisted.get_or_insert(error);
                }
            }
        }
        let walked = files.len() as u64;
        let changed = self.deps.store.changed_documents(self.pass, files).await?;
        self.counts.unchanged += walked - changed.len() as u64;
        self.index.report_progress(self.root, self.counts);
        if changed.is_empty() {
            return Ok(());
        }
        let mut chunks = self.deps.store.count_document_chunks().await?;
        for (path, stamp) in changed {
            if chunks >= MAX_DOCUMENT_CHUNKS {
                self.counts.skip(SkipCause::IndexFull);
            } else {
                chunks += self.index_file(path, stamp).await?;
            }
            self.index.report_progress(self.root, self.counts);
        }
        Ok(())
    }

    /// Extract and store the file at `path`, returning how many chunks that added.
    async fn index_file(&mut self, path: PathBuf, stamp: FileStamp) -> Result<u64, PassStopped> {
        let pages = match self.extract(&path).await? {
            Ok(pages) => pages,
            Err(reason) => {
                debug!(target: "assistd::documents", "skipped {}: {reason}", path.display());
                self.counts.skip(SkipCause::from(&reason));
                return Ok(0);
            }
        };
        let store = &self.deps.store;
        let added = store.index_document(self.pass, path, stamp, pages).await?;
        self.counts.indexed += 1;
        if !added.is_empty() {
            self.deps.embed_wake.notify_one();
        }
        Ok(added.len() as u64)
    }

    /// The pages of the file at `path`, or why it is not indexed. A worker that
    /// exits is waited for and asked again, and the file is skipped once the worker
    /// has exited over it [`WORKER_EXITS_PER_FILE`] times.
    async fn extract(
        &mut self,
        path: &Path,
    ) -> Result<Result<Vec<String>, SkipReason>, PassStopped> {
        let mut worker_exits = 0;
        loop {
            self.worker_ready().await?;
            let extractor = &self.deps.extractor;
            match extractor.extract(path, self.max_file_bytes).await {
                Ok(extraction) => return Ok(indexable_pages(extraction)),
                Err(ExtractorError::NotReady(_)) => {}
                Err(ExtractorError::Exited) => {
                    worker_exits += 1;
                    if worker_exits == WORKER_EXITS_PER_FILE {
                        return Ok(Err(SkipReason::ExtractorFailed {
                            message: "the extraction worker exited while reading it".into(),
                        }));
                    }
                }
            }
        }
    }

    /// Wait out a worker that is starting; errors on shutdown, or when no worker
    /// is coming.
    async fn worker_ready(&mut self) -> Result<(), PassStopped> {
        let started = |readiness: &Result<(), NotReady>| *readiness != Err(NotReady::Starting);
        tokio::select! {
            biased;
            _ = self.shutdown.wait_for(|stop| *stop) => Err(PassStopped::Shutdown),
            readiness = self.worker.wait_for(started) => match readiness.as_deref() {
                Ok(Ok(())) => Ok(()),
                Ok(Err(not_ready)) => Err(PassStopped::Worker(not_ready.clone())),
                Err(_) => Err(PassStopped::Shutdown),
            },
        }
    }
}

/// Run one pass over `scope` of `root`, leaving out the files `guards` call sensitive
/// and reporting its progress to `index`. Stops between files once `shutdown` flips.
pub(super) async fn index_root(
    deps: &IndexDeps,
    index: &DocumentIndex,
    root: Arc<IndexRoot>,
    scope: WalkScope,
    guards: Arc<PathGuards>,
    shutdown: &watch::Receiver<bool>,
) -> RootPass {
    let mut counts = PassCounts::default();
    let end = match unavailable(&root).await {
        Some(why) => PassEnd::RootUnavailable(why),
        None => {
            let stored = walk_and_store(deps, index, &root, scope, guards, shutdown, &mut counts);
            match stored.await {
                Ok(()) => PassEnd::Finished,
                Err(stopped) => PassEnd::Incomplete(stopped.to_string()),
            }
        }
    };
    RootPass {
        root: root.path().to_path_buf(),
        ended_at: Utc::now(),
        end,
        counts,
    }
}

async fn unavailable(root: &Arc<IndexRoot>) -> Option<String> {
    let root = Arc::clone(root);
    on_blocking_pool(move || root_unavailable(root.path())).await
}

/// Begin a pass, store what the walk finds, and finish the pass only when all of
/// `scope` was listed and the root is still there.
async fn walk_and_store(
    deps: &IndexDeps,
    index: &DocumentIndex,
    root: &Arc<IndexRoot>,
    scope: WalkScope,
    guards: Arc<PathGuards>,
    shutdown: &watch::Receiver<bool>,
    counts: &mut PassCounts,
) -> Result<(), PassStopped> {
    let pass = deps.store.begin_index_pass(root.path()).await?;
    let mut running = RunningPass {
        deps,
        index,
        root: root.path(),
        pass,
        max_file_bytes: root.max_file_bytes(),
        counts,
        unlisted: None,
        shutdown: shutdown.clone(),
        worker: deps.extractor.subscribe_readiness(),
    };
    let (batches_tx, batches) = mpsc::channel(WALK_QUEUE_BATCHES);
    let (walked_root, walked_scope) = (Arc::clone(root), scope.clone());
    let walk = on_blocking_pool(move || {
        walk_in_batches(&walked_root, &walked_scope, &guards, &batches_tx);
    });
    let ((), stored) = tokio::join!(walk, store_batches(&mut running, batches));
    stored?;
    if let Some(error) = running.unlisted.take() {
        return Err(PassStopped::Unlisted(error));
    }
    if let Some(why) = unavailable(root).await {
        return Err(PassStopped::RootGone(why));
    }
    running.counts.removed = match scope {
        WalkScope::WholeRoot => deps.store.finish_index_pass(pass).await?,
        WalkScope::Under(paths) => {
            let paths = Arc::unwrap_or_clone(paths).into_iter().collect();
            deps.store.finish_index_pass_under(pass, paths).await?
        }
    };
    Ok(())
}

/// Store every batch the walk sends. Returning drops the receiver, which stops
/// the walk.
async fn store_batches(
    running: &mut RunningPass<'_>,
    mut batches: mpsc::Receiver<Vec<Walked>>,
) -> Result<(), PassStopped> {
    while let Some(batch) = batches.recv().await {
        running.store_batch(batch).await?;
    }
    Ok(())
}

/// Walk `scope` of `root` on the calling thread, sending what it finds in batches
/// of [`WALK_BATCH_FILES`] until the receiver is dropped.
fn walk_in_batches(
    root: &IndexRoot,
    scope: &WalkScope,
    guards: &PathGuards,
    batches: &mpsc::Sender<Vec<Walked>>,
) {
    let mut batch = Vec::with_capacity(WALK_BATCH_FILES);
    root.walk(scope, guards, |walked| {
        batch.push(walked);
        if batch.len() < WALK_BATCH_FILES {
            return ControlFlow::Continue(());
        }
        let full = std::mem::replace(&mut batch, Vec::with_capacity(WALK_BATCH_FILES));
        match batches.blocking_send(full) {
            Ok(()) => ControlFlow::Continue(()),
            Err(_) => ControlFlow::Break(()),
        }
    });
    if !batch.is_empty() {
        let _ = batches.blocking_send(batch);
    }
}

/// The pages to store for `extraction`. A file none of whose pages has text is
/// skipped as [`SkipReason::NoTextLayer`].
fn indexable_pages(extraction: Extraction) -> Result<Vec<String>, SkipReason> {
    match extraction {
        Extraction::Pages { pages, .. } if pages.iter().any(|page| !page.trim().is_empty()) => {
            Ok(pages)
        }
        Extraction::Pages { .. } => Err(SkipReason::NoTextLayer),
        Extraction::Skipped { reason } => Err(reason),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use assistd_config::DocumentsConfig;
    use assistd_memory::{SqliteHandle, SqliteSemanticStore};

    use super::*;

    #[tokio::test]
    async fn a_pass_removes_documents_whose_files_are_gone_but_none_of_a_root_that_is() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root_dir = temp.path().join("notes");
        let (kept, deleted) = (root_dir.join("kept.md"), root_dir.join("deleted.md"));
        fs::create_dir(&root_dir).expect("root");
        fs::write(&kept, "kept").expect("file");
        let stamp = FileStamp::of(&fs::metadata(&kept).expect("metadata"));
        let handle = SqliteHandle::open(&temp.path().join("memory.db"))
            .await
            .expect("database");
        let store = Arc::new(SqliteSemanticStore::new(handle));
        let seeded = store.begin_index_pass(&root_dir).await.unwrap();
        for path in [&kept, &deleted] {
            let chunks = store
                .index_document(seeded, path.clone(), stamp, vec!["kept".into()])
                .await
                .unwrap();
            assert_eq!(chunks.len(), 1);
        }
        let deps = IndexDeps {
            store: store.clone(),
            extractor: Arc::new(Extractor::starting()),
            embed_wake: Arc::default(),
        };
        let root = Arc::new(IndexRoot::new(&root_dir, &DocumentsConfig::default()).unwrap());
        let guards = Arc::new(PathGuards::default());
        let (_stop, shutdown) = watch::channel(false);
        let index = DocumentIndex::default();
        let pass = || {
            index_root(
                &deps,
                &index,
                root.clone(),
                WalkScope::WholeRoot,
                guards.clone(),
                &shutdown,
            )
        };

        let present = pass().await;
        assert_eq!(present.end, PassEnd::Finished);
        assert_eq!((present.counts.unchanged, present.counts.removed), (1, 1));
        assert_eq!(store.count_document_chunks().await.unwrap(), 1);

        fs::remove_file(&kept).expect("emptied root");
        let emptied = pass().await;
        fs::remove_dir(&root_dir).expect("removed root");
        let missing = pass().await;
        for absent in [emptied, missing] {
            assert!(
                matches!(absent.end, PassEnd::RootUnavailable(_)),
                "{absent:?}"
            );
            assert_eq!(absent.counts, PassCounts::default());
        }
        assert_eq!(store.count_document_chunks().await.unwrap(), 1);
    }
}
