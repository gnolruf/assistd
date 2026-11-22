//! When index passes run: over every root after each request, and between requests
//! over what a root's watch reports, one root at a time, until shutdown.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use assistd_config::{AbsoluteDir, DocumentsConfig};
use assistd_memory::DocumentStore;
use assistd_tools::SandboxInfo;
use assistd_tools::policy::{PathGuards, on_blocking_pool};
use parking_lot::Mutex;
use tokio::sync::{Notify, watch};
use tracing::{debug, info, warn};

use super::DocumentsUnavailable;
use super::pass::{IndexDeps, PassCounts, PassEnd, RootPass, SkipCause, index_root};
use super::transcribe::PageTranscription;
use super::walk::{IndexRoot, WalkScope};
use super::watch::RootWatches;

/// The pass under way over one index root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassProgress {
    pub root: PathBuf,
    /// What the pass has done so far.
    pub counts: PassCounts,
}

/// How far the index passes have got since the daemon started.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexProgress {
    /// Rounds of one pass over every root that have ended.
    pub rounds_ended: u64,
    /// Whether such a round is under way. A root's watch has passes run outside one.
    pub round_running: bool,
    /// The pass under way; `None` between passes.
    pub running: Option<PassProgress>,
}

impl IndexProgress {
    /// The root a pass is walking.
    pub fn running_root(&self) -> Option<&Path> {
        self.running.as_ref().map(|pass| pass.root.as_path())
    }
}

/// The document index as the rest of the daemon sees it: where a pass is asked
/// for, how far the passes have got, what the last pass over each root did,
/// which roots are not watched, and how page transcription goes.
#[derive(Debug, Default)]
pub struct DocumentIndex {
    requested: Notify,
    passes: Mutex<Vec<RootPass>>,
    unwatched: Mutex<BTreeMap<PathBuf, String>>,
    progress: watch::Sender<IndexProgress>,
    transcription: PageTranscription,
}

impl DocumentIndex {
    /// Ask for a round of one pass over every root. One asked for while a round runs
    /// starts when that round ends. Returns the [`IndexProgress::rounds_ended`] at
    /// which a round begun after this call has ended.
    pub fn request_pass(&self) -> u64 {
        let answered_at = {
            let progress = self.progress.borrow();
            let rounds_to_wait = if progress.round_running { 2 } else { 1 };
            progress.rounds_ended + rounds_to_wait
        };
        self.requested.notify_one();
        answered_at
    }

    /// What the last pass over the whole of each root did, in the order they first
    /// ran; empty until the first pass ends.
    pub fn last_passes(&self) -> Vec<RootPass> {
        self.passes.lock().clone()
    }

    /// Why changes under `root` wait for the next pass instead of being indexed as
    /// they happen; `None` unless its watch failed.
    pub fn unwatched(&self, root: &Path) -> Option<String> {
        self.unwatched.lock().get(root).cloned()
    }

    /// Follows the passes as they run.
    pub fn subscribe_progress(&self) -> watch::Receiver<IndexProgress> {
        self.progress.subscribe()
    }

    pub fn transcription(&self) -> &PageTranscription {
        &self.transcription
    }

    /// Record what the pass over `root` has done so far.
    pub(super) fn report_progress(&self, root: &Path, counts: &PassCounts) {
        self.progress.send_modify(|progress| {
            progress.running = Some(PassProgress {
                root: root.to_path_buf(),
                counts: counts.clone(),
            });
        });
    }

    /// Record that `root` has no watch and why, and say so in the log.
    pub(super) fn leave_to_passes(&self, root: &Path, why: String) {
        warn!(
            target: "assistd::documents",
            "{} is not watched, so changes under it wait for the next pass: {why}",
            root.display()
        );
        self.unwatched.lock().insert(root.to_path_buf(), why);
    }

    fn record(&self, pass: RootPass) {
        let mut passes = self.passes.lock();
        match passes.iter_mut().find(|last| last.root == pass.root) {
            Some(last) => *last = pass,
            None => passes.push(pass),
        }
    }

    fn begin_round(&self) {
        self.progress
            .send_modify(|progress| progress.round_running = true);
    }

    fn end_round(&self) {
        self.progress.send_modify(|progress| {
            progress.rounds_ended += 1;
            progress.round_running = false;
            progress.running = None;
        });
    }

    fn end_watched_passes(&self) {
        self.progress
            .send_if_modified(|progress| progress.running.take().is_some());
    }
}

/// What the pass loop woke up to run.
enum Due {
    /// A pass over every root, which was asked for.
    Round,
    /// A pass over each of these, which the roots' watches reported.
    Stale(Vec<(Arc<IndexRoot>, WalkScope)>),
}

/// The passes over the configured index roots.
#[derive(Debug)]
pub struct IndexPasses {
    roots: Vec<Arc<IndexRoot>>,
    sandbox: Arc<SandboxInfo>,
}

impl IndexPasses {
    /// Passes over `documents.roots` that skip what `sandbox`'s path guards call
    /// sensitive. Errors on a `documents.exclude` pattern that is not a valid glob.
    pub(super) fn new(
        documents: &DocumentsConfig,
        sandbox: Arc<SandboxInfo>,
    ) -> Result<Self, DocumentsUnavailable> {
        let roots = documents
            .roots
            .iter()
            .map(|root| IndexRoot::new(root.as_path(), documents).map(Arc::new))
            .collect::<Result<_, _>>()
            .map_err(DocumentsUnavailable::Exclude)?;
        Ok(Self { roots, sandbox })
    }

    /// Watch the roots, then until `shutdown` flips run a pass over every root each
    /// time `index` is asked for one, and over what a watch reports once it has
    /// settled. The database behind `deps.store` must stay open until this returns.
    pub async fn run(
        self,
        deps: IndexDeps,
        index: &DocumentIndex,
        mut shutdown: watch::Receiver<bool>,
    ) {
        let mut watches = tokio::select! {
            biased;
            _ = shutdown.wait_for(|stop| *stop) => return,
            watches = RootWatches::start(&self.roots, index) => watches,
        };
        loop {
            let due = tokio::select! {
                biased;
                _ = shutdown.wait_for(|stop| *stop) => return,
                () = index.requested.notified() => Due::Round,
                () = watches.settled() => Due::Stale(watches.take_stale(index)),
            };
            match due {
                Due::Round => self.index_every_root(&deps, index, &shutdown).await,
                Due::Stale(stale) => self.index_stale(stale, &deps, index, &shutdown).await,
            }
        }
    }

    async fn index_every_root(
        &self,
        deps: &IndexDeps,
        index: &DocumentIndex,
        shutdown: &watch::Receiver<bool>,
    ) {
        index.begin_round();
        let guards = self.path_guards().await;
        for root in &self.roots {
            if *shutdown.borrow() {
                return;
            }
            index_whole_root(deps, index, root, &guards, shutdown).await;
        }
        index.end_round();
    }

    async fn index_stale(
        &self,
        stale: Vec<(Arc<IndexRoot>, WalkScope)>,
        deps: &IndexDeps,
        index: &DocumentIndex,
        shutdown: &watch::Receiver<bool>,
    ) {
        if stale.is_empty() {
            return;
        }
        let guards = self.path_guards().await;
        for (root, scope) in stale {
            if *shutdown.borrow() {
                return;
            }
            match scope {
                WalkScope::WholeRoot => {
                    index_whole_root(deps, index, &root, &guards, shutdown).await;
                }
                WalkScope::Under(_) => {
                    let guards = Arc::clone(&guards);
                    let pass = index_root(deps, index, root, scope, guards, shutdown).await;
                    log_changes(&pass);
                }
            }
        }
        index.end_watched_passes();
    }

    async fn path_guards(&self) -> Arc<PathGuards> {
        let sandbox = Arc::clone(&self.sandbox);
        Arc::new(on_blocking_pool(move || sandbox.path_guards()).await)
    }
}

async fn index_whole_root(
    deps: &IndexDeps,
    index: &DocumentIndex,
    root: &Arc<IndexRoot>,
    guards: &Arc<PathGuards>,
    shutdown: &watch::Receiver<bool>,
) {
    index.report_progress(root.path(), &PassCounts::default());
    let (root, guards) = (Arc::clone(root), Arc::clone(guards));
    let pass = index_root(deps, index, root, WalkScope::WholeRoot, guards, shutdown).await;
    log_pass(&pass);
    index.record(pass);
}

/// Forget every stored index root that `roots` does not list, with its documents.
pub async fn remove_dropped_roots(store: &dyn DocumentStore, roots: &[AbsoluteDir]) {
    let stored = match store.index_roots().await {
        Ok(stored) => stored,
        Err(error) => {
            warn!(target: "assistd::documents", "cannot list the indexed folders: {error}");
            return;
        }
    };
    let is_listed = |stored: &Path| roots.iter().any(|root| root.as_path() == stored);
    for dropped in stored.iter().filter(|stored| !is_listed(stored)) {
        match store.remove_index_root(dropped).await {
            Ok(documents) => info!(
                target: "assistd::documents",
                "{} is no longer in documents.roots; removed its {documents} document(s)",
                dropped.display()
            ),
            Err(error) => warn!(
                target: "assistd::documents",
                "cannot remove the indexed folder {}: {error}",
                dropped.display()
            ),
        }
    }
}

fn log_pass(pass: &RootPass) {
    let (root, counts) = (pass.root.display(), &pass.counts);
    let skipped: u64 = counts.skipped.values().sum();
    match &pass.end {
        PassEnd::Finished => info!(
            target: "assistd::documents",
            "indexed {root}: {} indexed, {} unchanged, {} removed, {skipped} skipped",
            counts.indexed, counts.unchanged, counts.removed
        ),
        PassEnd::RootUnavailable(why) => warn!(
            target: "assistd::documents",
            "skipped {root} and kept its documents: {why}"
        ),
        PassEnd::Incomplete(why) => warn!(
            target: "assistd::documents",
            "the pass over {root} removed no document: {why} ({} indexed, {} unchanged, \
             {skipped} skipped)",
            counts.indexed, counts.unchanged
        ),
    }
    warn_if_index_full(pass);
}

fn log_changes(pass: &RootPass) {
    let (root, counts) = (pass.root.display(), &pass.counts);
    let skipped: u64 = counts.skipped.values().sum();
    match &pass.end {
        PassEnd::Finished => debug!(
            target: "assistd::documents",
            "indexed changes under {root}: {} indexed, {} unchanged, {} removed, {skipped} skipped",
            counts.indexed, counts.unchanged, counts.removed
        ),
        PassEnd::RootUnavailable(why) | PassEnd::Incomplete(why) => warn!(
            target: "assistd::documents",
            "changes under {root} wait for the next pass: {why}"
        ),
    }
    warn_if_index_full(pass);
}

fn warn_if_index_full(pass: &RootPass) {
    if pass.counts.skipped.contains_key(&SkipCause::IndexFull) {
        warn!(
            target: "assistd::documents",
            "the document index is full; {} has files that were not stored",
            pass.root.display()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pass_asked_for_mid_round_is_answered_by_the_round_after() {
        let index = DocumentIndex::default();
        assert_eq!(index.request_pass(), 1, "asked while idle");
        index.begin_round();
        assert_eq!(index.request_pass(), 2, "asked while the first round runs");
        index.end_round();
        assert_eq!(index.request_pass(), 2, "asked once that round has ended");
    }
}
