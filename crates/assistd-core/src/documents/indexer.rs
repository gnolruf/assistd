//! When index passes run: after each request, one root at a time, until shutdown.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use assistd_config::{AbsoluteDir, DocumentsConfig};
use assistd_memory::SemanticStore;
use assistd_tools::SandboxInfo;
use assistd_tools::policy::on_blocking_pool;
use parking_lot::Mutex;
use tokio::sync::{Notify, watch};
use tracing::{info, warn};

use super::DocumentsUnavailable;
use super::pass::{IndexDeps, PassCounts, PassEnd, RootPass, SkipCause, index_root};
use super::walk::IndexRoot;

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
    /// The pass under way; `None` between rounds.
    pub running: Option<PassProgress>,
}

impl IndexProgress {
    /// The root a pass is walking.
    pub fn running_root(&self) -> Option<&Path> {
        self.running.as_ref().map(|pass| pass.root.as_path())
    }
}

/// The document index as the rest of the daemon sees it: where a pass is asked
/// for, how far the passes have got, and what the last pass over each root did.
#[derive(Debug, Default)]
pub struct DocumentIndex {
    requested: Notify,
    passes: Mutex<Vec<RootPass>>,
    progress: watch::Sender<IndexProgress>,
}

impl DocumentIndex {
    /// Ask for a round of one pass over every root. One asked for while a round runs
    /// starts when that round ends. Returns the [`IndexProgress::rounds_ended`] at
    /// which a round begun after this call has ended.
    pub fn request_pass(&self) -> u64 {
        let answered_at = {
            let progress = self.progress.borrow();
            let rounds_to_wait = if progress.running.is_some() { 2 } else { 1 };
            progress.rounds_ended + rounds_to_wait
        };
        self.requested.notify_one();
        answered_at
    }

    /// What the last pass over each root did, in the order they ran; empty until
    /// the first pass ends.
    pub fn last_passes(&self) -> Vec<RootPass> {
        self.passes.lock().clone()
    }

    /// Follows the passes as they run.
    pub fn subscribe_progress(&self) -> watch::Receiver<IndexProgress> {
        self.progress.subscribe()
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

    fn record(&self, pass: RootPass) {
        let mut passes = self.passes.lock();
        match passes.iter_mut().find(|last| last.root == pass.root) {
            Some(last) => *last = pass,
            None => passes.push(pass),
        }
    }

    fn end_round(&self) {
        self.progress.send_modify(|progress| {
            progress.rounds_ended += 1;
            progress.running = None;
        });
    }
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

    /// Run a pass over every root each time `index` is asked for one, until
    /// `shutdown` flips. The database behind `deps.store` must stay open until
    /// this returns.
    pub async fn run(
        self,
        deps: IndexDeps,
        index: &DocumentIndex,
        mut shutdown: watch::Receiver<bool>,
    ) {
        loop {
            tokio::select! {
                biased;
                _ = shutdown.wait_for(|stop| *stop) => return,
                () = index.requested.notified() => {}
            }
            self.index_every_root(&deps, index, &shutdown).await;
        }
    }

    async fn index_every_root(
        &self,
        deps: &IndexDeps,
        index: &DocumentIndex,
        shutdown: &watch::Receiver<bool>,
    ) {
        let sandbox = Arc::clone(&self.sandbox);
        let guards = Arc::new(on_blocking_pool(move || sandbox.path_guards()).await);
        for root in &self.roots {
            if *shutdown.borrow() {
                return;
            }
            index.report_progress(root.path(), &PassCounts::default());
            let pass =
                index_root(deps, index, Arc::clone(root), Arc::clone(&guards), shutdown).await;
            log_pass(&pass);
            index.record(pass);
        }
        index.end_round();
    }
}

/// Forget every stored index root that `roots` does not list, with its documents.
pub async fn remove_dropped_roots(store: &dyn SemanticStore, roots: &[AbsoluteDir]) {
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
    if counts.skipped.contains_key(&SkipCause::IndexFull) {
        warn!(
            target: "assistd::documents",
            "the document index is full; {root} has files that were not stored"
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
        index.report_progress(Path::new("/notes"), &PassCounts::default());
        assert_eq!(index.request_pass(), 2, "asked while the first round runs");
        index.end_round();
        assert_eq!(index.request_pass(), 2, "asked once that round has ended");
    }
}
