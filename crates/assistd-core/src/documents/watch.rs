//! Watching the index roots with inotify between passes: which paths changed under
//! each root, and whether its watch lost changes or stopped.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use notify::{EventKind, RecursiveMode};
use tokio::sync::Notify;

use super::indexer::DocumentIndex;
use super::walk::{IndexRoot, WalkScope};
use crate::dir_watch::{DirWatch, Report};

/// How long changes are gathered after the first one before they are indexed, so
/// a file saved in several writes is read once.
const SETTLE_WINDOW: Duration = Duration::from_secs(2);

/// Most changed paths held for one root; a root with more has a pass run over it.
const MAX_HELD_PATHS: usize = 65_536;

/// The paths a root's watch saw change since it was last asked.
#[derive(Debug, Default)]
struct Changed {
    paths: BTreeSet<PathBuf>,
    /// Changes went unreported: the kernel's event queue overflowed, or more paths
    /// changed than are held.
    lost_changes: bool,
}

impl Changed {
    /// Take in one event under `root`, and say whether it left anything to index.
    /// Reads change nothing and no walk reaches a hidden or excluded path, so those
    /// are dropped.
    fn record(&mut self, event: notify::Event, root: &IndexRoot) -> bool {
        if event.need_rescan() {
            self.lost_changes = true;
            return true;
        }
        if matches!(event.kind, EventKind::Access(_)) {
            return false;
        }
        let mut walked = event.paths;
        walked.retain(|path| !root.leaves_out(path));
        let changed = !walked.is_empty();
        walked.into_iter().for_each(|path| self.hold(path));
        changed
    }

    fn hold(&mut self, path: PathBuf) {
        if self.paths.len() < MAX_HELD_PATHS {
            self.paths.insert(path);
        } else {
            self.lost_changes = true;
        }
    }
}

#[derive(Debug)]
struct RootWatch {
    root: Arc<IndexRoot>,
    watch: DirWatch<Changed>,
}

/// The inotify watches on the index roots that could be watched.
#[derive(Debug)]
pub(super) struct RootWatches {
    watches: Vec<RootWatch>,
    wake: Arc<Notify>,
}

impl RootWatches {
    /// Watch each of `roots` and everything under it. One that cannot be watched is
    /// left to passes, which `index` records with why.
    pub(super) async fn start(roots: &[Arc<IndexRoot>], index: &DocumentIndex) -> Self {
        let wake = Arc::new(Notify::new());
        let mut watches = Vec::with_capacity(roots.len());
        for root in roots {
            let events_under = Arc::clone(root);
            let watch = DirWatch::start(
                root.path().to_path_buf(),
                RecursiveMode::Recursive,
                Arc::clone(&wake),
                move |changed: &mut Changed, event| changed.record(event, &events_under),
            );
            match watch.await {
                Ok(watch) => watches.push(RootWatch {
                    root: Arc::clone(root),
                    watch,
                }),
                Err(why) => index.leave_to_passes(root.path(), why),
            }
        }
        Self { watches, wake }
    }

    /// Resolves [`SETTLE_WINDOW`] after a watch reports, or after the call when one
    /// already has.
    pub(super) async fn settled(&self) {
        while !self.watches.iter().any(|root| root.watch.holds_report()) {
            self.wake.notified().await;
        }
        tokio::time::sleep(SETTLE_WINDOW).await;
    }

    /// What each watch reported since it was last asked, as the walk that brings the
    /// index in step: the whole root once changes were lost or the watch stopped, else
    /// the changed paths. A stopped watch is dropped, which `index` records with why.
    pub(super) fn take_stale(&mut self, index: &DocumentIndex) -> Vec<(Arc<IndexRoot>, WalkScope)> {
        let mut stale = Vec::new();
        self.watches.retain(|RootWatch { root, watch }| {
            let Report {
                changes:
                    Changed {
                        paths,
                        lost_changes,
                    },
                lost,
                ..
            } = watch.take();
            let still_watched = lost.is_none();
            if lost_changes || !still_watched {
                stale.push((Arc::clone(root), WalkScope::WholeRoot));
            } else if !paths.is_empty() {
                stale.push((Arc::clone(root), WalkScope::Under(Arc::new(paths))));
            }
            if let Some(why) = lost {
                index.leave_to_passes(root.path(), why);
            }
            still_watched
        });
        stale
    }
}
