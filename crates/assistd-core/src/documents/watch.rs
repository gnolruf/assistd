//! Watching the index roots with inotify between passes: which paths changed under
//! each root, and whether its watch lost changes or stopped.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use assistd_tools::policy::on_blocking_pool;
use notify::{ErrorKind, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use parking_lot::Mutex;
use tokio::sync::Notify;

use super::indexer::DocumentIndex;
use super::walk::{IndexRoot, WalkScope};
use crate::tasks::path_watch::removes;

/// How long changes are gathered after the first one before they are indexed, so
/// a file saved in several writes is read once.
const SETTLE_WINDOW: Duration = Duration::from_secs(2);

/// Most changed paths held for one root; a root with more has a pass run over it.
const MAX_HELD_PATHS: usize = 65_536;

/// What a root's watch has reported since it was last asked.
#[derive(Debug, Default)]
struct Reported {
    changed: BTreeSet<PathBuf>,
    /// Changes went unreported: the kernel's event queue overflowed, or more paths
    /// changed than are held.
    lost_changes: bool,
    /// Why the watch stopped reporting, once it has.
    stopped: Option<String>,
}

impl Reported {
    fn is_empty(&self) -> bool {
        self.changed.is_empty() && !self.lost_changes && self.stopped.is_none()
    }

    fn hold(&mut self, path: PathBuf) {
        if self.changed.len() < MAX_HELD_PATHS {
            self.changed.insert(path);
        } else {
            self.lost_changes = true;
        }
    }
}

/// One watched root, shared with the thread its events arrive on.
#[derive(Debug)]
struct WatchedRoot {
    root: Arc<IndexRoot>,
    reported: Mutex<Reported>,
    /// Wakes whoever waits for any root to report.
    wake: Arc<Notify>,
}

impl WatchedRoot {
    /// Take in one event of the watch. Reads change nothing, so they are dropped.
    fn record(&self, event: notify::Result<notify::Event>) {
        let mut reported = self.reported.lock();
        match event {
            Err(error) => reported.stopped = Some(watch_failure(&error)),
            Ok(event) if event.need_rescan() => reported.lost_changes = true,
            Ok(event) if matches!(event.kind, EventKind::Access(_)) => return,
            Ok(event) if removes(&event, self.root.path()) => {
                reported.stopped = Some("the folder was removed or moved".to_owned());
            }
            Ok(event) => event.paths.into_iter().for_each(|path| reported.hold(path)),
        }
        self.wake.notify_one();
    }
}

#[derive(Debug)]
struct RootWatch {
    watched: Arc<WatchedRoot>,
    _watcher: RecommendedWatcher,
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
            let watched = Arc::new(WatchedRoot {
                root: Arc::clone(root),
                reported: Mutex::default(),
                wake: Arc::clone(&wake),
            });
            let events_to = Arc::clone(&watched);
            match on_blocking_pool(move || watch_recursively(events_to)).await {
                Ok(watcher) => watches.push(RootWatch {
                    watched,
                    _watcher: watcher,
                }),
                Err(error) => index.leave_to_passes(root.path(), watch_failure(&error)),
            }
        }
        Self { watches, wake }
    }

    /// Resolves [`SETTLE_WINDOW`] after a watch reports, or after the call when one
    /// already has.
    pub(super) async fn settled(&self) {
        let holds_report = |watch: &RootWatch| !watch.watched.reported.lock().is_empty();
        if !self.watches.iter().any(holds_report) {
            self.wake.notified().await;
        }
        tokio::time::sleep(SETTLE_WINDOW).await;
    }

    /// What each watch reported since it was last asked, as the walk that brings the
    /// index in step: the whole root once changes were lost or the watch stopped, else
    /// the changed paths. A stopped watch is dropped, which `index` records with why.
    pub(super) fn take_stale(&mut self, index: &DocumentIndex) -> Vec<(Arc<IndexRoot>, WalkScope)> {
        let mut stale = Vec::new();
        self.watches.retain(|watch| {
            let root = &watch.watched.root;
            let Reported {
                changed,
                lost_changes,
                stopped,
            } = std::mem::take(&mut *watch.watched.reported.lock());
            let still_watched = stopped.is_none();
            if lost_changes || !still_watched {
                stale.push((Arc::clone(root), WalkScope::WholeRoot));
            } else if !changed.is_empty() {
                stale.push((Arc::clone(root), WalkScope::Under(Arc::new(changed))));
            }
            if let Some(why) = stopped {
                index.leave_to_passes(root.path(), why);
            }
            still_watched
        });
        stale
    }
}

/// An inotify watch on `watched`'s root and every directory under it, reporting to
/// `watched`. Blocks on the filesystem while it adds one watch per directory.
fn watch_recursively(watched: Arc<WatchedRoot>) -> notify::Result<RecommendedWatcher> {
    let root = watched.root.path().to_path_buf();
    let mut watcher = notify::recommended_watcher(move |event| watched.record(event))?;
    watcher.watch(&root, RecursiveMode::Recursive)?;
    Ok(watcher)
}

/// Why a watch could not be set up or stopped, naming the limit when that is it.
fn watch_failure(error: &notify::Error) -> String {
    match error.kind {
        ErrorKind::MaxFilesWatch => {
            "no inotify watches are left (fs.inotify.max_user_watches)".to_owned()
        }
        _ => error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use assistd_config::DocumentsConfig;
    use notify::event::Flag;

    use super::*;

    #[tokio::test]
    async fn an_overflow_of_the_event_queue_has_a_pass_run_over_the_root_it_came_from() {
        let temp = tempfile::tempdir().expect("tempdir");
        let roots: Vec<Arc<IndexRoot>> = ["notes", "papers"]
            .into_iter()
            .map(|name| {
                let path = temp.path().join(name);
                std::fs::create_dir(&path).expect("root");
                Arc::new(IndexRoot::new(&path, &DocumentsConfig::default()).unwrap())
            })
            .collect();
        let index = DocumentIndex::default();
        let mut watches = RootWatches::start(&roots, &index).await;
        let [_, papers] = &watches.watches[..] else {
            panic!("expected both roots watched, got {watches:?}");
        };

        let overflow = notify::Event::new(EventKind::Other).set_flag(Flag::Rescan);
        papers.watched.record(Ok(overflow));
        let stale = watches.take_stale(&index);

        let [(root, WalkScope::WholeRoot)] = &stale[..] else {
            panic!("expected one pass over a whole root, got {stale:?}");
        };
        assert_eq!(root.path(), roots[1].path());
        assert!(watches.take_stale(&index).is_empty());
    }
}
