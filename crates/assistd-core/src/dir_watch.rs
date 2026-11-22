//! One inotify watch on a directory that follows no symlink below it, holding what
//! it saw until it is asked and why it was lost.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use assistd_tools::policy::on_blocking_pool;
use notify::event::{ModifyKind, RenameMode};
use notify::{Config, ErrorKind, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use parking_lot::Mutex;
use tokio::sync::Notify;

/// What a watch saw since it was last asked.
#[derive(Debug, Default)]
pub(crate) struct Report<C> {
    pub(crate) changes: C,
    /// Why the watch stopped reporting, once it has: it erred, or its directory was
    /// removed or moved.
    pub(crate) lost: Option<String>,
    unread: bool,
}

/// The side of a watch its events arrive on.
#[derive(Debug)]
struct Seen<C> {
    dir: PathBuf,
    report: Mutex<Report<C>>,
    wake: Arc<Notify>,
}

impl<C> Seen<C> {
    /// Take in one event: the loss of the watch, or whatever `record` keeps of it.
    fn take_in(
        &self,
        event: notify::Result<notify::Event>,
        record: &mut impl FnMut(&mut C, notify::Event) -> bool,
    ) {
        let mut report = self.report.lock();
        let reported = match event {
            Err(error) => {
                report.lost = Some(watch_failure(&error));
                true
            }
            Ok(event) if removes(&event, &self.dir) => {
                report.lost = Some("the watched folder was removed or moved".to_owned());
                true
            }
            Ok(event) => record(&mut report.changes, event),
        };
        if reported {
            report.unread = true;
            drop(report);
            self.wake.notify_one();
        }
    }
}

/// A watch on one directory, gathering its events into a `C`.
#[derive(Debug)]
pub(crate) struct DirWatch<C> {
    seen: Arc<Seen<C>>,
    _watcher: RecommendedWatcher,
}

impl<C: Default + Send + 'static> DirWatch<C> {
    /// Watch `dir`, which may be a symlink, and with `RecursiveMode::Recursive` every
    /// directory under it that no symlink leads to. `record` adds an event to the
    /// changes held and says whether it did; each report notifies `wake`. Errs with
    /// why the watch could not be set up.
    pub(crate) async fn start<R>(
        dir: PathBuf,
        mode: RecursiveMode,
        wake: Arc<Notify>,
        record: R,
    ) -> Result<Self, String>
    where
        R: FnMut(&mut C, notify::Event) -> bool + Send + 'static,
    {
        let seen = Arc::new(Seen {
            dir,
            report: Mutex::default(),
            wake,
        });
        let events_to = Arc::clone(&seen);
        let watcher = on_blocking_pool(move || watch(events_to, mode, record))
            .await
            .map_err(|error| watch_failure(&error))?;
        Ok(Self {
            seen,
            _watcher: watcher,
        })
    }

    /// Whether anything was reported since the watch was last asked.
    pub(crate) fn holds_report(&self) -> bool {
        self.seen.report.lock().unread
    }

    /// Resolves `window` after the first report not yet taken, so that a burst of
    /// changes is taken as one, or as soon as the watch is lost. Only for a watch
    /// that shares its `wake` with no other.
    pub(crate) async fn settled(&self, window: Duration) {
        self.woken_until(|report| report.unread).await;
        let lost = self.woken_until(|report| report.lost.is_some());
        let _ = tokio::time::timeout(window, lost).await;
    }

    /// Take what the watch saw since it was last asked.
    pub(crate) fn take(&self) -> Report<C> {
        std::mem::take(&mut *self.seen.report.lock())
    }

    async fn woken_until(&self, ready: impl Fn(&Report<C>) -> bool) {
        while !self.report_is(&ready) {
            self.seen.wake.notified().await;
        }
    }

    fn report_is(&self, ready: impl Fn(&Report<C>) -> bool) -> bool {
        ready(&self.seen.report.lock())
    }
}

/// Set up the inotify watch that reports to `seen`. Blocks on the filesystem while a
/// recursive watch adds one inotify watch per directory.
fn watch<C, R>(
    seen: Arc<Seen<C>>,
    mode: RecursiveMode,
    mut record: R,
) -> notify::Result<RecommendedWatcher>
where
    C: Send + 'static,
    R: FnMut(&mut C, notify::Event) -> bool + Send + 'static,
{
    let entered = entered_through_links(&seen.dir);
    let mut watcher = RecommendedWatcher::new(
        move |event| seen.take_in(event, &mut record),
        Config::default().with_follow_symlinks(false),
    )?;
    watcher.watch(&entered, mode)?;
    Ok(watcher)
}

/// `dir` spelled with a trailing separator, which has a watch that follows no symlink
/// still enter a directory that is one.
fn entered_through_links(dir: &Path) -> PathBuf {
    dir.join("")
}

/// Whether `event` removes or moves away `dir` itself, which ends a watch on it.
fn removes(event: &notify::Event, dir: &Path) -> bool {
    matches!(
        event.kind,
        EventKind::Remove(_) | EventKind::Modify(ModifyKind::Name(RenameMode::From))
    ) && event.paths.iter().any(|path| path == dir)
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
    use super::*;

    const REPORTED_WITHIN: Duration = Duration::from_secs(5);

    async fn watch_recording<R>(dir: PathBuf, record: R) -> DirWatch<bool>
    where
        R: FnMut(&mut bool, notify::Event) -> bool + Send + 'static,
    {
        DirWatch::start(dir, RecursiveMode::Recursive, Arc::default(), record)
            .await
            .unwrap()
    }

    async fn watch_writes(dir: PathBuf) -> DirWatch<bool> {
        watch_recording(dir, |changed, event| {
            let wrote = !matches!(event.kind, EventKind::Access(_));
            *changed |= wrote;
            wrote
        })
        .await
    }

    #[tokio::test]
    async fn a_burst_of_changes_inside_the_window_is_one_report() {
        const WINDOW: Duration = Duration::from_millis(600);
        let dir = tempfile::tempdir().unwrap();
        let watch = watch_writes(dir.path().to_path_buf()).await;

        for contents in ["one", "two", "three"] {
            std::fs::write(dir.path().join("notes.txt"), contents).unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        tokio::time::timeout(REPORTED_WITHIN, watch.settled(WINDOW))
            .await
            .expect("the burst was not reported");
        assert!(watch.take().changes);

        let another = tokio::time::timeout(WINDOW * 2, watch.settled(WINDOW)).await;
        assert!(another.is_err(), "the burst was reported more than once");
    }

    #[tokio::test]
    async fn removing_the_watched_directory_reports_the_watch_lost_without_the_window() {
        let parent = tempfile::tempdir().unwrap();
        let watched = parent.path().join("watched");
        std::fs::create_dir(&watched).unwrap();
        let watch = watch_recording(watched.clone(), |_, _| false).await;

        std::fs::remove_dir(&watched).unwrap();

        tokio::time::timeout(REPORTED_WITHIN, watch.settled(Duration::from_secs(3600)))
            .await
            .expect("the removal was not reported");
        assert!(watch.take().lost.is_some());
    }

    #[tokio::test]
    async fn a_watched_tree_is_entered_through_a_link_to_it_but_not_through_links_inside_it() {
        let parent = tempfile::tempdir().unwrap();
        let (tree, outside) = (parent.path().join("tree"), parent.path().join("outside"));
        std::fs::create_dir_all(tree.join("nested")).unwrap();
        std::fs::create_dir(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, tree.join("nested/linked")).unwrap();
        let link_to_tree = parent.path().join("link-to-tree");
        std::os::unix::fs::symlink(&tree, &link_to_tree).unwrap();
        let watch = watch_writes(link_to_tree).await;

        std::fs::write(outside.join("unwatched.txt"), "text").unwrap();
        let behind_the_link =
            tokio::time::timeout(Duration::from_millis(300), watch.settled(Duration::ZERO));
        assert!(behind_the_link.await.is_err(), "the link was followed");

        for watched in ["top.txt", "nested/deep.txt"] {
            std::fs::write(tree.join(watched), "text").unwrap();
            tokio::time::timeout(REPORTED_WITHIN, watch.settled(Duration::ZERO))
                .await
                .unwrap_or_else(|_| panic!("the change to {watched} was not reported"));
            watch.take();
        }
    }
}
