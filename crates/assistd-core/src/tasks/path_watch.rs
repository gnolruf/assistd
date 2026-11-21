//! Watching a `PathChanged` trigger's path with inotify and gathering each
//! burst of changes under it into one fire.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use notify::event::{ModifyKind, RenameMode};
use notify::{Config, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::{mpsc, watch};
use tracing::{error, warn};

use assistd_ipc::{TaskEvent, TaskId, TaskTrigger};
use assistd_utils::backoff::{RestartDecision, RestartPolicy};

use super::TaskService;
use super::event_triggers::event_trigger;

/// Changes held between bursts; more than this inside one window add nothing.
const CHANGE_BUFFER: usize = 16;

/// Why a watch on a trigger's path stopped.
enum WatchEnd {
    /// The watch could not be set up.
    Failed(String),
    /// The watch was set up, then lost.
    Lost(String),
}

/// What is watched for a trigger's path, and which changes there count.
enum Watched {
    /// A directory and everything under it.
    Tree(PathBuf),
    /// One file, seen through a watch on `parent` so that a save which replaces the
    /// file is still seen; `file` is the trigger's path with symlinks resolved.
    File { parent: PathBuf, file: PathBuf },
}

impl Watched {
    /// A path that is not a directory is watched as a file, whether or not it exists.
    fn of(path: &Path) -> Self {
        if path.is_dir() {
            return Self::Tree(path.to_path_buf());
        }
        let file = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        match file.parent() {
            Some(parent) => Self::File {
                parent: parent.to_path_buf(),
                file,
            },
            None => Self::Tree(file),
        }
    }

    /// The directory the inotify watch sits on, whose removal ends the watch.
    fn root(&self) -> &Path {
        match self {
            Self::Tree(root) | Self::File { parent: root, .. } => root,
        }
    }

    fn mode(&self) -> RecursiveMode {
        match self {
            Self::Tree(_) => RecursiveMode::Recursive,
            Self::File { .. } => RecursiveMode::NonRecursive,
        }
    }

    /// Whether `event` is about what the trigger names; a rename counts for where it
    /// lands, which is its last path.
    fn concerns(&self, event: &notify::Event) -> bool {
        match self {
            Self::Tree(_) => true,
            Self::File { file, .. } => event.paths.last() == Some(file),
        }
    }
}

impl TaskService {
    /// Fire `PathChanged` trigger `trigger` of task `task` once per burst of changes
    /// under its path, retrying with backoff whenever the watch fails or is lost, for
    /// as long as the future runs.
    pub(super) async fn watch_path(self: Arc<Self>, task: TaskId, trigger: TaskTrigger) {
        let Some((
            TaskEvent::PathChanged {
                path,
                debounce_secs,
            },
            _,
        )) = event_trigger(&trigger)
        else {
            return;
        };
        let debounce = Duration::from_secs(u64::from(*debounce_secs));
        let mut policy = RestartPolicy::default();
        loop {
            let started = Instant::now();
            let reason = match self.watch_until_lost(task, &trigger, path, debounce).await {
                WatchEnd::Failed(reason) => {
                    policy.record_spawn_failure();
                    reason
                }
                WatchEnd::Lost(reason) => {
                    policy.record_session_end(started.elapsed());
                    reason
                }
            };
            let delay = retry_delay(task, &mut policy);
            warn!(
                target: "assistd::tasks",
                %task,
                path = %path.display(),
                error = %reason,
                retry_secs = delay.as_secs(),
                "watching the trigger's path stopped"
            );
            self.events.record_watch_failure(
                task,
                &trigger,
                format!("watching {} stopped: {reason}", path.display()),
            );
            tokio::time::sleep(delay).await;
        }
    }

    /// Fire `trigger` of `task` once per burst of changes under `path` until the watch
    /// on it fails or is lost.
    async fn watch_until_lost(
        &self,
        task: TaskId,
        trigger: &TaskTrigger,
        path: &Path,
        debounce: Duration,
    ) -> WatchEnd {
        let (changes_tx, mut changes) = mpsc::channel(CHANGE_BUFFER);
        let (lost_tx, mut lost) = watch::channel(None);
        let _watcher =
            match watch_changes_off_runtime(path.to_path_buf(), changes_tx, lost_tx).await {
                Ok(watcher) => watcher,
                Err(e) => return WatchEnd::Failed(e.to_string()),
            };
        self.events.clear_watch_failure(task, trigger);
        loop {
            tokio::select! {
                burst = next_burst(&mut changes, debounce) => {
                    if burst.is_none() {
                        return WatchEnd::Lost("the watch closed".into());
                    }
                    self.fire_watched_trigger(task, trigger).await;
                }
                reason = lost_reason(&mut lost) => return WatchEnd::Lost(reason),
            }
        }
    }
}

/// Register the next retry of `task`'s watch and pick how long to wait before it.
fn retry_delay(task: TaskId, policy: &mut RestartPolicy) -> Duration {
    let decision = policy.next_restart(Instant::now());
    if let RestartDecision::SlowRetry { delay, cap } = decision {
        error!(
            target: "assistd::tasks",
            %task,
            retry_secs = delay.as_secs(),
            "the watch on the trigger's path {cap}; slowing retries"
        );
    }
    decision.delay()
}

/// [`watch_changes`] run on the blocking pool, since a recursive watch walks the
/// whole tree adding one inotify watch per directory.
async fn watch_changes_off_runtime(
    path: PathBuf,
    changes: mpsc::Sender<()>,
    lost: watch::Sender<Option<String>>,
) -> notify::Result<RecommendedWatcher> {
    tokio::task::spawn_blocking(move || watch_changes(&path, changes, lost))
        .await
        .map_err(|e| notify::Error::generic(&e.to_string()))?
}

/// An inotify watch on `path`, a directory with everything under it or a single file,
/// sending to `changes` on each change and why to `lost` once the watched directory is
/// removed or moved, or the watch errs.
fn watch_changes(
    path: &Path,
    changes: mpsc::Sender<()>,
    lost: watch::Sender<Option<String>>,
) -> notify::Result<RecommendedWatcher> {
    let watched = Watched::of(path);
    let root = entered_through_links(watched.root());
    let mode = watched.mode();
    let mut watcher = RecommendedWatcher::new(
        move |event: notify::Result<notify::Event>| match event {
            Ok(event) if removes(&event, watched.root()) => {
                lost.send_replace(Some("the path was removed or moved".into()));
            }
            Ok(event) if is_change(event.kind) && watched.concerns(&event) => {
                let _ = changes.try_send(());
            }
            Ok(_) => {}
            Err(e) => {
                lost.send_replace(Some(e.to_string()));
            }
        },
        Config::default().with_follow_symlinks(false),
    )?;
    watcher.watch(&root, mode)?;
    Ok(watcher)
}

/// `root` spelled with a trailing separator, which has a watch that follows no symlink
/// still enter a root that is one.
pub(crate) fn entered_through_links(root: &Path) -> PathBuf {
    root.join("")
}

/// Whether `event` removes or moves away `root` itself, which ends a watch on it.
pub(crate) fn removes(event: &notify::Event, root: &Path) -> bool {
    matches!(
        event.kind,
        EventKind::Remove(_) | EventKind::Modify(ModifyKind::Name(RenameMode::From))
    ) && event.paths.iter().any(|path| path == root)
}

/// Whether `kind` is a create, a content change, or a move in.
fn is_change(kind: EventKind) -> bool {
    matches!(
        kind,
        EventKind::Create(_)
            | EventKind::Modify(
                ModifyKind::Any
                    | ModifyKind::Data(_)
                    | ModifyKind::Other
                    | ModifyKind::Name(RenameMode::To | RenameMode::Both | RenameMode::Any)
            )
    )
}

/// Wait until `lost` holds why the watch was lost, and take it.
async fn lost_reason(lost: &mut watch::Receiver<Option<String>>) -> String {
    let reason = lost
        .wait_for(Option::is_some)
        .await
        .ok()
        .and_then(|reason| reason.clone());
    reason.unwrap_or_else(|| "the watch closed".into())
}

/// Wait for a change, then absorb every other change within `window` of it; `None`
/// once `changes` closes.
async fn next_burst(changes: &mut mpsc::Receiver<()>, window: Duration) -> Option<()> {
    changes.recv().await?;
    tokio::time::sleep(window).await;
    while changes.try_recv().is_ok() {}
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOW: Duration = Duration::from_secs(2);

    #[tokio::test(start_paused = true)]
    async fn a_burst_of_changes_inside_the_debounce_window_fires_once() {
        let (tx, mut changes) = mpsc::channel(CHANGE_BUFFER);
        let _open = tx.clone();
        let burst = tokio::spawn(async move {
            for _ in 0..4 {
                tx.send(()).await.unwrap();
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        });

        assert!(next_burst(&mut changes, WINDOW).await.is_some());
        burst.await.unwrap();
        let another = tokio::time::timeout(WINDOW * 5, next_burst(&mut changes, WINDOW)).await;
        assert!(another.is_err(), "the burst fired more than once");
    }

    #[tokio::test]
    async fn removing_the_watched_directory_reports_the_watch_lost() {
        let parent = tempfile::tempdir().unwrap();
        let watched = parent.path().join("watched");
        std::fs::create_dir(&watched).unwrap();
        let (changes_tx, _changes) = mpsc::channel(CHANGE_BUFFER);
        let (lost_tx, mut lost) = watch::channel(None);
        let _watcher = watch_changes_off_runtime(watched.clone(), changes_tx, lost_tx)
            .await
            .unwrap();

        std::fs::remove_dir(&watched).unwrap();

        tokio::time::timeout(Duration::from_secs(5), lost.wait_for(Option::is_some))
            .await
            .expect("the removal was not reported")
            .unwrap();
    }

    #[tokio::test]
    async fn a_save_that_replaces_the_watched_file_is_a_change_and_keeps_the_watch() {
        let parent = tempfile::tempdir().unwrap();
        let watched = parent.path().join("notes.txt");
        std::fs::write(&watched, "old").unwrap();
        let (changes_tx, mut changes) = mpsc::channel(CHANGE_BUFFER);
        let (lost_tx, lost) = watch::channel(None);
        let _watcher = watch_changes_off_runtime(watched.clone(), changes_tx, lost_tx)
            .await
            .unwrap();

        for contents in ["new", "newer"] {
            let saved = parent.path().join("notes.txt.tmp");
            std::fs::write(&saved, contents).unwrap();
            while changes.try_recv().is_ok() {}
            std::fs::rename(&saved, &watched).unwrap();

            tokio::time::timeout(Duration::from_secs(5), changes.recv())
                .await
                .expect("the save was not reported")
                .unwrap();
        }
        assert_eq!(*lost.borrow(), None);
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
        let (changes_tx, mut changes) = mpsc::channel(CHANGE_BUFFER);
        let (lost_tx, _lost) = watch::channel(None);
        let _watcher = watch_changes_off_runtime(link_to_tree, changes_tx, lost_tx)
            .await
            .unwrap();

        std::fs::write(outside.join("unwatched.txt"), "text").unwrap();
        let behind_the_link = tokio::time::timeout(Duration::from_millis(300), changes.recv());
        assert!(behind_the_link.await.is_err(), "the link was followed");

        for watched in ["top.txt", "nested/deep.txt"] {
            std::fs::write(tree.join(watched), "text").unwrap();
            tokio::time::timeout(Duration::from_secs(5), changes.recv())
                .await
                .unwrap_or_else(|_| panic!("the change to {watched} was not reported"))
                .unwrap();
            while changes.try_recv().is_ok() {}
        }
    }
}
