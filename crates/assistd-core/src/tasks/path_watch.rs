//! Watching a `PathChanged` trigger's path with inotify and gathering each
//! burst of changes under it into one fire.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use notify::event::{ModifyKind, RenameMode};
use notify::{EventKind, RecursiveMode};
use tracing::{error, warn};

use assistd_ipc::{TaskEvent, TaskId, TaskTrigger};
use assistd_tools::policy::on_blocking_pool;
use assistd_utils::backoff::{RestartDecision, RestartPolicy};

use super::TaskService;
use super::event_triggers::event_trigger;
use crate::dir_watch::{DirWatch, Report};

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
        let watch = match watch_trigger_path(path.to_path_buf()).await {
            Ok(watch) => watch,
            Err(reason) => return WatchEnd::Failed(reason),
        };
        self.events.clear_watch_failure(task, trigger);
        loop {
            watch.settled(debounce).await;
            let Report { changes, lost, .. } = watch.take();
            if changes {
                self.fire_watched_trigger(task, trigger).await;
            }
            if let Some(reason) = lost {
                return WatchEnd::Lost(reason);
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

/// An inotify watch on `path`, a directory with everything under it or a single file,
/// holding whether what the trigger names changed.
async fn watch_trigger_path(path: PathBuf) -> Result<DirWatch<bool>, String> {
    let watched = on_blocking_pool(move || Watched::of(&path)).await;
    DirWatch::start(
        watched.root().to_path_buf(),
        watched.mode(),
        Arc::default(),
        move |changed: &mut bool, event| {
            let concerned = is_change(event.kind) && watched.concerns(&event);
            *changed |= concerned;
            concerned
        },
    )
    .await
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_save_that_replaces_the_watched_file_is_a_change_and_keeps_the_watch() {
        let parent = tempfile::tempdir().unwrap();
        let watched = parent.path().join("notes.txt");
        std::fs::write(&watched, "old").unwrap();
        let watch = watch_trigger_path(watched.clone()).await.unwrap();

        for contents in ["new", "newer"] {
            let saved = parent.path().join("notes.txt.tmp");
            std::fs::write(&saved, contents).unwrap();
            watch.take();
            std::fs::rename(&saved, &watched).unwrap();

            tokio::time::timeout(Duration::from_secs(5), watch.settled(Duration::ZERO))
                .await
                .expect("the save was not reported");
            let Report { changes, lost, .. } = watch.take();
            assert!(changes);
            assert_eq!(lost, None);
        }
    }
}
