//! Watching a `PathChanged` trigger's path with inotify and gathering each
//! burst of changes under it into one fire.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use notify::event::{ModifyKind, RenameMode};
use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::{mpsc, watch};
use tracing::{error, warn};

use assistd_ipc::{TaskEvent, TaskId, TaskTrigger};
use assistd_utils::backoff::{RestartDecision, RestartPolicy};

use super::AppState;
use super::event_triggers::event_trigger;

/// Changes held between bursts; more than this inside one window add nothing.
const CHANGE_BUFFER: usize = 16;

/// The wait between retries once a watch keeps failing.
const SLOW_RETRY_INTERVAL: Duration = Duration::from_secs(300);

/// Why a watch on a trigger's path stopped.
enum WatchEnd {
    /// The watch could not be set up.
    Failed(String),
    /// The watch was set up, then lost.
    Lost(String),
}

impl AppState {
    /// Fire `PathChanged` trigger `trigger` of task `task` once per burst of changes
    /// under its path, retrying with backoff whenever the watch fails or is lost, for
    /// as long as the future runs.
    pub(in crate::state) async fn watch_path(self: Arc<Self>, task: TaskId, trigger: TaskTrigger) {
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
            self.runtime.task_events.record_watch_failure(
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
        self.runtime.task_events.clear_watch_failure(task, trigger);
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

/// Register the next retry of `task`'s watch and pick how long to wait before it, at
/// most [`SLOW_RETRY_INTERVAL`].
fn retry_delay(task: TaskId, policy: &mut RestartPolicy) -> Duration {
    match policy.next_restart(Instant::now()) {
        RestartDecision::Backoff { delay, .. } => delay,
        RestartDecision::ConsecutiveCapReached { failures } => {
            error!(
                target: "assistd::tasks",
                %task,
                failures,
                retry_secs = SLOW_RETRY_INTERVAL.as_secs(),
                "watching the trigger's path failed {failures} times in a row; slowing retries"
            );
            SLOW_RETRY_INTERVAL
        }
        RestartDecision::WindowCapReached { restarts } => {
            error!(
                target: "assistd::tasks",
                %task,
                restarts,
                retry_secs = SLOW_RETRY_INTERVAL.as_secs(),
                "the trigger's path watch restarted {restarts} times in the rolling window; slowing retries"
            );
            SLOW_RETRY_INTERVAL
        }
    }
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

/// An inotify watch on `path` and everything under it, sending to `changes` on each
/// change and why to `lost` once `path` itself is removed or moved, or the watch errs.
fn watch_changes(
    path: &Path,
    changes: mpsc::Sender<()>,
    lost: watch::Sender<Option<String>>,
) -> notify::Result<RecommendedWatcher> {
    let root = path.to_path_buf();
    let mut watcher =
        notify::recommended_watcher(move |event: notify::Result<notify::Event>| match event {
            Ok(event) if removes(&event, &root) => {
                lost.send_replace(Some("the path was removed or moved".into()));
            }
            Ok(event) if is_change(event.kind) => {
                let _ = changes.try_send(());
            }
            Ok(_) => {}
            Err(e) => {
                lost.send_replace(Some(e.to_string()));
            }
        })?;
    watcher.watch(path, RecursiveMode::Recursive)?;
    Ok(watcher)
}

/// Whether `event` removes or moves away `root` itself, which ends a watch on it.
fn removes(event: &notify::Event, root: &Path) -> bool {
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
mod tests;
