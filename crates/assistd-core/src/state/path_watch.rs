//! Watching a `PathChanged` trigger's path with inotify and gathering each
//! burst of changes under it into one fire.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use notify::event::{ModifyKind, RenameMode};
use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::mpsc;
use tracing::warn;

use assistd_ipc::{TaskEvent, TaskId, TaskTrigger};

use super::AppState;
use super::event_triggers::event_trigger;

/// Changes held between bursts; more than this inside one window add nothing.
const CHANGE_BUFFER: usize = 16;

impl AppState {
    /// Fire `PathChanged` trigger `trigger` of task `task` once per burst of changes
    /// under its path, for as long as the future runs.
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
        let (tx, mut changes) = mpsc::channel(CHANGE_BUFFER);
        let _watcher = match watch_changes_off_runtime(path.clone(), tx).await {
            Ok(watcher) => watcher,
            Err(e) => {
                warn!(
                    target: "assistd::tasks",
                    %task,
                    path = %path.display(),
                    error = %e,
                    "watching the trigger's path failed"
                );
                return;
            }
        };
        while next_burst(&mut changes, debounce).await.is_some() {
            self.fire_watched_trigger(task, &trigger).await;
        }
    }
}

/// [`watch_changes`] run on the blocking pool, since a recursive watch walks the
/// whole tree adding one inotify watch per directory.
async fn watch_changes_off_runtime(
    path: PathBuf,
    tx: mpsc::Sender<()>,
) -> notify::Result<RecommendedWatcher> {
    tokio::task::spawn_blocking(move || watch_changes(&path, tx))
        .await
        .map_err(|e| notify::Error::generic(&e.to_string()))?
}

/// An inotify watch on `path` and everything under it, sending to `tx` on each change.
fn watch_changes(path: &Path, tx: mpsc::Sender<()>) -> notify::Result<RecommendedWatcher> {
    let mut watcher = notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
        if event.is_ok_and(|event| is_change(event.kind)) {
            let _ = tx.try_send(());
        }
    })?;
    watcher.watch(path, RecursiveMode::Recursive)?;
    Ok(watcher)
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
