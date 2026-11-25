//! `WatchedPaths`: the paths saved tasks watch, each writable only by a run of its task.

use std::path::{Path, PathBuf};

use parking_lot::RwLock;

use assistd_ipc::TaskId;

/// A path whose changes start a run of `task`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchedPath {
    pub task: TaskId,
    pub task_name: String,
    /// Canonical; a file, or a directory watched with everything under it.
    pub path: PathBuf,
}

impl WatchedPath {
    /// Why only a run of the task may change `path`, as written by whoever named it.
    pub(crate) fn refusal(&self, path: &str) -> String {
        format!(
            "`{path}` is watched by the task \"{}\", so only that task's runs may change it",
            self.task_name
        )
    }
}

/// The paths enabled tasks watch, which a turn may change only as a run of the task
/// watching them.
#[derive(Debug, Default)]
pub struct WatchedPaths(RwLock<Vec<WatchedPath>>);

impl WatchedPaths {
    /// Watch `watched` and nothing else.
    pub fn replace(&self, watched: Vec<WatchedPath>) {
        *self.0.write() = watched;
    }

    /// What a run of task `run_of`, or with `None` any other turn, may not change: every
    /// path another task watches, also where its own task watches the same place.
    pub fn locked_for(&self, run_of: Option<TaskId>) -> LockedPaths {
        let watched = self.0.read();
        LockedPaths(
            watched
                .iter()
                .filter(|watched| Some(watched.task) != run_of)
                .cloned()
                .collect(),
        )
    }
}

/// The watched paths one turn may not change.
#[derive(Debug, Clone, Default)]
pub struct LockedPaths(Vec<WatchedPath>);

impl LockedPaths {
    /// The watch the canonical `path` is at or under, if any.
    pub(crate) fn over(&self, path: &Path) -> Option<&WatchedPath> {
        self.0
            .iter()
            .find(|watched| path.starts_with(&watched.path))
    }

    pub(crate) fn paths(&self) -> impl Iterator<Item = &Path> {
        self.0.iter().map(|watched| watched.path.as_path())
    }
}
