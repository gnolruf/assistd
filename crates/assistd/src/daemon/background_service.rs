//! A subsystem's work running on its own task while the socket already serves.

use std::panic::AssertUnwindSafe;

use futures_util::FutureExt;
use tokio::sync::watch;
use tokio::task::JoinHandle;

/// A subsystem's background task, or none for a disabled subsystem.
pub(super) struct BackgroundService<T> {
    subsystem: &'static str,
    task: Option<JoinHandle<Option<T>>>,
}

impl<T: Send + 'static> BackgroundService<T> {
    pub(super) fn disabled(subsystem: &'static str) -> Self {
        Self {
            subsystem,
            task: None,
        }
    }

    /// Run `work` on a task; a panic in it is logged.
    pub(super) fn spawn(
        subsystem: &'static str,
        work: impl Future<Output = T> + Send + 'static,
    ) -> Self {
        Self::spawn_marking(subsystem, work, || {})
    }

    /// Run `work` on a task. If it panics, `mark_unavailable` runs instead of
    /// leaving the subsystem starting.
    pub(super) fn spawn_marking(
        subsystem: &'static str,
        work: impl Future<Output = T> + Send + 'static,
        mark_unavailable: impl FnOnce() + Send + 'static,
    ) -> Self {
        let task = tokio::spawn(async move {
            let finished = AssertUnwindSafe(work).catch_unwind().await.ok();
            if finished.is_none() {
                tracing::error!("{subsystem}: background task panicked; disabled this run");
                mark_unavailable();
            }
            finished
        });
        Self {
            subsystem,
            task: Some(task),
        }
    }

    /// Whether the task has stopped; true for a disabled subsystem.
    pub(super) fn is_finished(&self) -> bool {
        self.task.as_ref().is_none_or(JoinHandle::is_finished)
    }

    /// Wait for the task to stop and take what it returned: nothing when the
    /// subsystem is disabled or its task panicked.
    pub(super) async fn join(self) -> Option<T> {
        self.task?.await.unwrap_or_else(|e| {
            tracing::error!("{}: background task failed: {e}", self.subsystem);
            None
        })
    }
}

/// Call `publish` with each new value of `changes` until `shutdown` flips.
pub(super) async fn publish_changes<T>(
    mut changes: watch::Receiver<T>,
    mut shutdown: watch::Receiver<bool>,
    mut publish: impl FnMut(&T),
) {
    loop {
        tokio::select! {
            biased;
            _ = shutdown.wait_for(|stop| *stop) => return,
            changed = changes.changed() => {
                if changed.is_err() {
                    return;
                }
                publish(&changes.borrow_and_update());
            }
        }
    }
}
