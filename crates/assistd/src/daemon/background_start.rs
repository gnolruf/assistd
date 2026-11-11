//! A subsystem start that runs on its own task while the socket already serves.

use std::panic::AssertUnwindSafe;

use futures_util::FutureExt;
use tokio::task::JoinHandle;

/// A start running in the background, or none for a disabled subsystem.
pub(super) struct BackgroundStart<T> {
    subsystem: &'static str,
    task: Option<JoinHandle<Option<T>>>,
}

impl<T: Send + 'static> BackgroundStart<T> {
    pub(super) fn disabled(subsystem: &'static str) -> Self {
        Self {
            subsystem,
            task: None,
        }
    }

    /// Run `start` on a task. If it panics, `mark_unavailable` runs and the
    /// start yields `None` instead of leaving the subsystem starting.
    pub(super) fn spawn(
        subsystem: &'static str,
        start: impl Future<Output = Option<T>> + Send + 'static,
        mark_unavailable: impl FnOnce() + Send + 'static,
    ) -> Self {
        let task = tokio::spawn(async move {
            let started = AssertUnwindSafe(start).catch_unwind().await;
            started.unwrap_or_else(|_| {
                tracing::error!("{subsystem}: startup panicked; disabled this run");
                mark_unavailable();
                None
            })
        });
        Self {
            subsystem,
            task: Some(task),
        }
    }

    /// Whether the start has stopped; true for a disabled subsystem.
    pub(super) fn is_finished(&self) -> bool {
        self.task.as_ref().is_none_or(JoinHandle::is_finished)
    }

    /// Wait for the start to stop and take what it brought up.
    pub(super) async fn join(self) -> Option<T> {
        self.task?.await.unwrap_or_else(|e| {
            tracing::error!("{}: startup task failed: {e}", self.subsystem);
            None
        })
    }
}
