//! `TaskRunner`: the task runs in flight, each cancellable on its own and all
//! stopped together at shutdown.

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex as StdMutex;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use assistd_ipc::TaskRunId;

type RunningMap = Arc<StdMutex<HashMap<TaskRunId, RunCancel>>>;

/// Task runs in flight in the background, by run id.
#[derive(Debug, Default)]
pub struct TaskRunner {
    tracker: TaskTracker,
    running: RunningMap,
    /// Parent of every run's token; fired once the daemon stops.
    stopping: CancellationToken,
}

/// Cancels one run, remembering that it was asked to.
#[derive(Debug, Clone)]
struct RunCancel {
    token: CancellationToken,
    requested: Arc<AtomicBool>,
}

/// A run's place in a [`TaskRunner`], given up when dropped.
#[derive(Debug)]
pub(in crate::state) struct RegisteredRun {
    run: TaskRunId,
    /// Fired by [`TaskRunner::cancel`] and [`TaskRunner::stop`].
    pub(in crate::state) cancel: CancellationToken,
    requested: Arc<AtomicBool>,
    running: RunningMap,
}

impl TaskRunner {
    /// Run the future `start` builds for `run` in the background, handing it the run's
    /// registration, which [`Self::cancel`] and [`Self::stop`] cancel.
    pub(in crate::state) fn spawn<F>(&self, run: TaskRunId, start: impl FnOnce(RegisteredRun) -> F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let cancel = RunCancel {
            token: self.stopping.child_token(),
            requested: Arc::default(),
        };
        self.running.lock().insert(run, cancel.clone());
        let registered = RegisteredRun {
            run,
            cancel: cancel.token,
            requested: cancel.requested,
            running: Arc::clone(&self.running),
        };
        self.tracker.spawn(start(registered));
    }

    /// Cancel run `run`; `false` when it is not in flight.
    pub(in crate::state) fn cancel(&self, run: TaskRunId) -> bool {
        let running = self.running.lock();
        let Some(cancel) = running.get(&run) else {
            return false;
        };
        cancel.requested.store(true, Ordering::Release);
        cancel.token.cancel();
        true
    }

    /// Cancel every run and wait for each to record how it ended.
    pub async fn stop(&self) {
        self.stopping.cancel();
        self.tracker.close();
        self.tracker.wait().await;
    }
}

impl RegisteredRun {
    /// Whether [`TaskRunner::cancel`] cancelled the run.
    pub(in crate::state) fn cancel_requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }
}

impl Drop for RegisteredRun {
    fn drop(&mut self) {
        self.running.lock().remove(&self.run);
    }
}
