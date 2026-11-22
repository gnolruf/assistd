//! Model lanes: the turns the model decodes at once, one per server slot, of which
//! the first is always the main backend's, and the `LaneLease` that holds one.

use std::future::pending;
use std::sync::Arc;

use tokio::sync::watch;

use assistd_ipc::TaskRunId;
use assistd_llm::{SlotAssignment, SlotHolder};

use super::queue::TaskQueue;

/// The main backend's lane, and the slot its requests are pinned to.
pub(super) const MAIN_LANE: u32 = 0;

/// What holds a lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LaneUse {
    UserTurn,
    /// A task prompt being refined.
    Refine,
    /// A session title being generated.
    Title,
    Run(TaskRunId),
}

/// A hold on one lane, which nothing holds any other way; dropping it frees the lane.
#[derive(Debug)]
pub(crate) struct LaneLease {
    lane: u32,
    queue: Arc<TaskQueue>,
}

impl LaneLease {
    pub(super) fn new(lane: u32, queue: Arc<TaskQueue>) -> Self {
        Self { lane, queue }
    }

    pub(super) fn lane(&self) -> u32 {
        self.lane
    }

    /// The slot assignment for `holder` on this lane.
    pub(crate) fn slot(&self, holder: SlotHolder) -> SlotAssignment {
        if self.on_main_lane() {
            SlotAssignment::Shared(holder)
        } else {
            SlotAssignment::Pinned(self.lane)
        }
    }

    /// Whether this is the main backend's lane and slot.
    pub(crate) fn on_main_lane(&self) -> bool {
        self.lane == MAIN_LANE
    }
}

impl Drop for LaneLease {
    fn drop(&mut self) {
        self.queue.release(self.lane);
    }
}

/// Wait for `rx` to change; never returns once its sender is gone.
pub(crate) async fn changed_or_pending<T>(rx: &mut watch::Receiver<T>) {
    if rx.changed().await.is_err() {
        pending::<()>().await;
    }
}
