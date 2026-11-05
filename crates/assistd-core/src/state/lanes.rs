//! `ModelLanes`: the conversations the model decodes at once, one per server
//! slot, of which the first is always the main conversation's.

use std::collections::BTreeMap;
use std::future::pending;
use std::num::NonZeroU32;
use std::sync::Arc;

use tokio::sync::{Mutex, OwnedMutexGuard, watch};

use assistd_ipc::{LaneHolder, TaskRunId};
use assistd_llm::{ConversationKey, SlotAssignment};

/// The main conversation's lane, and the slot its requests are pinned to.
const MAIN_LANE: u32 = 0;

/// The model's lanes. With one slot a task run takes the main lane only between user
/// turns; with more, task runs take the others, one each.
#[derive(Debug)]
pub(in crate::state) struct ModelLanes {
    /// Held by every user turn, and by a task run on the main lane.
    main: Arc<Mutex<()>>,
    slots: watch::Receiver<NonZeroU32>,
    /// The task run holding each lane it holds.
    held: watch::Sender<BTreeMap<u32, TaskRunId>>,
}

impl ModelLanes {
    /// Lanes over the server's slot count `slots`, sharing the main lane `main` with
    /// user turns. A closed `slots` channel leaves the count where it is.
    pub(in crate::state) fn new(
        main: Arc<Mutex<()>>,
        slots: watch::Receiver<NonZeroU32>,
    ) -> Arc<Self> {
        Arc::new(Self {
            main,
            slots,
            held: watch::channel(BTreeMap::new()).0,
        })
    }

    /// Wait for a lane for `run`. Cancel-safe.
    pub(in crate::state) async fn admit(self: &Arc<Self>, run: TaskRunId) -> LaneLease {
        let mut slots = self.slots.clone();
        loop {
            let total = slots.borrow_and_update().get();
            if total == 1 {
                tokio::select! {
                    guard = Arc::clone(&self.main).lock_owned() => {
                        return self.lease(run, MAIN_LANE, Some(guard));
                    }
                    () = changed_or_pending(&mut slots) => {}
                }
            } else {
                let mut held = self.held.subscribe();
                if let Some(lane) = self.claim_task_lane(run, total) {
                    return self.lease(run, lane, None);
                }
                tokio::select! {
                    _ = held.changed() => {}
                    () = changed_or_pending(&mut slots) => {}
                }
            }
        }
    }

    /// Wait for the main lane for a queued user turn. Cancel-safe.
    pub(in crate::state) async fn admit_user_turn(&self) -> OwnedMutexGuard<()> {
        Arc::clone(&self.main).lock_owned().await
    }

    /// Whether user turns and task runs share the model's only lane.
    pub(in crate::state) fn is_single(&self) -> bool {
        self.slots.borrow().get() == 1
    }

    /// Whether a task run holds the main lane.
    pub(in crate::state) fn main_held_by_run(&self) -> bool {
        self.held.borrow().contains_key(&MAIN_LANE)
    }

    /// What holds each lane now.
    pub(in crate::state) fn holders(&self) -> Vec<LaneHolder> {
        let total = self.slots.borrow().get();
        let held = self.held.borrow();
        (0..total)
            .map(|lane| match held.get(&lane) {
                Some(&run) => LaneHolder::Task { run },
                None if lane == MAIN_LANE => LaneHolder::Main,
                None => LaneHolder::Idle,
            })
            .collect()
    }

    /// Watch the lanes' holders and the slot count, yielding when either changes.
    pub(in crate::state) fn watch(&self) -> LanesWatch {
        LanesWatch {
            held: self.held.subscribe(),
            slots: self.slots.clone(),
        }
    }

    fn claim_task_lane(&self, run: TaskRunId, total: u32) -> Option<u32> {
        let mut claimed = None;
        self.held.send_if_modified(|held| {
            claimed = (MAIN_LANE + 1..total).find(|lane| !held.contains_key(lane));
            claimed.inspect(|&lane| {
                held.insert(lane, run);
            });
            claimed.is_some()
        });
        claimed
    }

    fn lease(
        self: &Arc<Self>,
        run: TaskRunId,
        lane: u32,
        main: Option<OwnedMutexGuard<()>>,
    ) -> LaneLease {
        if main.is_some() {
            self.held.send_modify(|held| {
                held.insert(lane, run);
            });
        }
        LaneLease {
            lane,
            lanes: Arc::clone(self),
            _main: main,
        }
    }
}

/// A task run's hold on one lane, given up when dropped.
#[derive(Debug)]
pub(in crate::state) struct LaneLease {
    lane: u32,
    lanes: Arc<ModelLanes>,
    _main: Option<OwnedMutexGuard<()>>,
}

impl LaneLease {
    /// The slot assignment for conversation `key` on this lane.
    pub(in crate::state) fn slot(&self, key: ConversationKey) -> SlotAssignment {
        if self.on_main_lane() {
            SlotAssignment::Shared(key)
        } else {
            SlotAssignment::Pinned(self.lane)
        }
    }

    /// Whether the run shares the main conversation's lane and slot.
    pub(in crate::state) fn on_main_lane(&self) -> bool {
        self.lane == MAIN_LANE
    }
}

impl Drop for LaneLease {
    fn drop(&mut self) {
        self.lanes.held.send_modify(|held| {
            held.remove(&self.lane);
        });
    }
}

/// Receivers on a [`ModelLanes`]' holders and slot count.
#[derive(Debug)]
pub(in crate::state) struct LanesWatch {
    held: watch::Receiver<BTreeMap<u32, TaskRunId>>,
    slots: watch::Receiver<NonZeroU32>,
}

impl LanesWatch {
    /// Wait for the holders or the slot count to change. Cancel-safe.
    pub(in crate::state) async fn changed(&mut self) {
        tokio::select! {
            _ = self.held.changed() => {}
            () = changed_or_pending(&mut self.slots) => {}
        }
        self.held.mark_unchanged();
        self.slots.mark_unchanged();
    }
}

/// Wait for `rx` to change; never returns once its sender is gone.
pub(in crate::state) async fn changed_or_pending<T>(rx: &mut watch::Receiver<T>) {
    if rx.changed().await.is_err() {
        pending::<()>().await;
    }
}

#[cfg(test)]
mod tests;
