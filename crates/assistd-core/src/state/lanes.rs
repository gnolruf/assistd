//! `ModelLanes`: the turns the model decodes at once, one per server
//! slot, of which the first is always the main backend's.

use std::collections::BTreeMap;
use std::future::pending;
use std::num::NonZeroU32;
use std::sync::Arc;

use tokio::sync::{Mutex, OwnedMutexGuard, watch};

use assistd_ipc::{LaneHolder, TaskRunId};
use assistd_llm::{SlotAssignment, SlotHolder};

/// The main backend's lane, and the slot its requests are pinned to.
const MAIN_LANE: u32 = 0;

/// The model's lanes. With one slot a task run takes the main lane only between user
/// turns; with more, task runs take the others, one each.
#[derive(Debug)]
pub(crate) struct ModelLanes {
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
    pub(crate) async fn admit(self: &Arc<Self>, run: TaskRunId) -> LaneLease {
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
    pub(crate) async fn admit_user_turn(&self) -> OwnedMutexGuard<()> {
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
    pub(crate) fn watch(&self) -> LanesWatch {
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
pub(crate) struct LaneLease {
    lane: u32,
    lanes: Arc<ModelLanes>,
    _main: Option<OwnedMutexGuard<()>>,
}

impl LaneLease {
    /// The slot assignment for `holder` on this lane.
    pub(crate) fn slot(&self, holder: SlotHolder) -> SlotAssignment {
        if self.on_main_lane() {
            SlotAssignment::Shared(holder)
        } else {
            SlotAssignment::Pinned(self.lane)
        }
    }

    /// Whether the run shares the main backend's lane and slot.
    pub(crate) fn on_main_lane(&self) -> bool {
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
pub(crate) struct LanesWatch {
    held: watch::Receiver<BTreeMap<u32, TaskRunId>>,
    slots: watch::Receiver<NonZeroU32>,
}

impl LanesWatch {
    /// Wait for the holders or the slot count to change. Cancel-safe.
    pub(crate) async fn changed(&mut self) {
        tokio::select! {
            _ = self.held.changed() => {}
            () = changed_or_pending(&mut self.slots) => {}
        }
        self.held.mark_unchanged();
        self.slots.mark_unchanged();
    }
}

/// Wait for `rx` to change; never returns once its sender is gone.
pub(crate) async fn changed_or_pending<T>(rx: &mut watch::Receiver<T>) {
    if rx.changed().await.is_err() {
        pending::<()>().await;
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use futures_util::FutureExt;

    use super::*;

    fn lanes(slots: u32) -> (Arc<ModelLanes>, watch::Sender<NonZeroU32>) {
        let (tx, rx) = watch::channel(NonZeroU32::new(slots).unwrap());
        (ModelLanes::new(Arc::new(Mutex::new(())), rx), tx)
    }

    #[tokio::test]
    async fn task_runs_never_outnumber_the_lanes_beside_the_main_one() {
        let (lanes, _slots) = lanes(3);
        let _user_turn = lanes.main.clone().lock_owned().await;

        let first = lanes
            .admit(TaskRunId(1))
            .now_or_never()
            .expect("a free lane");
        let second = lanes
            .admit(TaskRunId(2))
            .now_or_never()
            .expect("a free lane");
        assert_ne!(first.slot(SlotHolder::Main), second.slot(SlotHolder::Main));
        assert!(!first.on_main_lane() && !second.on_main_lane());
        let third = tokio::spawn({
            let lanes = Arc::clone(&lanes);
            async move { lanes.admit(TaskRunId(3)).await }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!third.is_finished(), "a third run took a lane");
        assert_eq!(
            lanes.holders(),
            [
                LaneHolder::Main,
                LaneHolder::Task { run: TaskRunId(1) },
                LaneHolder::Task { run: TaskRunId(2) },
            ]
        );

        drop(first);
        let third = tokio::time::timeout(Duration::from_secs(1), third)
            .await
            .expect("the freed lane went unused")
            .unwrap();
        assert_eq!(third.slot(SlotHolder::Main), SlotAssignment::Pinned(1));
    }

    #[tokio::test]
    async fn with_one_lane_a_task_run_waits_out_the_user_turn_and_holds_off_the_next() {
        let (lanes, _slots) = lanes(1);
        let user_turn = lanes.main.clone().lock_owned().await;
        let run = tokio::spawn({
            let lanes = Arc::clone(&lanes);
            async move { lanes.admit(TaskRunId(1)).await }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!run.is_finished(), "a run started during a user turn");

        drop(user_turn);
        let lease = tokio::time::timeout(Duration::from_secs(1), run)
            .await
            .expect("the run never took the free lane")
            .unwrap();
        let holder = SlotHolder::TaskRun(TaskRunId(1));
        assert_eq!(lease.slot(holder), SlotAssignment::Shared(holder));
        assert_eq!(lanes.holders(), [LaneHolder::Task { run: TaskRunId(1) }]);
        assert!(lanes.main.try_lock().is_err(), "a user turn could start");

        drop(lease);
        assert!(lanes.main.try_lock().is_ok());
        assert_eq!(lanes.holders(), [LaneHolder::Main]);
    }
}
