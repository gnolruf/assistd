//! `Activity`: who is using the model at one moment, and the scheduling rules read
//! off it. `ActivityWatch` assembles it and resolves when any part of it changes.

use std::collections::BTreeMap;
use std::num::NonZeroU32;

use tokio::sync::watch;
use tokio::time::Instant;

use assistd_config::TaskInterruption;
use assistd_ipc::{InteractionHold, LaneHolder, SuspendReason, TaskRunId};

use super::interaction::InteractionState;
use super::lanes::{MAIN_LANE, changed_or_pending};
use crate::presence::SettledPresence;

/// Who is using the model: the user's interaction, the task run on each lane that one
/// holds, the number of lanes, and presence's settled state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Activity {
    interaction: InteractionState,
    runs: BTreeMap<u32, TaskRunId>,
    lanes: NonZeroU32,
    presence: SettledPresence,
}

impl Activity {
    /// Why waiting task runs may not start: a sleep for the GPU first, then the user's
    /// interaction on the model's only lane; `None` when they may.
    pub(crate) fn waiting_runs_held(&self) -> Option<SuspendReason> {
        if self.asleep_for_gpu() {
            Some(SuspendReason::GpuBusy)
        } else {
            self.interaction_holds_queue()
                .then_some(SuspendReason::UserInteraction)
        }
    }

    /// Whether `run` holds the main lane and must give it to the user's interaction
    /// now, under `interruption`.
    pub(crate) fn run_must_yield(&self, run: TaskRunId, interruption: TaskInterruption) -> bool {
        interruption == TaskInterruption::Interrupt
            && self.interaction_holds_queue()
            && self.runs.get(&MAIN_LANE) == Some(&run)
    }

    /// The hold the queue snapshot shows: the user's interaction on the only lane,
    /// with when it ends once it has gone quiet.
    pub(crate) fn interaction_hold(&self) -> Option<InteractionHold> {
        if !self.single_lane() {
            return None;
        }
        match self.interaction {
            InteractionState::Idle => None,
            InteractionState::Talking { .. } => Some(InteractionHold { resumes_at: None }),
            InteractionState::Quiet { resumes_at, .. } => Some(InteractionHold {
                resumes_at: Some(resumes_at),
            }),
        }
    }

    /// Whether the user's interaction keeps task runs off the model's only lane.
    pub(crate) fn interaction_holds_queue(&self) -> bool {
        self.single_lane() && self.interaction_open()
    }

    /// Whether the user's interaction lasts, its quiet time included.
    pub(crate) fn interaction_open(&self) -> bool {
        self.interaction != InteractionState::Idle
    }

    /// Whether a user turn is waiting for the lane or running.
    pub(crate) fn turn_in_flight(&self) -> bool {
        matches!(self.interaction, InteractionState::Talking { .. })
    }

    /// When a quiet interaction's quiet time runs out; `None` for any other.
    pub(crate) fn quiet_until(&self) -> Option<Instant> {
        match self.interaction {
            InteractionState::Quiet { until, .. } => Some(until),
            InteractionState::Idle | InteractionState::Talking { .. } => None,
        }
    }

    /// Whether a task run holds any lane.
    pub(crate) fn task_running(&self) -> bool {
        !self.runs.is_empty()
    }

    pub(crate) fn asleep_for_gpu(&self) -> bool {
        self.presence.asleep_for_gpu()
    }

    /// Whether user turns and task runs share the model's only lane.
    pub(super) fn single_lane(&self) -> bool {
        self.lanes.get() == 1
    }

    /// Whether a task run holds the main lane.
    pub(super) fn main_held_by_run(&self) -> bool {
        self.runs.contains_key(&MAIN_LANE)
    }

    /// What holds each lane.
    pub(super) fn lane_holders(&self) -> Vec<LaneHolder> {
        (0..self.lanes.get())
            .map(|lane| match self.runs.get(&lane) {
                Some(&run) => LaneHolder::Task { run },
                None if lane == MAIN_LANE => LaneHolder::Main,
                None => LaneHolder::Idle,
            })
            .collect()
    }
}

/// Receivers on every part of an [`Activity`].
#[derive(Debug, Clone)]
pub(crate) struct ActivityWatch {
    interaction: watch::Receiver<InteractionState>,
    runs: watch::Receiver<BTreeMap<u32, TaskRunId>>,
    lanes: watch::Receiver<NonZeroU32>,
    presence: watch::Receiver<SettledPresence>,
}

impl ActivityWatch {
    pub(super) fn new(
        interaction: watch::Receiver<InteractionState>,
        runs: watch::Receiver<BTreeMap<u32, TaskRunId>>,
        lanes: watch::Receiver<NonZeroU32>,
        presence: watch::Receiver<SettledPresence>,
    ) -> Self {
        Self {
            interaction,
            runs,
            lanes,
            presence,
        }
    }

    /// The activity now, after which [`Self::changed`] waits for the next change.
    pub(crate) fn current(&mut self) -> Activity {
        Activity {
            interaction: *self.interaction.borrow_and_update(),
            runs: self.runs.borrow_and_update().clone(),
            lanes: *self.lanes.borrow_and_update(),
            presence: *self.presence.borrow_and_update(),
        }
    }

    /// Wait for any part of the activity to change. Cancel-safe.
    pub(crate) async fn changed(&mut self) {
        tokio::select! {
            () = changed_or_pending(&mut self.interaction) => {}
            () = changed_or_pending(&mut self.runs) => {}
            () = changed_or_pending(&mut self.lanes) => {}
            () = changed_or_pending(&mut self.presence) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use chrono::{DateTime, Utc};

    use super::*;
    use crate::presence::SleepCause;

    const RUN: TaskRunId = TaskRunId(7);
    const TALKING: InteractionState = InteractionState::Talking {
        turns: NonZeroU32::MIN,
        ends_with_turn: false,
    };

    fn quiet(resumes_at: DateTime<Utc>) -> InteractionState {
        InteractionState::Quiet {
            until: Instant::now() + Duration::from_secs(60),
            resumes_at,
        }
    }

    fn activity(
        interaction: InteractionState,
        runs: &[(u32, TaskRunId)],
        lanes: u32,
        presence: SettledPresence,
    ) -> Activity {
        Activity {
            interaction,
            runs: runs.iter().copied().collect(),
            lanes: NonZeroU32::new(lanes).unwrap(),
            presence,
        }
    }

    /// What the three rules say of `activity`: why waiting runs are held, whether
    /// `RUN` must yield under each of `interrupt`, `after_task` and `after_queue`,
    /// and the hold the snapshot shows.
    fn rules(activity: &Activity) -> (Option<SuspendReason>, [bool; 3], Option<InteractionHold>) {
        let must_yield = [
            TaskInterruption::Interrupt,
            TaskInterruption::AfterTask,
            TaskInterruption::AfterQueue,
        ]
        .map(|interruption| activity.run_must_yield(RUN, interruption));
        (
            activity.waiting_runs_held(),
            must_yield,
            activity.interaction_hold(),
        )
    }

    #[test]
    fn rules_follow_the_interaction_the_lanes_and_the_gpu_sleep() {
        let resumes_at = Utc::now();
        let active = SettledPresence::Active;
        let gpu_sleep = SettledPresence::Sleeping(SleepCause::GpuBusy);
        let on_main = [(MAIN_LANE, RUN)];
        let talking_hold = Some(InteractionHold { resumes_at: None });
        let quiet_hold = Some(InteractionHold {
            resumes_at: Some(resumes_at),
        });
        let user = Some(SuspendReason::UserInteraction);
        let gpu = Some(SuspendReason::GpuBusy);
        let only_interrupt = [true, false, false];
        let never = [false; 3];
        let cases = [
            (
                "idle",
                activity(InteractionState::Idle, &on_main, 1, active),
                (None, never, None),
            ),
            (
                "talking on the only lane",
                activity(TALKING, &on_main, 1, active),
                (user, only_interrupt, talking_hold.clone()),
            ),
            (
                "quiet on the only lane",
                activity(quiet(resumes_at), &on_main, 1, active),
                (user, only_interrupt, quiet_hold),
            ),
            (
                "another run holds the only lane",
                activity(TALKING, &[(MAIN_LANE, TaskRunId(8))], 1, active),
                (user, never, talking_hold.clone()),
            ),
            (
                "no run holds the only lane",
                activity(TALKING, &[], 1, active),
                (user, never, talking_hold.clone()),
            ),
            (
                "talking beside a task lane",
                activity(TALKING, &[(1, RUN)], 2, active),
                (None, never, None),
            ),
            (
                "the slot count fell to one under a run on a task lane",
                activity(TALKING, &[(1, RUN)], 1, active),
                (user, never, talking_hold.clone()),
            ),
            (
                "asleep for the GPU",
                activity(InteractionState::Idle, &[], 2, gpu_sleep),
                (gpu, never, None),
            ),
            (
                "asleep for the GPU while talking",
                activity(TALKING, &on_main, 1, gpu_sleep),
                (gpu, only_interrupt, talking_hold.clone()),
            ),
            (
                "asleep for another cause",
                activity(
                    InteractionState::Idle,
                    &[],
                    1,
                    SettledPresence::Sleeping(SleepCause::Idle),
                ),
                (None, never, None),
            ),
        ];
        for (name, activity, expected) in cases {
            assert_eq!(rules(&activity), expected, "{name}");
        }
    }
}
