//! `Activity`: who is using the model at one moment, and the scheduling rules read
//! off it. `ActivityWatch` assembles it and resolves when any part of it changes.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;

use tokio::sync::watch;
use tokio::time::Instant;

use assistd_config::{Config, TaskInterruption};
use assistd_ipc::{Component, InteractionHold, LaneHolder, SuspendReason, TaskRunId};

use super::interaction::InteractionState;
use super::lanes::{LaneUse, MAIN_LANE, changed_or_pending};
use super::queue::QueueState;
use crate::presence::SettledPresence;

/// A class of background work that waits until the machine is free for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Work {
    /// Embedding the chunks of indexed documents.
    DocumentEmbedding,
    /// Reading pages that have no text with the OCR model.
    PageTranscription,
}

impl Work {
    pub(super) fn component(self) -> Component {
        match self {
            Work::DocumentEmbedding => Component::Embed,
            Work::PageTranscription => Component::Documents,
        }
    }
}

/// Who is using the model: the user's interaction, what holds each lane that is held,
/// the number of lanes, and presence's settled state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Activity {
    interaction: InteractionState,
    holders: BTreeMap<u32, LaneUse>,
    lanes: NonZeroU32,
    presence: SettledPresence,
}

impl Activity {
    pub(super) fn new(
        interaction: InteractionState,
        holders: BTreeMap<u32, LaneUse>,
        lanes: NonZeroU32,
        presence: SettledPresence,
    ) -> Self {
        Self {
            interaction,
            holders,
            lanes,
            presence,
        }
    }

    /// Whether `work` may run now. Document embedding may outside the user's interaction,
    /// a task run and a sleep for the GPU, and during each one `[documents]` embeds
    /// during; page transcription may while presence has settled in `Drowsy`.
    pub(crate) fn permits(&self, work: Work, config: &Config) -> bool {
        match work {
            Work::DocumentEmbedding => {
                let documents = &config.documents;
                (documents.embed_during_interaction || !self.interaction_open())
                    && (documents.embed_during_tasks || !self.task_running())
                    && (documents.embed_during_gpu_sleep || !self.asleep_for_gpu())
            }
            Work::PageTranscription => self.presence == SettledPresence::Drowsy,
        }
    }

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
            && self.holders.get(&MAIN_LANE) == Some(&LaneUse::Run(run))
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
        self.holders
            .values()
            .any(|holder| matches!(holder, LaneUse::Run(_)))
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
        matches!(self.holders.get(&MAIN_LANE), Some(LaneUse::Run(_)))
    }

    pub(super) fn main_lane_free(&self) -> bool {
        !self.holders.contains_key(&MAIN_LANE)
    }

    /// The lanes nothing holds.
    pub(super) fn free_lanes(&self) -> BTreeSet<u32> {
        (0..self.lanes.get())
            .filter(|lane| !self.holders.contains_key(lane))
            .collect()
    }

    /// Whether presence has settled in `Active`.
    pub(super) fn settled_active(&self) -> bool {
        self.presence == SettledPresence::Active
    }

    /// Each lane's holder as the wire shows it: a task run, else the main backend on
    /// its own lane.
    pub(super) fn lane_holders(&self) -> Vec<LaneHolder> {
        (0..self.lanes.get())
            .map(|lane| match self.holders.get(&lane) {
                Some(&LaneUse::Run(run)) => LaneHolder::Task { run },
                _ if lane == MAIN_LANE => LaneHolder::Main,
                _ => LaneHolder::Idle,
            })
            .collect()
    }
}

/// Receivers on every part of an [`Activity`].
#[derive(Debug, Clone)]
pub(crate) struct ActivityWatch {
    interaction: watch::Receiver<InteractionState>,
    queue: watch::Receiver<QueueState>,
    lanes: watch::Receiver<NonZeroU32>,
    presence: watch::Receiver<SettledPresence>,
}

impl ActivityWatch {
    pub(super) fn new(
        interaction: watch::Receiver<InteractionState>,
        queue: watch::Receiver<QueueState>,
        lanes: watch::Receiver<NonZeroU32>,
        presence: watch::Receiver<SettledPresence>,
    ) -> Self {
        Self {
            interaction,
            queue,
            lanes,
            presence,
        }
    }

    /// The activity now, after which [`Self::changed`] waits for the next change.
    pub(crate) fn current(&mut self) -> Activity {
        Activity {
            interaction: *self.interaction.borrow_and_update(),
            holders: self.queue.borrow_and_update().lane_holders().clone(),
            lanes: *self.lanes.borrow_and_update(),
            presence: *self.presence.borrow_and_update(),
        }
    }

    /// Wait for any part of the activity to change. Cancel-safe.
    pub(crate) async fn changed(&mut self) {
        tokio::select! {
            () = changed_or_pending(&mut self.interaction) => {}
            () = changed_or_pending(&mut self.queue) => {}
            () = changed_or_pending(&mut self.lanes) => {}
            () = changed_or_pending(&mut self.presence) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use assistd_config::DocumentsConfig;
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
            holders: runs
                .iter()
                .map(|&(lane, run)| (lane, LaneUse::Run(run)))
                .collect(),
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

    /// Whether document chunks may be embedded under `documents` while idle, in the
    /// user's interaction as it talks and goes quiet, in a task run and in a GPU sleep.
    fn embedding_permitted(documents: DocumentsConfig) -> [bool; 5] {
        let config = Config {
            documents,
            ..Config::default()
        };
        let active = SettledPresence::Active;
        let gpu_sleep = SettledPresence::Sleeping(SleepCause::GpuBusy);
        [
            activity(InteractionState::Idle, &[], 1, active),
            activity(TALKING, &[], 1, active),
            activity(quiet(Utc::now()), &[], 2, active),
            activity(InteractionState::Idle, &[(1, RUN)], 2, active),
            activity(InteractionState::Idle, &[], 1, gpu_sleep),
        ]
        .map(|activity| activity.permits(Work::DocumentEmbedding, &config))
    }

    #[test]
    fn document_embedding_is_refused_in_each_busy_state_except_the_ones_config_embeds_during() {
        let defaults = DocumentsConfig::default();
        let cases = [
            (defaults.clone(), [true, false, false, false, false]),
            (
                DocumentsConfig {
                    embed_during_interaction: true,
                    ..defaults.clone()
                },
                [true, true, true, false, false],
            ),
            (
                DocumentsConfig {
                    embed_during_tasks: true,
                    ..defaults.clone()
                },
                [true, false, false, true, false],
            ),
            (
                DocumentsConfig {
                    embed_during_gpu_sleep: true,
                    ..defaults
                },
                [true, false, false, false, true],
            ),
        ];
        for (documents, permitted) in cases {
            assert_eq!(
                embedding_permitted(documents.clone()),
                permitted,
                "{documents:?}"
            );
        }
    }

    #[test]
    fn page_transcription_is_permitted_only_while_settled_drowsy() {
        let config = Config::default();
        let cases = [
            (SettledPresence::Drowsy, true),
            (SettledPresence::Active, false),
            (SettledPresence::Sleeping(SleepCause::Idle), false),
            (SettledPresence::Sleeping(SleepCause::GpuBusy), false),
        ];
        for (presence, permitted) in cases {
            let activity = activity(InteractionState::Idle, &[], 1, presence);
            assert_eq!(
                activity.permits(Work::PageTranscription, &config),
                permitted,
                "{presence:?}"
            );
        }
    }
}
