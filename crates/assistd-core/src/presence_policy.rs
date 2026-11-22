//! The task that unloads and stops the model after the configured idle periods, sleeps
//! the daemon when another process grows into the model's VRAM, and wakes it again.

use std::future::pending;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use assistd_config::SleepConfig;
use assistd_ipc::{Component, PresenceState};
use assistd_utils::backoff::SLOW_RETRY_INTERVAL;

use crate::admission::lanes::changed_or_pending;
use crate::presence::{PresenceError, PresenceManager, SleepCause};
use crate::recovery::spawn_supervised;
use crate::vram::{VramLedger, VramSnapshot};

const GPU_POLL_INTERVAL: Duration = Duration::from_secs(5);
/// Failed NVML polls in a row after which polling slows to [`SLOW_RETRY_INTERVAL`].
const MAX_CONSECUTIVE_FAILURES: u32 = 10;
const IDLE_RETRY_INTERVAL: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IdleStep {
    Drowse,
    Sleep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    /// Take `step` unless the model was used within the last `after`.
    Idle {
        step: IdleStep,
        after: Duration,
    },
    SleepForGpu(VramSnapshot),
    WakeIntoRoom,
}

/// What keeps the idle step from being tried again once it has failed.
#[derive(Debug, Clone, Copy)]
enum IdleHold {
    None,
    /// The model was in use; lasts until the activity stamp moves on from this one.
    InUseSince(Instant),
    RetryAt(Instant),
}

impl IdleHold {
    /// How long to wait before an idle step due `after` the use at `last_use`;
    /// `None` while that use has not ended.
    fn wait(self, last_use: Instant, after: Duration) -> Option<Duration> {
        let until_idle = after.saturating_sub(last_use.elapsed());
        match self {
            IdleHold::InUseSince(stamp) if stamp == last_use => None,
            IdleHold::None | IdleHold::InUseSince(_) => Some(until_idle),
            IdleHold::RetryAt(at) => {
                Some(until_idle.max(at.saturating_duration_since(Instant::now())))
            }
        }
    }
}

/// What one NVML poll found, as `decide_gpu` sees it.
#[derive(Debug, Clone, Copy)]
struct Poll {
    state: PresenceState,
    asleep_for_gpu: bool,
    snapshot: VramSnapshot,
    contended: bool,
    /// Whether the model would fit beside the headroom if woken now.
    room_to_wake: bool,
}

/// The NVML polls and their pace, which slows while they keep failing.
struct GpuPolls {
    ledger: Arc<VramLedger>,
    next_poll: tokio::time::Instant,
    failures_in_a_row: u32,
}

impl GpuPolls {
    fn new(ledger: Arc<VramLedger>) -> Self {
        Self {
            ledger,
            next_poll: tokio::time::Instant::now(),
            failures_in_a_row: 0,
        }
    }

    /// Wait for the next poll and take it; `None` when NVML could not be read.
    async fn next(&mut self, presence: &PresenceManager) -> Option<Poll> {
        tokio::time::sleep_until(self.next_poll).await;
        let state = presence.state();
        let read = self.ledger.snapshot(presence.llama_pid());
        self.failures_in_a_row = match read {
            Ok(_) => 0,
            Err(_) => self.failures_in_a_row.saturating_add(1),
        };
        let interval = if self.failures_in_a_row >= MAX_CONSECUTIVE_FAILURES {
            SLOW_RETRY_INTERVAL
        } else {
            GPU_POLL_INTERVAL
        };
        self.next_poll = tokio::time::Instant::now() + interval;
        match read {
            Ok(snapshot) => Some(Poll {
                state,
                asleep_for_gpu: presence.settled().asleep_for_gpu(),
                snapshot,
                contended: self.ledger.observe(snapshot, state),
                room_to_wake: self.ledger.room_to_wake(snapshot).is_ok(),
            }),
            Err(e) => {
                warn!(
                    target: "assistd::presence_policy",
                    failures = self.failures_in_a_row,
                    next_poll_secs = interval.as_secs(),
                    "NVML poll failed: {e}"
                );
                None
            }
        }
    }
}

/// Spawn the presence policy, polling `ledger` when there is one. `None` when there
/// is none and both idle thresholds are 0.
pub fn spawn_policy(
    cfg: &SleepConfig,
    ledger: Option<Arc<VramLedger>>,
    presence: Arc<PresenceManager>,
    shutdown: watch::Receiver<bool>,
) -> Option<JoinHandle<()>> {
    let idle_enabled = cfg.idle_to_drowsy_mins > 0 || cfg.idle_to_sleep_mins > 0;
    if !idle_enabled && ledger.is_none() {
        info!(
            target: "assistd::presence_policy",
            "no idle threshold and no GPU to watch; presence policy disabled"
        );
        return None;
    }
    info!(
        target: "assistd::presence_policy",
        drowsy_mins = cfg.idle_to_drowsy_mins,
        sleep_mins = cfg.idle_to_sleep_mins,
        gpu_poll_secs = ledger.as_ref().map(|_| GPU_POLL_INTERVAL.as_secs()),
        gpu_headroom_mb = ledger.as_ref().map(|ledger| ledger.headroom_mb()),
        gpu_auto_wake = cfg.gpu_auto_wake,
        "presence policy enabled"
    );
    Some(spawn_supervised(
        "presence_policy",
        Component::PresencePolicy,
        run_policy(cfg.clone(), ledger, presence, shutdown),
    ))
}

async fn run_policy(
    cfg: SleepConfig,
    ledger: Option<Arc<VramLedger>>,
    presence: Arc<PresenceManager>,
    mut shutdown: watch::Receiver<bool>,
) {
    let mut gpu_polls = ledger.map(GpuPolls::new);
    let mut state_changes = presence.subscribe();
    let mut activity = presence.subscribe_activity();
    let mut idle_hold = IdleHold::None;

    loop {
        let state = *state_changes.borrow_and_update();
        let last_use = *activity.borrow_and_update();
        let idle_wait =
            next_idle_step(state, &cfg).and_then(|(_, after)| idle_hold.wait(last_use, after));
        let decided = tokio::select! {
            biased;
            () = stopped(&mut shutdown) => return,
            () = changed_or_pending(&mut state_changes) => None,
            () = changed_or_pending(&mut activity) => None,
            () = sleep_or_pending(idle_wait) => decide_idle(state, last_use.elapsed(), &cfg),
            poll = next_gpu_poll(&mut gpu_polls, &presence) => {
                poll.and_then(|poll| decide_gpu(poll, &cfg))
            }
        };
        let Some(action) = decided else {
            continue;
        };
        let transition = apply(action, Arc::clone(&presence));
        let outcome = tokio::select! {
            biased;
            () = stopped(&mut shutdown) => return,
            outcome = presence.detached("presence_policy_transition", transition) => outcome,
        };
        if matches!(action, Action::Idle { .. }) {
            idle_hold = idle_hold_after(outcome.as_ref(), last_use);
        }
        if let Some(Err(e)) = outcome {
            log_failed(action, &e);
        }
    }
}

async fn stopped(shutdown: &mut watch::Receiver<bool>) {
    let _ = shutdown.wait_for(|stopping| *stopping).await;
}

async fn sleep_or_pending(wait: Option<Duration>) {
    match wait {
        Some(wait) => tokio::time::sleep(wait).await,
        None => pending().await,
    }
}

async fn next_gpu_poll(polls: &mut Option<GpuPolls>, presence: &PresenceManager) -> Option<Poll> {
    match polls {
        Some(polls) => polls.next(presence).await,
        None => pending().await,
    }
}

async fn apply(action: Action, presence: Arc<PresenceManager>) -> Result<(), PresenceError> {
    match action {
        Action::Idle { step, after } => {
            debug!(target: "assistd::presence_policy", ?step, "idle threshold reached");
            match step {
                IdleStep::Drowse => presence.drowse_if_idle(after).await,
                IdleStep::Sleep => presence.sleep_if_idle(after).await,
            }
        }
        Action::SleepForGpu(snapshot) => {
            info!(
                target: "assistd::presence_policy",
                free_mb = snapshot.free_mb,
                foreign_mb = snapshot.foreign_mb,
                model_mb = snapshot.model_mb,
                "another process is crowding the model's VRAM; transitioning to Sleeping"
            );
            presence.sleep(SleepCause::GpuBusy).await
        }
        Action::WakeIntoRoom => {
            info!(
                target: "assistd::presence_policy",
                "the model fits in VRAM again; auto-waking"
            );
            presence.wake().await
        }
    }
}

/// The hold an idle step leaves, given how it ended and the activity stamp it was decided on.
fn idle_hold_after(outcome: Option<&Result<(), PresenceError>>, last_use: Instant) -> IdleHold {
    match outcome {
        Some(Ok(())) => IdleHold::None,
        Some(Err(PresenceError::InUse)) => IdleHold::InUseSince(last_use),
        Some(Err(_)) | None => IdleHold::RetryAt(Instant::now() + IDLE_RETRY_INTERVAL),
    }
}

fn log_failed(action: Action, error: &PresenceError) {
    match error {
        PresenceError::InUse => {
            debug!(target: "assistd::presence_policy", "the model is in use; not idle yet");
        }
        _ => warn!(target: "assistd::presence_policy", "{action:?} failed: {error:#}"),
    }
}

/// The step idleness takes from `state`, and the idle time it takes it after.
fn next_idle_step(state: PresenceState, cfg: &SleepConfig) -> Option<(IdleStep, Duration)> {
    let threshold = |mins: u64| (mins > 0).then(|| Duration::from_secs(mins * 60));
    let to_drowsy = threshold(cfg.idle_to_drowsy_mins).map(|after| (IdleStep::Drowse, after));
    let to_sleep = threshold(cfg.idle_to_sleep_mins).map(|after| (IdleStep::Sleep, after));
    match state {
        PresenceState::Active => to_drowsy.or(to_sleep),
        PresenceState::Drowsy => to_sleep,
        PresenceState::Sleeping | PresenceState::Waking | PresenceState::Downloading(_) => None,
    }
}

fn decide_idle(state: PresenceState, idle: Duration, cfg: &SleepConfig) -> Option<Action> {
    let (step, after) = next_idle_step(state, cfg)?;
    (idle >= after).then_some(Action::Idle { step, after })
}

fn decide_gpu(poll: Poll, cfg: &SleepConfig) -> Option<Action> {
    match poll.state {
        PresenceState::Active | PresenceState::Drowsy if poll.contended => {
            Some(Action::SleepForGpu(poll.snapshot))
        }
        PresenceState::Sleeping
            if poll.asleep_for_gpu && cfg.gpu_auto_wake && poll.room_to_wake =>
        {
            Some(Action::WakeIntoRoom)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use assistd_config::Config;
    use assistd_ipc::PresenceState::{Active, Drowsy, Sleeping};

    use super::*;

    const CROWDED: VramSnapshot = VramSnapshot {
        free_mb: 150,
        model_mb: 13000,
        foreign_mb: 900,
    };

    fn cfg(drowsy: u64, sleep: u64) -> SleepConfig {
        SleepConfig {
            idle_to_drowsy_mins: drowsy,
            idle_to_sleep_mins: sleep,
            gpu_auto_wake: true,
            ..Config::default().sleep
        }
    }

    fn poll(state: PresenceState, asleep_for_gpu: bool, contended: bool, room: bool) -> Poll {
        Poll {
            state,
            asleep_for_gpu,
            snapshot: CROWDED,
            contended,
            room_to_wake: room,
        }
    }

    #[test]
    fn decide_idle_by_state_idle_time_and_thresholds() {
        let cases = [
            (
                "active at drowsy",
                Active,
                30,
                (30, 120),
                Some(IdleStep::Drowse),
            ),
            (
                "active past sleep cascades via drowsy",
                Active,
                200,
                (30, 120),
                Some(IdleStep::Drowse),
            ),
            (
                "active, drowsy disabled, past sleep",
                Active,
                130,
                (0, 120),
                Some(IdleStep::Sleep),
            ),
            ("active, used since", Active, 29, (30, 120), None),
            (
                "drowsy at sleep",
                Drowsy,
                120,
                (30, 120),
                Some(IdleStep::Sleep),
            ),
            ("drowsy, sleep disabled", Drowsy, 200, (30, 0), None),
            ("sleeping", Sleeping, 500, (30, 120), None),
            ("active, both disabled", Active, 10_000, (0, 0), None),
        ];
        for (label, state, idle_mins, (drowsy, sleep), expected) in cases {
            let idle = Duration::from_secs(idle_mins * 60);
            let step = match decide_idle(state, idle, &cfg(drowsy, sleep)) {
                Some(Action::Idle { step, .. }) => Some(step),
                Some(other) => panic!("{label}: {other:?}"),
                None => None,
            };
            assert_eq!(step, expected, "{label}");
        }
    }

    #[test]
    fn decide_gpu_sleeps_on_contention_and_wakes_only_a_gpu_sleep_at_every_poll_with_room() {
        let cfg = cfg(30, 120);
        let cases = [
            (
                "drowsy, crowded",
                poll(Drowsy, false, true, false),
                Some(Action::SleepForGpu(CROWDED)),
            ),
            (
                "GPU sleep, still no room",
                poll(Sleeping, true, false, false),
                None,
            ),
            (
                "GPU sleep, fits again",
                poll(Sleeping, true, false, true),
                Some(Action::WakeIntoRoom),
            ),
            (
                "requested sleep, fits",
                poll(Sleeping, false, false, true),
                None,
            ),
        ];
        for (label, poll, expected) in cases {
            assert_eq!(decide_gpu(poll, &cfg), expected, "{label}");
        }
    }

    #[test]
    fn a_failed_idle_step_waits_for_the_use_to_end_or_for_the_retry_interval() {
        let after = Duration::from_secs(60);
        let last_use = Instant::now();
        let in_use = idle_hold_after(Some(&Err(PresenceError::InUse)), last_use);
        assert_eq!(in_use.wait(last_use, Duration::ZERO), None);
        let use_ended = last_use + Duration::from_secs(1);
        assert!(
            in_use
                .wait(use_ended, after)
                .is_some_and(|wait| wait > Duration::ZERO)
        );

        let failed = idle_hold_after(Some(&Err(PresenceError::NotDrowsy)), last_use);
        let retry_in = failed
            .wait(last_use, Duration::ZERO)
            .expect("a retry is due");
        assert!(retry_in > Duration::ZERO && retry_in <= IDLE_RETRY_INTERVAL);
    }
}
