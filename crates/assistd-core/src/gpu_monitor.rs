//! Sleeps the daemon when another process grows into the VRAM the model
//! needs, and optionally wakes it once the model fits again.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{info, warn};

use assistd_config::SleepConfig;
use assistd_ipc::{Component, PresenceState};

use crate::presence::PresenceManager;
use crate::recovery::spawn_supervised;
use crate::vram::{VramLedger, VramSnapshot};

const POLL_INTERVAL: Duration = Duration::from_secs(5);
const MAX_CONSECUTIVE_FAILURES: u32 = 10;

/// Whether the monitor caused the current `Sleeping` state; only its own
/// sleeps are auto-woken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SleepCause {
    None,
    Contention,
}

#[derive(Debug, PartialEq, Eq)]
enum Action {
    None,
    Sleep(VramSnapshot),
    Wake,
}

/// What one poll found, as `decide` sees it.
#[derive(Debug, Clone, Copy)]
struct Poll {
    state: PresenceState,
    snapshot: VramSnapshot,
    contended: bool,
    /// Whether the model would fit beside the headroom if woken now.
    room_to_wake: bool,
}

/// Spawn the contention monitor over `ledger`; `None` when there is no ledger.
pub fn spawn_monitor(
    cfg: &SleepConfig,
    ledger: Option<Arc<VramLedger>>,
    presence: Arc<PresenceManager>,
    shutdown: watch::Receiver<bool>,
) -> Option<JoinHandle<()>> {
    let ledger = ledger?;
    info!(
        target: "assistd::gpu_monitor",
        poll_secs = POLL_INTERVAL.as_secs(),
        headroom_mb = ledger.headroom_mb(),
        auto_wake = cfg.gpu_auto_wake,
        "GPU contention monitor enabled"
    );
    Some(spawn_supervised(
        "gpu_monitor",
        Component::GpuMonitor,
        run_monitor(ledger, cfg.clone(), presence, shutdown),
    ))
}

async fn run_monitor(
    ledger: Arc<VramLedger>,
    cfg: SleepConfig,
    presence: Arc<PresenceManager>,
    mut shutdown: watch::Receiver<bool>,
) {
    let mut tick = tokio::time::interval(POLL_INTERVAL);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    let mut sub = presence.subscribe();
    let mut cause = SleepCause::None;
    let mut consecutive_errors: u32 = 0;

    loop {
        tokio::select! {
            _ = tick.tick() => {
                let state = presence.state();
                let snapshot = match ledger.snapshot(presence.llama_pid()) {
                    Ok(snapshot) => {
                        consecutive_errors = 0;
                        snapshot
                    }
                    Err(e) => {
                        consecutive_errors += 1;
                        warn!(
                            target: "assistd::gpu_monitor",
                            "NVML poll failed ({consecutive_errors}x): {e}"
                        );
                        if consecutive_errors >= MAX_CONSECUTIVE_FAILURES {
                            warn!(
                                target: "assistd::gpu_monitor",
                                attempts = MAX_CONSECUTIVE_FAILURES,
                                "too many consecutive poll failures; exiting \
                                 monitor task (daemon keeps running)"
                            );
                            return;
                        }
                        continue;
                    }
                };
                let poll = Poll {
                    state,
                    snapshot,
                    contended: ledger.observe(snapshot, state),
                    room_to_wake: ledger.room_to_wake(snapshot).is_ok(),
                };
                apply(decide(poll, &cfg, cause), &presence, &mut cause).await;
            }
            _ = sub.changed() => {
                let now = *sub.borrow_and_update();
                if now == PresenceState::Active {
                    cause = SleepCause::None;
                    presence.hold_for_gpu(false);
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    return;
                }
            }
        }
    }
}

async fn apply(action: Action, presence: &PresenceManager, cause: &mut SleepCause) {
    match action {
        Action::None => {}
        Action::Sleep(snapshot) => {
            info!(
                target: "assistd::gpu_monitor",
                free_mb = snapshot.free_mb,
                foreign_mb = snapshot.foreign_mb,
                model_mb = snapshot.model_mb,
                "another process is crowding the model's VRAM; transitioning to Sleeping"
            );
            match presence.sleep().await {
                Ok(()) => *cause = SleepCause::Contention,
                Err(e) => {
                    warn!(target: "assistd::gpu_monitor", "sleep transition failed: {e:#}");
                    *cause = SleepCause::None;
                }
            }
            presence.hold_for_gpu(*cause == SleepCause::Contention);
        }
        Action::Wake => {
            info!(
                target: "assistd::gpu_monitor",
                "the model fits in VRAM again; auto-waking"
            );
            presence.hold_for_gpu(false);
            if let Err(e) = presence.wake().await {
                warn!(target: "assistd::gpu_monitor", "wake transition failed: {e:#}");
            }
            *cause = SleepCause::None;
        }
    }
}

fn decide(poll: Poll, cfg: &SleepConfig, cause: SleepCause) -> Action {
    match (poll.state, cause) {
        (PresenceState::Active | PresenceState::Drowsy, _) if poll.contended => {
            Action::Sleep(poll.snapshot)
        }
        (PresenceState::Sleeping, SleepCause::Contention)
            if cfg.gpu_auto_wake && poll.room_to_wake =>
        {
            Action::Wake
        }
        _ => Action::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CROWDED: VramSnapshot = VramSnapshot {
        free_mb: 150,
        model_mb: 13000,
        foreign_mb: 900,
    };

    fn poll(state: PresenceState, contended: bool, room_to_wake: bool) -> Poll {
        Poll {
            state,
            snapshot: CROWDED,
            contended,
            room_to_wake,
        }
    }

    #[test]
    fn decide_sleeps_on_contention_and_auto_wakes_only_its_own_sleep_once_it_fits() {
        let mut cfg = assistd_config::Config::default().sleep;
        cfg.gpu_auto_wake = true;
        let cases = [
            (
                "drowsy, crowded",
                poll(PresenceState::Drowsy, true, false),
                SleepCause::None,
                Action::Sleep(CROWDED),
            ),
            (
                "own sleep, still no room",
                poll(PresenceState::Sleeping, false, false),
                SleepCause::Contention,
                Action::None,
            ),
            (
                "own sleep, fits again",
                poll(PresenceState::Sleeping, false, true),
                SleepCause::Contention,
                Action::Wake,
            ),
            (
                "user sleep, fits",
                poll(PresenceState::Sleeping, false, true),
                SleepCause::None,
                Action::None,
            ),
        ];
        for (label, poll, cause, expected) in cases {
            assert_eq!(decide(poll, &cfg, cause), expected, "{label}");
        }
    }
}
