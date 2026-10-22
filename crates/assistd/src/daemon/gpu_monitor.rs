//! Sleeps the daemon when another process grows into the VRAM the model
//! needs, and optionally wakes it once the model fits again.

use std::sync::Arc;
use std::time::Duration;

use assistd_core::{Component, PresenceManager, PresenceState, SleepConfig, spawn_supervised};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{info, warn};

use super::vram::{VramLedger, VramSnapshot};

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

/// `None` when disabled in config or when NVML is unavailable.
pub(super) fn spawn_monitor(
    cfg: &SleepConfig,
    ledger: Option<Arc<VramLedger>>,
    presence: Arc<PresenceManager>,
    shutdown: watch::Receiver<bool>,
) -> Option<JoinHandle<()>> {
    let ledger = ledger?;
    info!(
        target: "assistd::gpu_monitor",
        poll_secs = cfg.gpu_poll_secs,
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
    let mut tick = tokio::time::interval(Duration::from_secs(cfg.gpu_poll_secs.get()));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    let mut sub = presence.subscribe();
    let mut cause = SleepCause::None;
    let mut consecutive_errors: u32 = 0;

    loop {
        tokio::select! {
            _ = tick.tick() => {
                let state = presence.state();
                let snapshot = match ledger.snapshot(presence.llama_pid().await) {
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
        }
        Action::Wake => {
            info!(
                target: "assistd::gpu_monitor",
                "the model fits in VRAM again; auto-waking"
            );
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
mod tests;
