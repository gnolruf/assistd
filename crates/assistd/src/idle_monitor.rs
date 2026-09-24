//! Drowses and sleeps the daemon after configured idle periods.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail};
use assistd_core::{Component, PresenceManager, PresenceState, SleepConfig, spawn_supervised};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{info, warn};

const POLL_INTERVAL_SECS: u64 = 10;

#[derive(Debug, PartialEq, Eq)]
enum Action {
    None,
    Drowse,
    Sleep,
}

/// Reject a sleep threshold at or below the drowsy threshold.
pub fn validate(cfg: &SleepConfig) -> Result<()> {
    if cfg.idle_to_drowsy_mins > 0
        && cfg.idle_to_sleep_mins > 0
        && cfg.idle_to_sleep_mins <= cfg.idle_to_drowsy_mins
    {
        bail!(
            "sleep.idle_to_sleep_mins must be greater than sleep.idle_to_drowsy_mins \
             (set either to 0 to disable that transition)"
        );
    }
    Ok(())
}

/// Spawn the idle monitor. `None` when both thresholds are 0.
pub fn spawn_monitor(
    cfg: &SleepConfig,
    presence: Arc<PresenceManager>,
    shutdown: watch::Receiver<bool>,
) -> Option<JoinHandle<()>> {
    if cfg.idle_to_drowsy_mins == 0 && cfg.idle_to_sleep_mins == 0 {
        info!(
            target: "assistd::idle_monitor",
            "sleep.idle_to_drowsy_mins = sleep.idle_to_sleep_mins = 0; idle monitor disabled"
        );
        return None;
    }
    info!(
        target: "assistd::idle_monitor",
        drowsy_mins = cfg.idle_to_drowsy_mins,
        sleep_mins = cfg.idle_to_sleep_mins,
        poll_secs = POLL_INTERVAL_SECS,
        "idle monitor enabled"
    );
    let cfg = cfg.clone();
    Some(spawn_supervised(
        "idle_monitor",
        Component::IdleMonitor,
        run_monitor(cfg, presence, shutdown),
    ))
}

async fn run_monitor(
    cfg: SleepConfig,
    presence: Arc<PresenceManager>,
    mut shutdown: watch::Receiver<bool>,
) {
    let mut tick = tokio::time::interval(Duration::from_secs(POLL_INTERVAL_SECS));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut sub = presence.subscribe();

    loop {
        tokio::select! {
            _ = tick.tick() => {
                let action = decide(presence.state(), presence.idle_duration(), &cfg);
                apply(action, &presence).await;
            }
            _ = sub.changed() => {
                let _ = *sub.borrow_and_update();
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    return;
                }
            }
        }
    }
}

/// Calls `drowse()` / `sleep()` directly rather than `set_presence`, so
/// an automatic transition does not reset the idle timer and the monitor
/// can progress Active → Drowsy → Sleeping.
async fn apply(action: Action, presence: &PresenceManager) {
    match action {
        Action::None => {}
        Action::Drowse => {
            info!(target: "assistd::idle_monitor", "idle threshold reached; drowsing");
            if let Err(e) = presence.drowse().await {
                warn!(target: "assistd::idle_monitor", "drowse failed: {e:#}");
            }
        }
        Action::Sleep => {
            info!(target: "assistd::idle_monitor", "idle threshold reached; sleeping");
            if let Err(e) = presence.sleep().await {
                warn!(target: "assistd::idle_monitor", "sleep failed: {e:#}");
            }
        }
    }
}

fn decide(state: PresenceState, idle: Duration, cfg: &SleepConfig) -> Action {
    let drowsy_threshold = Duration::from_secs(cfg.idle_to_drowsy_mins * 60);
    let sleep_threshold = Duration::from_secs(cfg.idle_to_sleep_mins * 60);
    match state {
        PresenceState::Active => {
            if cfg.idle_to_drowsy_mins > 0 && idle >= drowsy_threshold {
                Action::Drowse
            } else if cfg.idle_to_drowsy_mins == 0
                && cfg.idle_to_sleep_mins > 0
                && idle >= sleep_threshold
            {
                Action::Sleep
            } else {
                Action::None
            }
        }
        PresenceState::Drowsy => {
            if cfg.idle_to_sleep_mins > 0 && idle >= sleep_threshold {
                Action::Sleep
            } else {
                Action::None
            }
        }
        PresenceState::Sleeping => Action::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(drowsy: u64, sleep: u64) -> SleepConfig {
        let mut c = assistd_core::Config::default().sleep;
        c.idle_to_drowsy_mins = drowsy;
        c.idle_to_sleep_mins = sleep;
        c
    }

    #[test]
    fn validate_accepts_disabled_or_ordered_thresholds() {
        for (drowsy, sleep) in [(0, 0), (0, 120), (30, 0), (30, 120)] {
            validate(&cfg(drowsy, sleep))
                .unwrap_or_else(|e| panic!("({drowsy}, {sleep}) rejected: {e}"));
        }
    }

    #[test]
    fn validate_rejects_sleep_at_or_below_drowsy() {
        for (drowsy, sleep) in [(60, 30), (60, 60)] {
            let err = validate(&cfg(drowsy, sleep)).expect_err("must be rejected");
            assert!(
                err.to_string()
                    .contains("idle_to_sleep_mins must be greater than"),
                "({drowsy}, {sleep}): {err}"
            );
        }
    }

    #[test]
    fn decide_by_state_idle_time_and_thresholds() {
        use PresenceState::{Active, Drowsy, Sleeping};
        let cases = [
            ("active before drowsy", Active, 10, (30, 120), Action::None),
            ("active at drowsy", Active, 30, (30, 120), Action::Drowse),
            (
                "active between thresholds",
                Active,
                90,
                (30, 120),
                Action::Drowse,
            ),
            (
                "active past sleep cascades via drowsy",
                Active,
                200,
                (30, 120),
                Action::Drowse,
            ),
            (
                "active, drowsy disabled, before sleep",
                Active,
                60,
                (0, 120),
                Action::None,
            ),
            (
                "active, drowsy disabled, past sleep",
                Active,
                130,
                (0, 120),
                Action::Sleep,
            ),
            (
                "active, sleep disabled",
                Active,
                30,
                (30, 0),
                Action::Drowse,
            ),
            ("drowsy before sleep", Drowsy, 60, (30, 120), Action::None),
            ("drowsy at sleep", Drowsy, 120, (30, 120), Action::Sleep),
            ("drowsy, sleep disabled", Drowsy, 200, (30, 0), Action::None),
            ("sleeping", Sleeping, 500, (30, 120), Action::None),
            (
                "active, both disabled",
                Active,
                10_000,
                (0, 0),
                Action::None,
            ),
            (
                "drowsy, both disabled",
                Drowsy,
                10_000,
                (0, 0),
                Action::None,
            ),
        ];
        for (label, state, idle_mins, (drowsy, sleep), expected) in cases {
            let idle = Duration::from_secs(idle_mins * 60);
            assert_eq!(
                decide(state, idle, &cfg(drowsy, sleep)),
                expected,
                "{label}"
            );
        }
    }
}
