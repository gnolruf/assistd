//! Automatic idle-based sleep monitor.
//!
//! Polls `PresenceManager::idle_duration()` on a fixed interval and
//! drives `drowse()` / `sleep()` when the configured thresholds are
//! crossed. Any user interaction (query, TUI submit, CLI presence
//! command, hotkey) resets `PresenceManager::last_activity` and
//! naturally defers the next transition.
//!
//! Like `gpu_monitor`, this module is a *driver* of `PresenceManager`.
//! It calls `presence.drowse()` / `presence.sleep()` directly rather
//! than going through `set_presence`, which means automatic transitions
//! do not themselves reset the idle timer — a crucial invariant so that
//! the monitor can make forward progress through Active → Drowsy →
//! Sleeping.
//!
//! Disabled when both `sleep.idle_to_drowsy_mins` and
//! `sleep.idle_to_sleep_mins` are 0; returns `None` from
//! [`spawn_monitor`] with an info log.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail};
use assistd_core::{PresenceManager, PresenceState, SleepConfig};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{info, warn};

/// How often to re-check idle duration against thresholds. The
/// thresholds themselves are in minutes, so sub-minute granularity is
/// irrelevant — 10 s keeps the TUI countdown feeling live while
/// imposing near-zero cost at idle (one std-mutex read + compare).
const POLL_INTERVAL_SECS: u64 = 10;

/// Decision output for one poll cycle. Pure and trivially unit-testable.
#[derive(Debug, PartialEq, Eq)]
enum Action {
    None,
    Drowse,
    Sleep,
}

/// Pre-flight check called from daemon startup, mirroring
/// [`crate::gpu_monitor::validate`]. `Config::validate` already catches
/// this; we duplicate the rule here so the daemon's fail-fast path
/// stays symmetric across subsystems.
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

/// Spawn the idle monitor. Returns `None` when both thresholds are 0
/// (feature fully disabled).
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
    Some(tokio::spawn(async move {
        run_monitor(cfg, presence, shutdown).await
    }))
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
                // External transition (hotkey, query auto-wake, CLI
                // command, GPU monitor). Drop the old reading; the next
                // tick will re-evaluate against the new state.
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
                // Cascade: even if idle already exceeds the sleep
                // threshold, step through Drowsy on this tick so the
                // model weights are unloaded cleanly before the server
                // is torn down on the next tick.
                Action::Drowse
            } else if cfg.idle_to_drowsy_mins == 0
                && cfg.idle_to_sleep_mins > 0
                && idle >= sleep_threshold
            {
                // Drowsy step explicitly disabled — skip directly to Sleep.
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

    // --- validate -----------------------------------------------------

    #[test]
    fn validate_both_zero_is_ok() {
        validate(&cfg(0, 0)).expect("both zero disables the monitor");
    }

    #[test]
    fn validate_one_zero_is_ok() {
        validate(&cfg(0, 120)).expect("only sleep configured is valid");
        validate(&cfg(30, 0)).expect("only drowsy configured is valid");
    }

    #[test]
    fn validate_sleep_gt_drowsy_is_ok() {
        validate(&cfg(30, 120)).expect("standard ordering is valid");
    }

    #[test]
    fn validate_sleep_lt_drowsy_errors() {
        let err = validate(&cfg(60, 30)).unwrap_err();
        assert!(err.to_string().contains("idle_to_sleep_mins"));
    }

    #[test]
    fn validate_sleep_eq_drowsy_errors() {
        let err = validate(&cfg(60, 60)).unwrap_err();
        assert!(err.to_string().contains("idle_to_sleep_mins"));
    }

    // --- decide: Active ----------------------------------------------

    #[test]
    fn active_before_drowsy_threshold_no_action() {
        let a = decide(
            PresenceState::Active,
            Duration::from_secs(10 * 60),
            &cfg(30, 120),
        );
        assert_eq!(a, Action::None);
    }

    #[test]
    fn active_at_drowsy_threshold_drowses() {
        let a = decide(
            PresenceState::Active,
            Duration::from_secs(30 * 60),
            &cfg(30, 120),
        );
        assert_eq!(a, Action::Drowse);
    }

    #[test]
    fn active_past_drowsy_but_below_sleep_still_drowses() {
        let a = decide(
            PresenceState::Active,
            Duration::from_secs(90 * 60),
            &cfg(30, 120),
        );
        assert_eq!(a, Action::Drowse);
    }

    #[test]
    fn active_past_sleep_threshold_still_drowses_first_cascade() {
        // Cascade semantics: emit Drowse here, next tick will see
        // Drowsy + idle >= sleep and emit Sleep. This preserves the
        // ordered teardown (weights unload → server stop) even when
        // idle had already exceeded both thresholds at startup.
        let a = decide(
            PresenceState::Active,
            Duration::from_secs(200 * 60),
            &cfg(30, 120),
        );
        assert_eq!(a, Action::Drowse);
    }

    #[test]
    fn active_with_drowsy_disabled_before_sleep_no_action() {
        let a = decide(
            PresenceState::Active,
            Duration::from_secs(60 * 60),
            &cfg(0, 120),
        );
        assert_eq!(a, Action::None);
    }

    #[test]
    fn active_with_drowsy_disabled_past_sleep_sleeps_directly() {
        let a = decide(
            PresenceState::Active,
            Duration::from_secs(130 * 60),
            &cfg(0, 120),
        );
        assert_eq!(a, Action::Sleep);
    }

    #[test]
    fn active_with_sleep_disabled_still_drowses() {
        let a = decide(
            PresenceState::Active,
            Duration::from_secs(30 * 60),
            &cfg(30, 0),
        );
        assert_eq!(a, Action::Drowse);
    }

    // --- decide: Drowsy ----------------------------------------------

    #[test]
    fn drowsy_before_sleep_threshold_no_action() {
        let a = decide(
            PresenceState::Drowsy,
            Duration::from_secs(60 * 60),
            &cfg(30, 120),
        );
        assert_eq!(a, Action::None);
    }

    #[test]
    fn drowsy_at_sleep_threshold_sleeps() {
        let a = decide(
            PresenceState::Drowsy,
            Duration::from_secs(120 * 60),
            &cfg(30, 120),
        );
        assert_eq!(a, Action::Sleep);
    }

    #[test]
    fn drowsy_with_sleep_disabled_no_action() {
        let a = decide(
            PresenceState::Drowsy,
            Duration::from_secs(200 * 60),
            &cfg(30, 0),
        );
        assert_eq!(a, Action::None);
    }

    // --- decide: Sleeping --------------------------------------------

    #[test]
    fn sleeping_never_acts() {
        let a = decide(
            PresenceState::Sleeping,
            Duration::from_secs(500 * 60),
            &cfg(30, 120),
        );
        assert_eq!(a, Action::None);
    }

    #[test]
    fn both_disabled_returns_none_from_any_state() {
        let disabled = cfg(0, 0);
        for state in [
            PresenceState::Active,
            PresenceState::Drowsy,
            PresenceState::Sleeping,
        ] {
            let a = decide(state, Duration::from_secs(10_000 * 60), &disabled);
            assert_eq!(a, Action::None);
        }
    }
}
