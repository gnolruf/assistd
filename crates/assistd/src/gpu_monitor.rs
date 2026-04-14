//! Automatic GPU contention monitor.
//!
//! Polls NVML on a configurable interval for processes holding VRAM that
//! are not part of assistd (neither this process nor its llama-server
//! child). When a non-assistd process exceeds the configured VRAM
//! threshold, or its basename matches the denylist, the monitor drives
//! [`PresenceManager`] to `Sleeping` to free all VRAM held by
//! llama-server. Optional `gpu_auto_wake` brings the daemon back to
//! `Active` once the contender disappears.
//!
//! The monitor is a *driver* of [`PresenceManager`], in the same role as
//! the hotkey listener and the IPC `SetPresence` handler. It lives in
//! the binary crate (not `assistd-core`) because it pulls in the
//! optional, hardware-specific `nvml-wrapper` dependency behind the
//! `daemon` feature.
//!
//! Graceful degradation: if `Nvml::init` fails (no NVIDIA GPU, missing
//! driver), [`spawn_monitor`] returns `None` with a warning log. The
//! daemon continues to run without contention detection rather than
//! crashing — the very machines that lack NVIDIA GPUs are also the ones
//! least likely to have VRAM contention.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use assistd_core::{PresenceManager, PresenceState, SleepConfig};
use nvml_wrapper::{Nvml, enums::device::UsedGpuMemory};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{info, warn};

/// Stop polling after this many consecutive NVML failures. The daemon
/// keeps running without the monitor afterwards.
const MAX_CONSECUTIVE_FAILURES: u32 = 10;

/// Who owns the current Sleeping state, from the monitor's point of view.
///
/// `None` means "we did not cause this" — either the daemon is not
/// Sleeping, or it was put to sleep externally (user hit the hotkey, ran
/// `assistd sleep`, etc.). In that case we must never auto-wake, even if
/// a contending process later disappears: the user asked for Sleeping.
///
/// `Contention` means we put the daemon to sleep on the last poll
/// because the named PID was contending for VRAM. When that condition
/// goes away (the PID is gone or below threshold) AND `gpu_auto_wake` is
/// set, it is safe to wake again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SleepCause {
    None,
    Contention { pid: u32 },
}

/// Decision output for one poll cycle.
#[derive(Debug, PartialEq, Eq)]
enum Action {
    None,
    Sleep { pid: u32, name: String },
    Wake,
}

/// VRAM usage for a single foreign PID, aggregated across all GPUs.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ProcSample {
    pid: u32,
    used_mb: u64,
    name: String,
}

/// Pre-flight check called from daemon startup, mirroring
/// [`crate::hotkey::validate`]. Does NOT probe NVML — hardware absence
/// is handled at spawn time, not treated as a config error.
pub fn validate(cfg: &SleepConfig) -> Result<()> {
    if !cfg.gpu_monitor_enabled {
        return Ok(());
    }
    if cfg.gpu_poll_secs == 0 {
        anyhow::bail!("sleep.gpu_poll_secs must be greater than 0 when gpu_monitor_enabled");
    }
    if cfg.gpu_vram_threshold_mb == 0 {
        anyhow::bail!(
            "sleep.gpu_vram_threshold_mb must be greater than 0 when gpu_monitor_enabled"
        );
    }
    Ok(())
}

/// Spawn the GPU contention monitor. Returns `None` when the feature is
/// disabled in config or when `Nvml::init()` fails — the same idiom as
/// [`crate::hotkey::spawn_listener`]. Logs once in either case.
pub fn spawn_monitor(
    cfg: &SleepConfig,
    presence: Arc<PresenceManager>,
    shutdown: watch::Receiver<bool>,
) -> Option<JoinHandle<()>> {
    if !cfg.gpu_monitor_enabled {
        info!(
            target: "assistd::gpu_monitor",
            "sleep.gpu_monitor_enabled = false; GPU contention monitor disabled"
        );
        return None;
    }

    let nvml = match Nvml::init() {
        Ok(n) => n,
        Err(e) => {
            warn!(
                target: "assistd::gpu_monitor",
                "NVML init failed: {e}. GPU contention monitor disabled \
                 (no NVIDIA driver or no NVIDIA GPU?)"
            );
            return None;
        }
    };

    let device_count = nvml.device_count().unwrap_or(0);
    info!(
        target: "assistd::gpu_monitor",
        poll_secs = cfg.gpu_poll_secs,
        threshold_mb = cfg.gpu_vram_threshold_mb,
        auto_wake = cfg.gpu_auto_wake,
        devices = device_count,
        "GPU contention monitor enabled"
    );

    let cfg = cfg.clone();
    Some(tokio::spawn(async move {
        run_monitor(nvml, cfg, presence, shutdown).await
    }))
}

async fn run_monitor(
    nvml: Nvml,
    cfg: SleepConfig,
    presence: Arc<PresenceManager>,
    mut shutdown: watch::Receiver<bool>,
) {
    let self_pid = std::process::id();
    let mut tick = tokio::time::interval(Duration::from_secs(cfg.gpu_poll_secs));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    let mut sub = presence.subscribe();
    let mut cause = SleepCause::None;
    let mut consecutive_errors: u32 = 0;

    loop {
        tokio::select! {
            _ = tick.tick() => {
                let llama_pid = presence.llama_pid().await;
                let samples = match collect_foreign_usage(&nvml, self_pid, llama_pid) {
                    Ok(s) => {
                        consecutive_errors = 0;
                        s
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
                let action = decide(&samples, presence.state(), &cfg, cause);
                apply(action, &presence, &mut cause).await;
            }
            _ = sub.changed() => {
                // External actor (user hotkey, `assistd wake`, IPC) changed
                // presence. If we thought we owned a contention sleep and the
                // state is now Active, release ownership so we don't later
                // "auto-wake" something we didn't actually sleep.
                let now = *sub.borrow_and_update();
                if now == PresenceState::Active
                    && matches!(cause, SleepCause::Contention { .. })
                {
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
        Action::Sleep { pid, name } => {
            info!(
                target: "assistd::gpu_monitor",
                pid,
                name = %name,
                "GPU contention detected; transitioning to Sleeping"
            );
            match presence.sleep().await {
                Ok(()) => *cause = SleepCause::Contention { pid },
                Err(e) => {
                    warn!(target: "assistd::gpu_monitor", "sleep transition failed: {e:#}");
                    *cause = SleepCause::None;
                }
            }
        }
        Action::Wake => {
            info!(
                target: "assistd::gpu_monitor",
                "GPU contention cleared; auto-waking"
            );
            match presence.wake().await {
                Ok(()) => *cause = SleepCause::None,
                Err(e) => {
                    warn!(target: "assistd::gpu_monitor", "wake transition failed: {e:#}");
                    *cause = SleepCause::None;
                }
            }
        }
    }
}

fn decide(
    samples: &[ProcSample],
    state: PresenceState,
    cfg: &SleepConfig,
    cause: SleepCause,
) -> Action {
    let trigger = samples.iter().find(|s| {
        let denied = cfg.gpu_denylist.iter().any(|n| n == &s.name);
        let allowed = cfg.gpu_allowlist.iter().any(|n| n == &s.name);
        denied || (s.used_mb >= cfg.gpu_vram_threshold_mb && !allowed)
    });

    match (state, trigger, cause) {
        (PresenceState::Active | PresenceState::Drowsy, Some(t), _) => Action::Sleep {
            pid: t.pid,
            name: t.name.clone(),
        },
        (PresenceState::Sleeping, None, SleepCause::Contention { .. }) if cfg.gpu_auto_wake => {
            Action::Wake
        }
        _ => Action::None,
    }
}

fn collect_foreign_usage(
    nvml: &Nvml,
    self_pid: u32,
    llama_pid: Option<u32>,
) -> Result<Vec<ProcSample>> {
    let device_count = nvml.device_count()?;
    // Per-PID total VRAM summed across devices. Within a single device,
    // a PID with both a compute and a graphics context may appear twice
    // reporting the same allocation — take the max on that device to
    // avoid double-counting, then sum those per-device maxes.
    let mut total_by_pid: HashMap<u32, u64> = HashMap::new();
    for idx in 0..device_count {
        let device = nvml.device_by_index(idx)?;
        let mut per_device: HashMap<u32, u64> = HashMap::new();
        let compute = device.running_compute_processes().unwrap_or_default();
        let graphics = device.running_graphics_processes().unwrap_or_default();
        for p in compute.into_iter().chain(graphics) {
            if p.pid == self_pid || Some(p.pid) == llama_pid {
                continue;
            }
            let bytes = match p.used_gpu_memory {
                UsedGpuMemory::Used(n) => n,
                UsedGpuMemory::Unavailable => 0,
            };
            let entry = per_device.entry(p.pid).or_default();
            *entry = (*entry).max(bytes);
        }
        for (pid, bytes) in per_device {
            *total_by_pid.entry(pid).or_default() += bytes;
        }
    }

    Ok(total_by_pid
        .into_iter()
        .map(|(pid, bytes)| ProcSample {
            pid,
            used_mb: bytes / (1024 * 1024),
            name: read_comm(pid),
        })
        .collect())
}

fn read_comm(pid: u32) -> String {
    std::fs::read_to_string(format!("/proc/{pid}/comm"))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| format!("<pid {pid}>"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SleepConfig {
        SleepConfig {
            idle_to_drowsy_mins: 30,
            idle_to_sleep_mins: 120,
            suspend: false,
            gpu_monitor_enabled: true,
            gpu_poll_secs: 5,
            gpu_vram_threshold_mb: 2048,
            gpu_auto_wake: false,
            gpu_allowlist: vec!["Xorg".into(), "firefox".into()],
            gpu_denylist: Vec::new(),
        }
    }

    fn sample(pid: u32, used_mb: u64, name: &str) -> ProcSample {
        ProcSample {
            pid,
            used_mb,
            name: name.into(),
        }
    }

    // --- validate -----------------------------------------------------

    #[test]
    fn validate_disabled_is_ok() {
        let mut c = cfg();
        c.gpu_monitor_enabled = false;
        c.gpu_poll_secs = 0;
        c.gpu_vram_threshold_mb = 0;
        validate(&c).expect("disabled monitor tolerates zero fields");
    }

    #[test]
    fn validate_enabled_rejects_zero_poll() {
        let mut c = cfg();
        c.gpu_poll_secs = 0;
        let err = validate(&c).unwrap_err();
        assert!(err.to_string().contains("gpu_poll_secs"));
    }

    #[test]
    fn validate_enabled_rejects_zero_threshold() {
        let mut c = cfg();
        c.gpu_vram_threshold_mb = 0;
        let err = validate(&c).unwrap_err();
        assert!(err.to_string().contains("gpu_vram_threshold_mb"));
    }

    // --- decide: Active/Drowsy + trigger → Sleep ---------------------

    #[test]
    fn active_with_foreign_process_above_threshold_sleeps() {
        let s = [sample(42, 4096, "game")];
        let a = decide(&s, PresenceState::Active, &cfg(), SleepCause::None);
        assert_eq!(
            a,
            Action::Sleep {
                pid: 42,
                name: "game".into()
            }
        );
    }

    #[test]
    fn drowsy_with_foreign_process_above_threshold_sleeps() {
        let s = [sample(42, 4096, "game")];
        let a = decide(&s, PresenceState::Drowsy, &cfg(), SleepCause::None);
        assert!(matches!(a, Action::Sleep { .. }));
    }

    // --- decide: no trigger → None -----------------------------------

    #[test]
    fn active_below_threshold_no_action() {
        let s = [sample(42, 500, "something")];
        let a = decide(&s, PresenceState::Active, &cfg(), SleepCause::None);
        assert_eq!(a, Action::None);
    }

    #[test]
    fn active_no_processes_no_action() {
        let a = decide(&[], PresenceState::Active, &cfg(), SleepCause::None);
        assert_eq!(a, Action::None);
    }

    // --- decide: Sleeping arms ---------------------------------------

    #[test]
    fn sleeping_with_contender_stays_sleeping() {
        let s = [sample(42, 4096, "game")];
        let a = decide(
            &s,
            PresenceState::Sleeping,
            &cfg(),
            SleepCause::Contention { pid: 42 },
        );
        assert_eq!(a, Action::None);
    }

    #[test]
    fn sleeping_contention_gone_with_auto_wake_wakes() {
        let mut c = cfg();
        c.gpu_auto_wake = true;
        let a = decide(
            &[],
            PresenceState::Sleeping,
            &c,
            SleepCause::Contention { pid: 42 },
        );
        assert_eq!(a, Action::Wake);
    }

    #[test]
    fn sleeping_contention_gone_without_auto_wake_no_action() {
        let a = decide(
            &[],
            PresenceState::Sleeping,
            &cfg(),
            SleepCause::Contention { pid: 42 },
        );
        assert_eq!(a, Action::None);
    }

    #[test]
    fn sleeping_from_user_never_auto_wakes_even_with_auto_wake_true() {
        let mut c = cfg();
        c.gpu_auto_wake = true;
        let a = decide(&[], PresenceState::Sleeping, &c, SleepCause::None);
        assert_eq!(a, Action::None);
    }

    #[test]
    fn sleeping_from_user_with_contender_no_action() {
        // User-initiated sleep while a contender happens to be running.
        // We never re-sleep something that's already Sleeping, and we
        // never auto-wake because we don't own the sleep.
        let s = [sample(42, 4096, "game")];
        let a = decide(&s, PresenceState::Sleeping, &cfg(), SleepCause::None);
        assert_eq!(a, Action::None);
    }

    // --- allowlist / denylist ----------------------------------------

    #[test]
    fn allowlist_suppresses_threshold_trigger() {
        // firefox is allowlisted even though it's over threshold
        let s = [sample(42, 4096, "firefox")];
        let a = decide(&s, PresenceState::Active, &cfg(), SleepCause::None);
        assert_eq!(a, Action::None);
    }

    #[test]
    fn denylist_fires_below_threshold() {
        let mut c = cfg();
        c.gpu_denylist = vec!["miner".into()];
        let s = [sample(42, 10, "miner")];
        let a = decide(&s, PresenceState::Active, &c, SleepCause::None);
        assert_eq!(
            a,
            Action::Sleep {
                pid: 42,
                name: "miner".into()
            }
        );
    }

    #[test]
    fn denylist_wins_over_allowlist_when_both_match() {
        let mut c = cfg();
        c.gpu_allowlist = vec!["firefox".into()];
        c.gpu_denylist = vec!["firefox".into()];
        let s = [sample(42, 10, "firefox")];
        let a = decide(&s, PresenceState::Active, &c, SleepCause::None);
        assert!(matches!(a, Action::Sleep { .. }));
    }

    #[test]
    fn allowlist_entry_below_threshold_no_action() {
        let s = [sample(42, 500, "firefox")];
        let a = decide(&s, PresenceState::Active, &cfg(), SleepCause::None);
        assert_eq!(a, Action::None);
    }

    // --- multi-sample ------------------------------------------------

    #[test]
    fn multi_sample_picks_a_triggering_entry() {
        // Mix of allowed, below-threshold, and triggering — trigger wins.
        let s = [
            sample(1, 100, "idle"),
            sample(2, 8000, "firefox"), // allowlisted, not a trigger
            sample(3, 3000, "game"),    // triggers
        ];
        let a = decide(&s, PresenceState::Active, &cfg(), SleepCause::None);
        assert!(matches!(
            a,
            Action::Sleep { pid: 3, .. } | Action::Sleep { .. }
        ));
    }

    #[test]
    fn read_comm_unknown_pid_does_not_panic() {
        // An implausibly-large PID that cannot exist should fall back to
        // the placeholder form, not panic.
        let name = read_comm(u32::MAX);
        assert!(name.contains(&u32::MAX.to_string()));
    }

    // Live NVML smoke test. Skipped by default because it requires an
    // actual NVIDIA driver; run explicitly with:
    //   cargo test -p assistd-cli --features daemon -- --ignored live_nvml
    #[test]
    #[ignore = "requires NVIDIA driver"]
    fn live_nvml_collect_foreign_usage_does_not_panic() {
        let nvml = Nvml::init().expect("Nvml::init should succeed with NVIDIA driver present");
        let samples = collect_foreign_usage(&nvml, std::process::id(), None)
            .expect("collect_foreign_usage should succeed on a working system");
        for s in &samples {
            assert_ne!(s.pid, std::process::id(), "self pid must be filtered out");
        }
    }
}
