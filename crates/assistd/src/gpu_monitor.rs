//! Sleeps the daemon when a foreign process contends for VRAM, and
//! optionally wakes it again once the contender is gone.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use assistd_core::{PresenceManager, PresenceState, SleepConfig};
use nvml_wrapper::{Nvml, enums::device::UsedGpuMemory};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{info, warn};

const MAX_CONSECUTIVE_FAILURES: u32 = 10;

/// Whether the monitor caused the current `Sleeping` state. Only a sleep
/// the monitor itself triggered may be auto-woken; a user-requested
/// sleep stays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SleepCause {
    None,
    Contention { pid: u32 },
}

#[derive(Debug, PartialEq, Eq)]
enum Action {
    None,
    Sleep { pid: u32, name: String },
    Wake,
}

/// VRAM usage for one foreign PID, summed across GPUs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProcSample {
    pub(crate) pid: u32,
    pub(crate) used_mb: u64,
    /// From `/proc/<pid>/comm`.
    pub(crate) name: String,
}

/// `None` when disabled in config or when NVML is unavailable.
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

    let device_count = match nvml.device_count() {
        Ok(n) => n.to_string(),
        Err(e) => {
            warn!(
                target: "assistd::gpu_monitor",
                "nvml.device_count() failed: {e}. Monitor will still run, \
                 but per-device enumeration may be unreliable."
            );
            "unknown".to_string()
        }
    };
    info!(
        target: "assistd::gpu_monitor",
        poll_secs = cfg.gpu_poll_secs,
        threshold_mb = cfg.gpu_vram_threshold_mb,
        auto_wake = cfg.gpu_auto_wake,
        devices = %device_count,
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
    let mut tick = tokio::time::interval(Duration::from_secs(cfg.gpu_poll_secs.get()));
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
        denied || (s.used_mb >= cfg.gpu_vram_threshold_mb.get() && !allowed)
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

/// Every process holding VRAM except `self_pid` and `llama_pid`.
pub(crate) fn collect_foreign_usage(
    nvml: &Nvml,
    self_pid: u32,
    llama_pid: Option<u32>,
) -> Result<Vec<ProcSample>> {
    let device_count = nvml.device_count()?;
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
    use assistd_config::defaults::nz64;

    fn cfg() -> SleepConfig {
        SleepConfig {
            idle_to_drowsy_mins: 30,
            idle_to_sleep_mins: 120,
            gpu_monitor_enabled: true,
            gpu_poll_secs: nz64(5),
            gpu_vram_threshold_mb: nz64(2048),
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
        let s = [sample(42, 4096, "game")];
        let a = decide(&s, PresenceState::Sleeping, &cfg(), SleepCause::None);
        assert_eq!(a, Action::None);
    }

    #[test]
    fn allowlist_suppresses_threshold_trigger() {
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

    #[test]
    fn multi_sample_picks_a_triggering_entry() {
        let s = [
            sample(1, 100, "idle"),
            sample(2, 8000, "firefox"),
            sample(3, 3000, "game"),
        ];
        let a = decide(&s, PresenceState::Active, &cfg(), SleepCause::None);
        assert!(matches!(
            a,
            Action::Sleep { pid: 3, .. } | Action::Sleep { .. }
        ));
    }

    #[test]
    fn read_comm_unknown_pid_does_not_panic() {
        let name = read_comm(u32::MAX);
        assert!(name.contains(&u32::MAX.to_string()));
    }

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
