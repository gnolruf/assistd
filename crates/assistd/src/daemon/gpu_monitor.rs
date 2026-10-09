//! Sleeps the daemon when a foreign process contends for VRAM, and
//! optionally wakes it again once the contender is gone.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use assistd_core::{Component, PresenceManager, PresenceState, SleepConfig, spawn_supervised};
use nvml_wrapper::{Nvml, enums::device::UsedGpuMemory};
use procfs::process::{Process, Stat};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{info, warn};

const MAX_CONSECUTIVE_FAILURES: u32 = 10;
const MAX_ANCESTOR_DEPTH: usize = 64;

/// Whether the monitor caused the current `Sleeping` state; only its own
/// sleeps are auto-woken.
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
pub(super) fn spawn_monitor(
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
    Some(spawn_supervised(
        "gpu_monitor",
        Component::GpuMonitor,
        run_monitor(nvml, cfg, presence, shutdown),
    ))
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
            if let Err(e) = presence.wake().await {
                warn!(target: "assistd::gpu_monitor", "wake transition failed: {e:#}");
            }
            *cause = SleepCause::None;
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

/// Every process holding VRAM except the daemon itself, the llama-server
/// router, and the router's model children.
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
            if is_own_process(p.pid, self_pid, llama_pid) {
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

fn is_own_process(pid: u32, self_pid: u32, llama_pid: Option<u32>) -> bool {
    pid == self_pid || llama_pid.is_some_and(|root| descends_from(pid, root, read_parent_pid))
}

/// `true` when `pid` is `root` or has `root` among its ancestors. The walk
/// ends where `parent_of` stops answering, at init, or after a bounded depth.
fn descends_from(pid: u32, root: u32, parent_of: impl Fn(u32) -> Option<u32>) -> bool {
    std::iter::successors(Some(pid), |&current| {
        parent_of(current).filter(|&parent| parent != 0 && parent != current)
    })
    .take(MAX_ANCESTOR_DEPTH)
    .any(|ancestor| ancestor == root)
}

fn read_parent_pid(pid: u32) -> Option<u32> {
    u32::try_from(read_stat(pid)?.ppid).ok()
}

fn read_comm(pid: u32) -> String {
    read_stat(pid).map_or_else(|| format!("<pid {pid}>"), |stat| stat.comm)
}

fn read_stat(pid: u32) -> Option<Stat> {
    Process::new(i32::try_from(pid).ok()?).ok()?.stat().ok()
}

#[cfg(test)]
mod tests {
    use assistd_config::defaults::nz64;

    use super::*;

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

    fn sleep(pid: u32, name: &str) -> Action {
        Action::Sleep {
            pid,
            name: name.into(),
        }
    }

    #[test]
    fn decide_applies_threshold_allowlist_and_denylist() {
        let mut c = cfg();
        c.gpu_denylist = vec!["miner".into(), "Xorg".into()];
        let cases = [
            (
                "above threshold",
                vec![sample(42, 4096, "game")],
                sleep(42, "game"),
            ),
            (
                "allowlisted above threshold",
                vec![sample(42, 4096, "firefox")],
                Action::None,
            ),
            (
                "denylist beats allowlist",
                vec![sample(42, 10, "Xorg")],
                sleep(42, "Xorg"),
            ),
            (
                "first triggering sample wins",
                vec![
                    sample(1, 100, "idle"),
                    sample(2, 8000, "firefox"),
                    sample(3, 3000, "game"),
                    sample(4, 9000, "other"),
                ],
                sleep(3, "game"),
            ),
        ];
        for (label, samples, expected) in cases {
            let a = decide(&samples, PresenceState::Active, &c, SleepCause::None);
            assert_eq!(a, expected, "{label}");
        }
    }

    #[test]
    fn decide_sleeps_from_drowsy_and_only_wakes_its_own_sleep() {
        let contender = || vec![sample(42, 4096, "game")];
        let contention = SleepCause::Contention { pid: 42 };
        let cases = [
            (
                "drowsy with contender",
                PresenceState::Drowsy,
                contender(),
                false,
                SleepCause::None,
                sleep(42, "game"),
            ),
            (
                "sleeping, contender remains",
                PresenceState::Sleeping,
                contender(),
                true,
                contention,
                Action::None,
            ),
            (
                "contention gone, auto-wake",
                PresenceState::Sleeping,
                vec![],
                true,
                contention,
                Action::Wake,
            ),
            (
                "user sleep, auto-wake",
                PresenceState::Sleeping,
                vec![],
                true,
                SleepCause::None,
                Action::None,
            ),
        ];
        for (label, state, samples, auto_wake, cause, expected) in cases {
            let mut c = cfg();
            c.gpu_auto_wake = auto_wake;
            assert_eq!(decide(&samples, state, &c, cause), expected, "{label}");
        }
    }

    #[test]
    fn descends_from_follows_the_parent_chain_to_the_router() {
        let parents: HashMap<u32, u32> = [
            (100, 1),
            (200, 100),
            (300, 200),
            (400, 1),
            (500, 501),
            (501, 500),
        ]
        .into_iter()
        .collect();
        let parent_of = |pid: u32| parents.get(&pid).copied();
        let cases = [
            ("router itself", 100, true),
            ("grandchild", 300, true),
            ("sibling of router", 400, false),
            ("parent cycle", 500, false),
        ];
        for (label, pid, expected) in cases {
            assert_eq!(descends_from(pid, 100, parent_of), expected, "{label}");
        }
    }
}
