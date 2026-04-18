use crate::defaults::{
    DEFAULT_GPU_MONITOR_ENABLED, DEFAULT_GPU_POLL_SECS, DEFAULT_GPU_VRAM_THRESHOLD_MB,
    DEFAULT_IDLE_TO_DROWSY_MINS, DEFAULT_IDLE_TO_SLEEP_MINS, default_gpu_allowlist,
};
use serde::{Deserialize, Serialize};

/// Sleep/idle policy settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SleepConfig {
    /// Minutes of no user interaction after which the daemon drops
    /// `Active → Drowsy` (model weights unloaded, server stays alive).
    /// `0` disables this transition. Independent of `idle_to_sleep_mins`.
    #[serde(default = "default_idle_to_drowsy_mins")]
    pub idle_to_drowsy_mins: u64,
    /// Minutes of no user interaction after which the daemon drops to
    /// `Sleeping` (llama-server stopped, all VRAM freed). Must be greater
    /// than `idle_to_drowsy_mins` when both are non-zero. `0` disables
    /// this transition.
    #[serde(default = "default_idle_to_sleep_mins")]
    pub idle_to_sleep_mins: u64,
    /// Whether to suspend the machine (true) or just deactivate (false).
    pub suspend: bool,

    /// Enable the automatic GPU contention monitor. When true, a background
    /// task polls NVML for other processes using VRAM and transitions the
    /// daemon to Sleeping when a configurable threshold is exceeded.
    #[serde(default = "default_gpu_monitor_enabled")]
    pub gpu_monitor_enabled: bool,
    /// NVML poll interval in seconds.
    #[serde(default = "default_gpu_poll_secs")]
    pub gpu_poll_secs: u64,
    /// Per-process VRAM threshold in MiB. A non-assistd process holding at
    /// least this much VRAM triggers a transition to Sleeping. 2048 = 2 GiB.
    #[serde(default = "default_gpu_vram_threshold_mb")]
    pub gpu_vram_threshold_mb: u64,
    /// Automatically transition back to Active when the contending process
    /// exits.
    #[serde(default)]
    pub gpu_auto_wake: bool,
    /// Process basenames (matched against `/proc/<pid>/comm`, kernel-truncated
    /// at 16 bytes) that never trigger contention sleep, even above the
    /// threshold.
    #[serde(default = "default_gpu_allowlist")]
    pub gpu_allowlist: Vec<String>,
    /// Process basenames that always trigger sleep when present, regardless
    /// of their VRAM usage.
    #[serde(default)]
    pub gpu_denylist: Vec<String>,
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            idle_to_drowsy_mins: DEFAULT_IDLE_TO_DROWSY_MINS,
            idle_to_sleep_mins: DEFAULT_IDLE_TO_SLEEP_MINS,
            suspend: false,
            gpu_monitor_enabled: DEFAULT_GPU_MONITOR_ENABLED,
            gpu_poll_secs: DEFAULT_GPU_POLL_SECS,
            gpu_vram_threshold_mb: DEFAULT_GPU_VRAM_THRESHOLD_MB,
            gpu_auto_wake: false,
            gpu_allowlist: default_gpu_allowlist(),
            gpu_denylist: Vec::new(),
        }
    }
}

fn default_idle_to_drowsy_mins() -> u64 {
    DEFAULT_IDLE_TO_DROWSY_MINS
}

fn default_idle_to_sleep_mins() -> u64 {
    DEFAULT_IDLE_TO_SLEEP_MINS
}

fn default_gpu_monitor_enabled() -> bool {
    DEFAULT_GPU_MONITOR_ENABLED
}

fn default_gpu_poll_secs() -> u64 {
    DEFAULT_GPU_POLL_SECS
}

fn default_gpu_vram_threshold_mb() -> u64 {
    DEFAULT_GPU_VRAM_THRESHOLD_MB
}
