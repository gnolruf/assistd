use std::num::NonZeroU64;

use serde::{Deserialize, Serialize};

use crate::defaults::{
    DEFAULT_GPU_MONITOR_ENABLED, DEFAULT_GPU_POLL_SECS, DEFAULT_GPU_VRAM_THRESHOLD_MB,
    DEFAULT_IDLE_TO_DROWSY_MINS, DEFAULT_IDLE_TO_SLEEP_MINS, default_gpu_allowlist,
};

/// Sleep/idle policy settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SleepConfig {
    /// Idle minutes before `Active → Drowsy` (weights unloaded, server
    /// kept). `0` disables.
    pub idle_to_drowsy_mins: u64,
    /// Idle minutes before `Sleeping` (llama-server stopped). `0` disables;
    /// otherwise it must exceed a non-zero `idle_to_drowsy_mins`.
    pub idle_to_sleep_mins: u64,
    /// Poll NVML and sleep when another process claims VRAM. Inert without
    /// an NVIDIA GPU.
    pub gpu_monitor_enabled: bool,
    /// NVML poll interval in seconds.
    pub gpu_poll_secs: NonZeroU64,
    /// VRAM, in MiB, at which another process triggers sleep.
    pub gpu_vram_threshold_mb: NonZeroU64,
    /// Return to `Active` when the contending process exits.
    pub gpu_auto_wake: bool,
    /// Process names that never trigger sleep, matched against
    /// `/proc/<pid>/comm` (which the kernel truncates at 16 bytes).
    pub gpu_allowlist: Vec<String>,
    /// Process names that trigger sleep regardless of VRAM use.
    pub gpu_denylist: Vec<String>,
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            idle_to_drowsy_mins: DEFAULT_IDLE_TO_DROWSY_MINS,
            idle_to_sleep_mins: DEFAULT_IDLE_TO_SLEEP_MINS,
            gpu_monitor_enabled: DEFAULT_GPU_MONITOR_ENABLED,
            gpu_poll_secs: DEFAULT_GPU_POLL_SECS,
            gpu_vram_threshold_mb: DEFAULT_GPU_VRAM_THRESHOLD_MB,
            gpu_auto_wake: false,
            gpu_allowlist: default_gpu_allowlist(),
            gpu_denylist: Vec::new(),
        }
    }
}
