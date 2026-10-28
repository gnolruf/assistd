//! Sleep and idle policy configuration.

use std::num::NonZeroU64;

use serde::{Deserialize, Serialize};

use crate::defaults::{
    DEFAULT_GPU_HEADROOM_MB, DEFAULT_GPU_MONITOR_ENABLED, DEFAULT_GPU_POLL_SECS,
    DEFAULT_IDLE_TO_DROWSY_MINS, DEFAULT_IDLE_TO_SLEEP_MINS,
};
use crate::settings::{DescribeSettings, describe_fields};

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
    /// Poll NVML: sleep when another process crowds the model's VRAM, and
    /// refuse a wake the model would not fit. Inert without an NVIDIA GPU.
    pub gpu_monitor_enabled: bool,
    /// NVML poll interval in seconds.
    pub gpu_poll_secs: NonZeroU64,
    /// Free VRAM, in MiB, to keep beside the model. Another process growing
    /// while less is free puts the daemon to sleep, and a wake that would
    /// leave less is refused. `0` keeps no margin and never sleeps for it.
    pub gpu_headroom_mb: u64,
    /// After a sleep for contention, return to `Active` once the model fits
    /// again with the headroom to spare.
    pub gpu_auto_wake: bool,
}

impl DescribeSettings for SleepConfig {
    describe_fields! {
        idle_to_drowsy_mins => (
            "drowse after idle (min)",
            "Idle minutes before the model is unloaded; 0 disables"
        ),
        idle_to_sleep_mins => (
            "sleep after idle (min)",
            "Idle minutes before llama-server stops; 0 disables"
        ),
        gpu_monitor_enabled => (
            "enabled",
            "Sleep when other processes crowd the model's VRAM (NVIDIA only)"
        ),
        gpu_poll_secs => ("check interval (s)", "Seconds between VRAM checks"),
        gpu_headroom_mb => (
            "free VRAM to keep (MiB)",
            "Free VRAM to keep beside the model; 0 keeps no margin"
        ),
        gpu_auto_wake => (
            "wake when VRAM frees up",
            "Wake again once the model fits with headroom to spare"
        ),
    }
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            idle_to_drowsy_mins: DEFAULT_IDLE_TO_DROWSY_MINS,
            idle_to_sleep_mins: DEFAULT_IDLE_TO_SLEEP_MINS,
            gpu_monitor_enabled: DEFAULT_GPU_MONITOR_ENABLED,
            gpu_poll_secs: DEFAULT_GPU_POLL_SECS,
            gpu_headroom_mb: DEFAULT_GPU_HEADROOM_MB,
            gpu_auto_wake: false,
        }
    }
}
