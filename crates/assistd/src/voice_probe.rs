//! Binary-side [`assistd_voice::BusyProbe`] implementation.
//!
//! Bridges the voice crate's trait to the daemon's [`PresenceManager`]
//! (for LLM-stream awareness) and an optional NVML handle (for
//! foreign-process detection, reusing the same per-PID VRAM scan
//! [`crate::gpu_monitor`] uses to drive auto-sleep). When NVML init
//! fails the probe collapses to "no foreign busy" — matching the
//! graceful-degradation idiom in `gpu_monitor::spawn_monitor`.

use std::sync::Arc;
use std::time::Duration;

use assistd_core::PresenceManager;
use assistd_voice::BusyProbe;
use async_trait::async_trait;
use nvml_wrapper::Nvml;

use crate::gpu_monitor;

/// Minimum per-process VRAM (MiB) to treat a foreign PID as actually
/// contending for the GPU. Set low enough to catch a second model
/// runner but above the noise floor of a desktop compositor with a
/// backing shader on a secondary display.
const FOREIGN_VRAM_THRESHOLD_MB: u64 = 100;

pub struct PresenceGpuProbe {
    presence: Arc<PresenceManager>,
    nvml: Option<Arc<Nvml>>,
    self_pid: u32,
}

impl PresenceGpuProbe {
    /// Build a probe. NVML initialization happens eagerly so the first
    /// `foreign_gpu_busy()` call is cheap. A failing init logs at warn
    /// and the probe reports "no foreign contention" from then on,
    /// preserving the acceptance criterion on machines without
    /// NVIDIA GPUs.
    pub fn new(presence: Arc<PresenceManager>) -> Self {
        let nvml = match Nvml::init() {
            Ok(n) => Some(Arc::new(n)),
            Err(err) => {
                tracing::warn!(
                    target: "assistd::voice::probe",
                    "NVML init failed ({err}); foreign-process contention detection disabled"
                );
                None
            }
        };
        let self_pid = std::process::id();
        Self {
            presence,
            nvml,
            self_pid,
        }
    }
}

#[async_trait]
impl BusyProbe for PresenceGpuProbe {
    async fn wait_until_llm_idle(&self, timeout: Duration) -> bool {
        self.presence.wait_until_llm_idle(timeout).await
    }

    fn foreign_gpu_busy(&self) -> bool {
        let Some(nvml) = self.nvml.as_ref() else {
            return false;
        };
        let llama_pid = self.presence.llama_pid_blocking();
        match gpu_monitor::collect_foreign_usage(nvml.as_ref(), self.self_pid, llama_pid) {
            Ok(samples) => samples
                .iter()
                .any(|p| p.used_mb >= FOREIGN_VRAM_THRESHOLD_MB),
            Err(err) => {
                tracing::debug!(
                    target: "assistd::voice::probe",
                    "NVML scan failed: {err}; treating GPU as free"
                );
                false
            }
        }
    }

    fn presence_active(&self) -> bool {
        self.presence.state() == assistd_ipc::PresenceState::Active
    }
}
