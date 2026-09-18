//! [`assistd_voice::BusyProbe`] backed by presence state and NVML.

use std::sync::Arc;
use std::time::Duration;

use assistd_core::PresenceManager;
use assistd_voice::BusyProbe;
use async_trait::async_trait;
use nvml_wrapper::Nvml;

use crate::gpu_monitor;

/// Minimum per-process VRAM (MiB) for a foreign PID to count as GPU
/// contention. Above a desktop compositor's noise floor, below a second
/// model runner.
const FOREIGN_VRAM_THRESHOLD_MB: u64 = 100;

/// Reports the GPU busy while an LLM stream is in flight or a foreign
/// process holds VRAM. Without NVML, foreign contention is never
/// reported.
pub struct PresenceGpuProbe {
    presence: Arc<PresenceManager>,
    nvml: Option<Arc<Nvml>>,
    self_pid: u32,
    /// Process names that may hold VRAM without counting as foreign. Must
    /// include `llama-server`: in router mode the model-running child is
    /// a separate PID from the one presence tracks, so the PID filter
    /// alone would push every transcription onto the CPU.
    allowlist: Vec<String>,
}

impl PresenceGpuProbe {
    pub fn new(presence: Arc<PresenceManager>, allowlist: Vec<String>) -> Self {
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
            allowlist,
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
            Ok(samples) => samples.iter().any(|p| {
                if self.allowlist.iter().any(|n| n == &p.name) {
                    return false;
                }
                p.used_mb >= FOREIGN_VRAM_THRESHOLD_MB
            }),
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
