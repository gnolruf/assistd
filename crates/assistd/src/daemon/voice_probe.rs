//! [`assistd_voice::BusyProbe`] backed by presence state and the GPU monitor.

use std::sync::Arc;

use assistd_core::{PresenceManager, PresenceState};
use assistd_voice::BusyProbe;

use super::vram::VramLedger;

/// Reports the GPU unavailable while presence is not `Active` or the GPU
/// monitor last found another process crowding the model. Without NVML,
/// contention is never reported.
#[derive(Debug)]
pub(super) struct PresenceGpuProbe {
    presence: Arc<PresenceManager>,
    vram: Option<Arc<VramLedger>>,
}

impl PresenceGpuProbe {
    pub(super) fn new(presence: Arc<PresenceManager>, vram: Option<Arc<VramLedger>>) -> Self {
        Self { presence, vram }
    }
}

impl BusyProbe for PresenceGpuProbe {
    fn foreign_gpu_busy(&self) -> bool {
        self.vram.as_ref().is_some_and(|vram| vram.contended())
    }

    fn presence_active(&self) -> bool {
        self.presence.state() == PresenceState::Active
    }
}
