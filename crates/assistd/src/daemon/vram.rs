//! Who holds the GPU's VRAM and how much the model needs, shared by the
//! contention monitor, the pre-wake room check, and transcription's GPU probe.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use assistd_core::{PresenceState, SleepConfig, VramShortfall, WakeRoom};
use nvml_wrapper::{Nvml, enums::device::UsedGpuMemory};
use parking_lot::Mutex;
use procfs::process::Process;
use tracing::{info, warn};

const BYTES_PER_MB: u64 = 1024 * 1024;
const MAX_ANCESTOR_DEPTH: usize = 64;

/// Growth in other processes' VRAM, in MiB, that counts as one of them
/// claiming the GPU; smaller drift is desktop clients resizing buffers.
const FOREIGN_GROWTH_FLOOR_MB: u64 = 100;

/// VRAM across every GPU, in MiB. The daemon's own allocations, such as
/// transcription's, count toward neither holder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VramSnapshot {
    pub(super) free_mb: u64,
    /// Held by the llama-server router and its model children.
    pub(super) model_mb: u64,
    /// Held by every process outside assistd.
    pub(super) foreign_mb: u64,
}

/// What the polls have taught about the model and the other processes.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct Learned {
    /// Most VRAM the model has held while `Active`; `None` until measured.
    pub(super) model_footprint_mb: Option<u64>,
    /// Least foreign VRAM seen since the model was last up; `None` while
    /// it is down.
    pub(super) foreign_baseline_mb: Option<u64>,
    /// Whether the last poll found another process crowding the model.
    pub(super) contended: bool,
}

impl Learned {
    /// Fold in a poll taken in `state` and record whether another process
    /// has grown into the headroom since the model came up.
    pub(super) fn observe(
        &mut self,
        snapshot: VramSnapshot,
        state: PresenceState,
        headroom_mb: u64,
    ) {
        if state == PresenceState::Active {
            let footprint = self.model_footprint_mb.unwrap_or(0).max(snapshot.model_mb);
            self.model_footprint_mb = Some(footprint);
        }
        if !matches!(state, PresenceState::Active | PresenceState::Drowsy) {
            self.foreign_baseline_mb = None;
            self.contended = false;
            return;
        }
        let baseline = self
            .foreign_baseline_mb
            .map_or(snapshot.foreign_mb, |least| least.min(snapshot.foreign_mb));
        self.foreign_baseline_mb = Some(baseline);
        self.contended = snapshot.free_mb < headroom_mb
            && snapshot.foreign_mb >= baseline + FOREIGN_GROWTH_FLOOR_MB;
    }

    /// Whether the model, at its measured footprint, fits beside the
    /// headroom; an unmeasured model is let through to try.
    pub(super) fn room_to_wake(
        &self,
        snapshot: VramSnapshot,
        headroom_mb: u64,
    ) -> Result<(), VramShortfall> {
        let Some(footprint) = self.model_footprint_mb else {
            return Ok(());
        };
        let needed_mb = footprint.saturating_sub(snapshot.model_mb) + headroom_mb;
        if snapshot.free_mb >= needed_mb {
            Ok(())
        } else {
            Err(VramShortfall {
                free_mb: snapshot.free_mb,
                needed_mb,
            })
        }
    }
}

/// NVML plus what the monitor has learned from it.
#[derive(Debug)]
pub(super) struct VramLedger {
    nvml: Nvml,
    self_pid: u32,
    headroom_mb: u64,
    learned: Mutex<Learned>,
}

impl VramLedger {
    /// `None` when the monitor is off in config or there is no NVIDIA
    /// driver or GPU.
    pub(super) fn open(cfg: &SleepConfig) -> Option<Arc<Self>> {
        if !cfg.gpu_monitor_enabled {
            info!(
                target: "assistd::gpu_monitor",
                "sleep.gpu_monitor_enabled = false; GPU contention monitor and wake check disabled"
            );
            return None;
        }
        match Nvml::init() {
            Ok(nvml) => Some(Arc::new(Self {
                nvml,
                self_pid: std::process::id(),
                headroom_mb: cfg.gpu_headroom_mb,
                learned: Mutex::new(Learned::default()),
            })),
            Err(e) => {
                warn!(
                    target: "assistd::gpu_monitor",
                    "NVML init failed: {e}. GPU contention monitor and wake check disabled \
                     (no NVIDIA driver or no NVIDIA GPU?)"
                );
                None
            }
        }
    }

    pub(super) fn headroom_mb(&self) -> u64 {
        self.headroom_mb
    }

    /// VRAM now, with `llama_pid` and its descendants counted as the model.
    pub(super) fn snapshot(&self, llama_pid: Option<u32>) -> Result<VramSnapshot> {
        let mut snapshot = VramSnapshot {
            free_mb: 0,
            model_mb: 0,
            foreign_mb: 0,
        };
        for idx in 0..self.nvml.device_count()? {
            let device = self.nvml.device_by_index(idx)?;
            snapshot.free_mb += device.memory_info()?.free / BYTES_PER_MB;
            let compute = device.running_compute_processes().unwrap_or_default();
            let graphics = device.running_graphics_processes().unwrap_or_default();
            let mut per_pid: HashMap<u32, u64> = HashMap::new();
            for process in compute.into_iter().chain(graphics) {
                let bytes = match process.used_gpu_memory {
                    UsedGpuMemory::Used(bytes) => bytes,
                    UsedGpuMemory::Unavailable => 0,
                };
                let held = per_pid.entry(process.pid).or_default();
                *held = (*held).max(bytes);
            }
            for (pid, bytes) in per_pid {
                if pid == self.self_pid {
                    continue;
                }
                let held_mb = bytes / BYTES_PER_MB;
                if llama_pid.is_some_and(|root| descends_from(pid, root, read_parent_pid)) {
                    snapshot.model_mb += held_mb;
                } else {
                    snapshot.foreign_mb += held_mb;
                }
            }
        }
        Ok(snapshot)
    }

    /// Fold in a poll taken in `state`; true when another process now
    /// crowds the model.
    pub(super) fn observe(&self, snapshot: VramSnapshot, state: PresenceState) -> bool {
        let mut learned = self.learned.lock();
        learned.observe(snapshot, state, self.headroom_mb);
        learned.contended
    }

    /// Whether the last poll found another process crowding the model.
    pub(super) fn contended(&self) -> bool {
        self.learned.lock().contended
    }

    pub(super) fn room_to_wake(&self, snapshot: VramSnapshot) -> Result<(), VramShortfall> {
        self.learned.lock().room_to_wake(snapshot, self.headroom_mb)
    }
}

impl WakeRoom for VramLedger {
    fn check(&self, llama_pid: Option<u32>) -> Result<(), VramShortfall> {
        match self.snapshot(llama_pid) {
            Ok(snapshot) => self.room_to_wake(snapshot),
            Err(e) => {
                warn!(
                    target: "assistd::gpu_monitor",
                    "NVML read failed before wake: {e}; waking without a VRAM check"
                );
                Ok(())
            }
        }
    }
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
    let stat = Process::new(i32::try_from(pid).ok()?).ok()?.stat().ok()?;
    u32::try_from(stat.ppid).ok()
}

#[cfg(test)]
mod tests;
