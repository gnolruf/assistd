//! Lightweight CUDA availability probe via NVML. Mirrors the idiom in
//! `crates/assistd/src/gpu_monitor.rs`: if NVML init or `device_count`
//! fail, treat it as "no GPU" and let the caller fall back to CPU.

use nvml_wrapper::Nvml;

/// Returns `true` iff NVML initializes AND reports at least one device.
/// Any failure mode (no driver, `/dev/nvidia*` hidden, non-NVIDIA GPU) is
/// collapsed to `false`.
pub fn probe_cuda_available() -> bool {
    match Nvml::init() {
        Ok(nvml) => match nvml.device_count() {
            Ok(count) => count > 0,
            Err(err) => {
                tracing::debug!(
                    target: "assistd::voice::gpu",
                    "nvml.device_count() failed: {err}"
                );
                false
            }
        },
        Err(err) => {
            tracing::debug!(
                target: "assistd::voice::gpu",
                "Nvml::init() failed: {err}"
            );
            false
        }
    }
}
