//! CUDA availability probe via NVML.

use nvml_wrapper::Nvml;

/// True iff NVML initializes and reports at least one device. Every
/// failure mode collapses to `false`.
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
