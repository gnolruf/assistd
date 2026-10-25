//! CUDA availability: a GPU through NVML, and sherpa-onnx's CUDA provider
//! library where the binary loads its libraries from.

use std::path::Path;

use nvml_wrapper::Nvml;

/// The library a CUDA build of sherpa-onnx ships beside `libonnxruntime.so`.
const CUDA_PROVIDER_LIBRARY: &str = "libonnxruntime_providers_cuda.so";

/// Directories, relative to the executable's, on the binary's runpath.
const LIBRARY_DIRS: &[&str] = &[".", "../lib/assistd"];

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

/// True when sherpa-onnx's CUDA provider library sits in a directory the
/// binary loads its libraries from.
pub fn cuda_provider_installed() -> bool {
    let Some(exe_dir) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
    else {
        return false;
    };
    LIBRARY_DIRS
        .iter()
        .any(|dir| exe_dir.join(dir).join(CUDA_PROVIDER_LIBRARY).is_file())
}
