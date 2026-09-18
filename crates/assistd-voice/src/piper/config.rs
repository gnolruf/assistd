use std::path::PathBuf;
use std::time::Duration;

use crate::piper::cache::VoiceFiles;

/// Sampling noise scale, at the value upstream recommends.
pub const NOISE_SCALE: f32 = 0.667;
/// Phoneme noise scale, at the value upstream recommends.
pub const NOISE_W: f32 = 0.8;
/// Trailing silence Piper inserts after each utterance, in seconds.
pub const SENTENCE_SILENCE_SECS: f32 = 0.2;

/// Resolved, ready-to-spawn Piper configuration.
#[derive(Debug, Clone)]
pub struct PiperRuntimeConfig {
    pub binary_path: PathBuf,
    pub voice_files: VoiceFiles,
    pub length_scale: f32,
    pub noise_scale: f32,
    pub noise_w: f32,
    pub sentence_silence_secs: f32,
    pub espeak_data_dir: Option<PathBuf>,
    pub deadline: Duration,
    /// Pass `--cuda` to piper. A CPU-only binary exits non-zero, which
    /// the circuit breaker surfaces.
    pub use_cuda: bool,
    /// cpal output-device name (as in `aplay -L`); `None` for the
    /// system default.
    pub output_device: Option<String>,
}
