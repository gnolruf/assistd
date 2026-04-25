use std::path::PathBuf;
use std::time::Duration;

use crate::piper::cache::VoiceFiles;

/// Resolved, ready-to-spawn Piper configuration. Built once at startup
/// from `assistd_config::SynthesisConfig` plus the on-disk voice files
/// returned by [`crate::piper::cache::ensure_voice`]. Cloning is cheap
/// (everything is `Arc`-shareable) so the supervising service holds an
/// `Arc<PiperRuntimeConfig>` and each `OneShotSynth::synthesize` call
/// reads from it without locking.
#[derive(Debug, Clone)]
pub struct PiperRuntimeConfig {
    pub binary_path: String,
    pub voice_files: VoiceFiles,
    pub length_scale: f32,
    pub noise_scale: f32,
    pub noise_w: f32,
    pub sentence_silence_secs: f32,
    pub espeak_data_dir: Option<PathBuf>,
    pub deadline: Duration,
}
