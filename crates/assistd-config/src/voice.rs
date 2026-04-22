use std::path::PathBuf;

use crate::defaults::{
    DEFAULT_VOICE_HOTKEY, DEFAULT_VOICE_MAX_RECORDING_SECS, DEFAULT_WHISPER_BEAMS,
    DEFAULT_WHISPER_MODEL, DEFAULT_WHISPER_PREFER_GPU, DEFAULT_WHISPER_VAD_ENABLED,
    DEFAULT_WHISPER_VAD_MODEL, DEFAULT_WHISPER_VAD_SILENCE_SECS,
};
use serde::{Deserialize, Serialize};

/// Voice input settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VoiceConfig {
    /// Whether voice input is enabled.
    pub enabled: bool,
    /// ALSA/PulseAudio device name. `None` = system default.
    #[serde(default)]
    pub mic_device: Option<String>,
    /// Hotkey to hold for push-to-talk recording (e.g. "Super+Space").
    /// Empty disables the in-daemon/TUI global hotkey listener — the PTT
    /// IPC commands (`assistd ptt-start` / `ptt-stop`) still work.
    pub hotkey: String,
    /// Upper bound on a single PTT recording, in seconds. The ring
    /// buffer drops newer samples past this length; transcription still
    /// runs on whatever was captured.
    #[serde(default = "default_voice_max_recording_secs")]
    pub max_recording_secs: u32,
    /// Speech-to-text transcription settings.
    #[serde(default)]
    pub transcription: TranscriptionConfig,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mic_device: None,
            hotkey: DEFAULT_VOICE_HOTKEY.to_string(),
            max_recording_secs: DEFAULT_VOICE_MAX_RECORDING_SECS,
            transcription: TranscriptionConfig::default(),
        }
    }
}

fn default_voice_max_recording_secs() -> u32 {
    DEFAULT_VOICE_MAX_RECORDING_SECS
}

/// Whisper transcription settings. Every field has a default; the full
/// section can be omitted from the TOML.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TranscriptionConfig {
    /// HuggingFace identifier for the Whisper GGML model, formatted as
    /// `<owner>/<repo>:<file>`. Downloaded on first use and cached under
    /// `model_cache_dir` (or `$XDG_CACHE_HOME/assistd/whisper/`).
    #[serde(default = "default_whisper_model")]
    pub model: String,
    /// Prefer GPU inference when available. Falls back to CPU with a
    /// warning log if no CUDA device is detected or whisper-rs was built
    /// without the `cuda` feature.
    #[serde(default = "default_whisper_prefer_gpu")]
    pub prefer_gpu: bool,
    /// CPU thread count. `None` lets whisper.cpp choose.
    #[serde(default)]
    pub threads: Option<u32>,
    /// Number of beams for decoding. `1` = greedy; larger values improve
    /// accuracy at the cost of latency.
    #[serde(default = "default_whisper_beams")]
    pub beams: u32,
    /// Enable Silero VAD to trim silence before decoding.
    #[serde(default = "default_whisper_vad_enabled")]
    pub vad_enabled: bool,
    /// HuggingFace identifier for the VAD GGML model. Only used when
    /// `vad_enabled = true`.
    #[serde(default = "default_whisper_vad_model")]
    pub vad_model: String,
    /// Approximate minimum silence length (in seconds) required to split
    /// or trim a segment. Maps to whisper.cpp's VAD
    /// `min_silence_duration_ms`.
    #[serde(default = "default_whisper_vad_silence_secs")]
    pub vad_silence_secs: f32,
    /// Override for the on-disk model cache directory. `None` uses
    /// `$XDG_CACHE_HOME/assistd/whisper/` (or `~/.cache/assistd/whisper/`).
    #[serde(default)]
    pub model_cache_dir: Option<PathBuf>,
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_WHISPER_MODEL.to_string(),
            prefer_gpu: DEFAULT_WHISPER_PREFER_GPU,
            threads: None,
            beams: DEFAULT_WHISPER_BEAMS,
            vad_enabled: DEFAULT_WHISPER_VAD_ENABLED,
            vad_model: DEFAULT_WHISPER_VAD_MODEL.to_string(),
            vad_silence_secs: DEFAULT_WHISPER_VAD_SILENCE_SECS,
            model_cache_dir: None,
        }
    }
}

fn default_whisper_model() -> String {
    DEFAULT_WHISPER_MODEL.to_string()
}

fn default_whisper_vad_model() -> String {
    DEFAULT_WHISPER_VAD_MODEL.to_string()
}

fn default_whisper_prefer_gpu() -> bool {
    DEFAULT_WHISPER_PREFER_GPU
}

fn default_whisper_beams() -> u32 {
    DEFAULT_WHISPER_BEAMS
}

fn default_whisper_vad_enabled() -> bool {
    DEFAULT_WHISPER_VAD_ENABLED
}

fn default_whisper_vad_silence_secs() -> f32 {
    DEFAULT_WHISPER_VAD_SILENCE_SECS
}
