use std::path::PathBuf;

use crate::defaults::{
    DEFAULT_LISTEN_AGGRESSIVENESS, DEFAULT_LISTEN_ENABLED, DEFAULT_LISTEN_HOTKEY,
    DEFAULT_LISTEN_MAX_UTTERANCE_SECS, DEFAULT_LISTEN_MIN_UTTERANCE_MS,
    DEFAULT_LISTEN_ONSET_CONFIRM_MS, DEFAULT_LISTEN_PREROLL_MS, DEFAULT_LISTEN_SILENCE_MS,
    DEFAULT_LISTEN_START_ON_LAUNCH, DEFAULT_VOICE_HOTKEY, DEFAULT_VOICE_MAX_RECORDING_SECS,
    DEFAULT_WHISPER_BEAMS, DEFAULT_WHISPER_CPU_FALLBACK_ENABLED,
    DEFAULT_WHISPER_GPU_BUSY_TIMEOUT_MS, DEFAULT_WHISPER_MODEL, DEFAULT_WHISPER_PREFER_GPU,
    DEFAULT_WHISPER_VAD_ENABLED, DEFAULT_WHISPER_VAD_MODEL, DEFAULT_WHISPER_VAD_SILENCE_SECS,
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
    /// Hands-free continuous listening (VAD-gated). Disabled by default;
    /// when `enabled = true` the daemon keeps the mic open, segments
    /// utterances with `webrtc-vad`, and auto-dispatches each transcript
    /// to the agent loop.
    #[serde(default)]
    pub continuous: ContinuousListenConfig,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mic_device: None,
            hotkey: DEFAULT_VOICE_HOTKEY.to_string(),
            max_recording_secs: DEFAULT_VOICE_MAX_RECORDING_SECS,
            transcription: TranscriptionConfig::default(),
            continuous: ContinuousListenConfig::default(),
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
    /// How long Whisper will wait for an in-flight LLM stream to finish
    /// before falling back to a lazily-built CPU context. Only consulted
    /// when the primary Whisper context is GPU-backed and
    /// [`Self::cpu_fallback_enabled`] is true. `0` means "use CPU
    /// immediately whenever any LLM stream is inflight".
    #[serde(default = "default_whisper_gpu_busy_timeout_ms")]
    pub gpu_busy_timeout_ms: u32,
    /// Build a CPU fallback context on demand when the GPU is busy.
    /// Disable to force strict GPU-only transcription (users wait for
    /// the LLM stream to finish before their utterance is transcribed).
    #[serde(default = "default_whisper_cpu_fallback_enabled")]
    pub cpu_fallback_enabled: bool,
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
            gpu_busy_timeout_ms: DEFAULT_WHISPER_GPU_BUSY_TIMEOUT_MS,
            cpu_fallback_enabled: DEFAULT_WHISPER_CPU_FALLBACK_ENABLED,
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

fn default_whisper_gpu_busy_timeout_ms() -> u32 {
    DEFAULT_WHISPER_GPU_BUSY_TIMEOUT_MS
}

fn default_whisper_cpu_fallback_enabled() -> bool {
    DEFAULT_WHISPER_CPU_FALLBACK_ENABLED
}

/// Continuous (hands-free) listening settings. Runs only when
/// [`VoiceConfig::enabled`] and [`Self::enabled`] are both true.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContinuousListenConfig {
    /// Master switch for the feature. When false, no listener task is
    /// built even if `voice.enabled` is true — the PTT pipeline is
    /// unaffected.
    #[serde(default = "default_listen_enabled")]
    pub enabled: bool,
    /// Start listening automatically on daemon launch. When false the
    /// listener is built but idle; flip it on via the hotkey or the
    /// `assistd listen-start` IPC command.
    #[serde(default = "default_listen_start_on_launch")]
    pub start_on_launch: bool,
    /// Optional global hotkey that toggles listening on/off. Empty
    /// disables the hotkey binding; the IPC commands still work.
    #[serde(default = "default_listen_hotkey")]
    pub hotkey: String,
    /// Trailing silence required to mark the end of an utterance, in
    /// milliseconds. Shorter values respond faster; longer values
    /// tolerate mid-sentence pauses.
    #[serde(default = "default_listen_silence_ms")]
    pub silence_ms: u32,
    /// Utterances shorter than this are dropped without transcription —
    /// filters clicks and single-phoneme bursts.
    #[serde(default = "default_listen_min_utterance_ms")]
    pub min_utterance_ms: u32,
    /// Force-flush a utterance to whisper after this many seconds even
    /// if the user keeps speaking. Bounds memory use.
    #[serde(default = "default_listen_max_utterance_secs")]
    pub max_utterance_secs: u32,
    /// Audio kept in a rolling pre-roll ring and prepended to a new
    /// utterance so the first syllable isn't clipped between VAD onset
    /// confirmation and buffer start.
    #[serde(default = "default_listen_preroll_ms")]
    pub preroll_ms: u32,
    /// Consecutive voiced-frame duration needed to confirm speech
    /// onset. Guards against single-frame noise spikes (keyboard
    /// clicks, fan pops).
    #[serde(default = "default_listen_onset_confirm_ms")]
    pub onset_confirm_ms: u32,
    /// webrtc-vad aggressiveness level `0..=3`. Higher is more
    /// aggressive at rejecting non-speech.
    #[serde(default = "default_listen_aggressiveness")]
    pub aggressiveness: u8,
}

impl Default for ContinuousListenConfig {
    fn default() -> Self {
        Self {
            enabled: DEFAULT_LISTEN_ENABLED,
            start_on_launch: DEFAULT_LISTEN_START_ON_LAUNCH,
            hotkey: DEFAULT_LISTEN_HOTKEY.to_string(),
            silence_ms: DEFAULT_LISTEN_SILENCE_MS,
            min_utterance_ms: DEFAULT_LISTEN_MIN_UTTERANCE_MS,
            max_utterance_secs: DEFAULT_LISTEN_MAX_UTTERANCE_SECS,
            preroll_ms: DEFAULT_LISTEN_PREROLL_MS,
            onset_confirm_ms: DEFAULT_LISTEN_ONSET_CONFIRM_MS,
            aggressiveness: DEFAULT_LISTEN_AGGRESSIVENESS,
        }
    }
}

fn default_listen_enabled() -> bool {
    DEFAULT_LISTEN_ENABLED
}
fn default_listen_start_on_launch() -> bool {
    DEFAULT_LISTEN_START_ON_LAUNCH
}
fn default_listen_hotkey() -> String {
    DEFAULT_LISTEN_HOTKEY.to_string()
}
fn default_listen_silence_ms() -> u32 {
    DEFAULT_LISTEN_SILENCE_MS
}
fn default_listen_min_utterance_ms() -> u32 {
    DEFAULT_LISTEN_MIN_UTTERANCE_MS
}
fn default_listen_max_utterance_secs() -> u32 {
    DEFAULT_LISTEN_MAX_UTTERANCE_SECS
}
fn default_listen_preroll_ms() -> u32 {
    DEFAULT_LISTEN_PREROLL_MS
}
fn default_listen_onset_confirm_ms() -> u32 {
    DEFAULT_LISTEN_ONSET_CONFIRM_MS
}
fn default_listen_aggressiveness() -> u8 {
    DEFAULT_LISTEN_AGGRESSIVENESS
}
