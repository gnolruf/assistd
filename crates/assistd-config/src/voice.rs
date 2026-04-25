use std::path::PathBuf;

use crate::defaults::{
    DEFAULT_LISTEN_AGGRESSIVENESS, DEFAULT_LISTEN_ENABLED, DEFAULT_LISTEN_HOTKEY,
    DEFAULT_LISTEN_MAX_UTTERANCE_SECS, DEFAULT_LISTEN_MIN_UTTERANCE_MS,
    DEFAULT_LISTEN_ONSET_CONFIRM_MS, DEFAULT_LISTEN_PREROLL_MS, DEFAULT_LISTEN_SILENCE_MS,
    DEFAULT_LISTEN_START_ON_LAUNCH, DEFAULT_PIPER_BINARY, DEFAULT_PIPER_DEADLINE_SECS,
    DEFAULT_PIPER_ENABLED, DEFAULT_PIPER_LENGTH_SCALE, DEFAULT_PIPER_MAX_SENTENCE_CHARS,
    DEFAULT_PIPER_NOISE_SCALE, DEFAULT_PIPER_NOISE_W, DEFAULT_PIPER_PARTIAL_FLUSH_MS,
    DEFAULT_PIPER_SENTENCE_SILENCE_SECS, DEFAULT_PIPER_VOICE, DEFAULT_VOICE_HOTKEY,
    DEFAULT_VOICE_MAX_RECORDING_SECS, DEFAULT_WHISPER_BEAMS, DEFAULT_WHISPER_CPU_FALLBACK_ENABLED,
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
    /// Text-to-speech synthesis (Piper). Disabled by default; when
    /// `enabled = true` the daemon spawns piper per utterance and plays
    /// LLM responses aloud through the default audio output.
    #[serde(default)]
    pub synthesis: SynthesisConfig,
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
            synthesis: SynthesisConfig::default(),
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

/// Piper text-to-speech settings. The full section can be omitted from
/// the TOML; everything has a default. `enabled = false` means the
/// daemon won't spawn piper, won't download voice models, and any LLM
/// response stays silent.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SynthesisConfig {
    /// Master switch. When false, the daemon substitutes a silent
    /// `NoVoiceOutput` placeholder and skips Piper startup entirely.
    #[serde(default = "default_piper_enabled")]
    pub enabled: bool,
    /// Path to (or name of) the piper binary. Looked up via `$PATH`
    /// when the value is a bare command name.
    #[serde(default = "default_piper_binary")]
    pub binary_path: String,
    /// HuggingFace identifier for the Piper voice ONNX, formatted as
    /// `<owner>/<repo>:<file>` where `<file>` is the path of the
    /// `.onnx` file inside the repo. The matching `.onnx.json` is
    /// downloaded alongside it. Cached under `model_cache_dir` (or
    /// `$XDG_CACHE_HOME/assistd/piper/`).
    #[serde(default = "default_piper_voice")]
    pub voice: String,
    /// Override for the on-disk voice cache directory. `None` uses
    /// `$XDG_CACHE_HOME/assistd/piper/` (or `~/.cache/assistd/piper/`).
    #[serde(default)]
    pub model_cache_dir: Option<PathBuf>,
    /// Speaking-rate scale. `1.0` is the voice's natural rate; lower
    /// values speak faster, higher values speak slower.
    #[serde(default = "default_piper_length_scale")]
    pub length_scale: f32,
    /// Sampling noise scale (Piper `--noise-scale`). Higher = more
    /// variation in pitch/intonation.
    #[serde(default = "default_piper_noise_scale")]
    pub noise_scale: f32,
    /// Phoneme noise scale (Piper `--noise-w`). Higher = more variation
    /// in cadence/stress.
    #[serde(default = "default_piper_noise_w")]
    pub noise_w: f32,
    /// Trailing silence (seconds) Piper inserts after each utterance.
    /// Maps to `--sentence-silence`.
    #[serde(default = "default_piper_sentence_silence_secs")]
    pub sentence_silence_secs: f32,
    /// Optional override for Piper's espeak-ng data directory. Most
    /// distro packages set this themselves; only set when piper logs
    /// "Failed to load espeak-ng".
    #[serde(default)]
    pub espeak_data_dir: Option<PathBuf>,
    /// Per-utterance synthesis deadline in seconds. The piper child is
    /// killed if it hasn't returned PCM by this point.
    #[serde(default = "default_piper_deadline_secs")]
    pub deadline_secs: u32,
    /// Maximum sentence length fed to Piper. The sentence buffer flushes
    /// at the last whitespace before this cap when no terminator appears
    /// within the limit.
    #[serde(default = "default_piper_max_sentence_chars")]
    pub max_sentence_chars: u32,
    /// Idle gap (ms) between LLM deltas after which the sentence buffer
    /// is flushed even without a terminator. `0` disables the timeout
    /// flush — only the terminal `Done`-based flush is used. Inhibited
    /// while a tool call is in flight.
    #[serde(default = "default_piper_partial_flush_ms")]
    pub partial_flush_ms: u32,
    /// How fenced code blocks in the LLM response are spoken aloud.
    #[serde(default)]
    pub code_block_mode: CodeBlockMode,
}

/// How the sentence buffer treats fenced code blocks in the LLM response.
///
/// Triple-backtick fences are detected per-character as the stream comes
/// in. The selected mode decides what (if anything) is spoken.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodeBlockMode {
    /// Drop fenced content silently. The default — code in a chat
    /// response is rarely useful as speech.
    #[default]
    Skip,
    /// Drop fenced content but emit one short phrase per fence so the
    /// listener knows code was elided. Captures the fence's language tag
    /// (e.g. ```` ```rust ````) and speaks "Code block in rust." when
    /// the fence closes; falls back to "Code block." with no tag.
    Summarize,
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            enabled: DEFAULT_PIPER_ENABLED,
            binary_path: DEFAULT_PIPER_BINARY.to_string(),
            voice: DEFAULT_PIPER_VOICE.to_string(),
            model_cache_dir: None,
            length_scale: DEFAULT_PIPER_LENGTH_SCALE,
            noise_scale: DEFAULT_PIPER_NOISE_SCALE,
            noise_w: DEFAULT_PIPER_NOISE_W,
            sentence_silence_secs: DEFAULT_PIPER_SENTENCE_SILENCE_SECS,
            espeak_data_dir: None,
            deadline_secs: DEFAULT_PIPER_DEADLINE_SECS,
            max_sentence_chars: DEFAULT_PIPER_MAX_SENTENCE_CHARS,
            partial_flush_ms: DEFAULT_PIPER_PARTIAL_FLUSH_MS,
            code_block_mode: CodeBlockMode::Skip,
        }
    }
}

fn default_piper_enabled() -> bool {
    DEFAULT_PIPER_ENABLED
}
fn default_piper_binary() -> String {
    DEFAULT_PIPER_BINARY.to_string()
}
fn default_piper_voice() -> String {
    DEFAULT_PIPER_VOICE.to_string()
}
fn default_piper_length_scale() -> f32 {
    DEFAULT_PIPER_LENGTH_SCALE
}
fn default_piper_noise_scale() -> f32 {
    DEFAULT_PIPER_NOISE_SCALE
}
fn default_piper_noise_w() -> f32 {
    DEFAULT_PIPER_NOISE_W
}
fn default_piper_sentence_silence_secs() -> f32 {
    DEFAULT_PIPER_SENTENCE_SILENCE_SECS
}
fn default_piper_deadline_secs() -> u32 {
    DEFAULT_PIPER_DEADLINE_SECS
}
fn default_piper_max_sentence_chars() -> u32 {
    DEFAULT_PIPER_MAX_SENTENCE_CHARS
}
fn default_piper_partial_flush_ms() -> u32 {
    DEFAULT_PIPER_PARTIAL_FLUSH_MS
}
