use std::num::NonZeroU32;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::defaults::{
    DEFAULT_LISTEN_ENABLED, DEFAULT_LISTEN_HOTKEY, DEFAULT_LISTEN_MAX_UTTERANCE_SECS,
    DEFAULT_LISTEN_SILENCE_MS, DEFAULT_LISTEN_START_ON_LAUNCH, DEFAULT_PIPER_BINARY,
    DEFAULT_PIPER_DEADLINE_SECS, DEFAULT_PIPER_ENABLED, DEFAULT_PIPER_LENGTH_SCALE,
    DEFAULT_PIPER_MAX_SENTENCE_CHARS, DEFAULT_PIPER_PARTIAL_FLUSH_MS, DEFAULT_PIPER_SKIP_HOTKEY,
    DEFAULT_PIPER_TOGGLE_HOTKEY, DEFAULT_PIPER_VOICE, DEFAULT_VOICE_HOTKEY,
    DEFAULT_VOICE_MAX_RECORDING_SECS, DEFAULT_WHISPER_BEAMS, DEFAULT_WHISPER_MODEL,
    DEFAULT_WHISPER_PREFER_GPU, DEFAULT_WHISPER_VAD_ENABLED, DEFAULT_WHISPER_VAD_MODEL,
};

/// Voice input and output settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct VoiceConfig {
    /// Enables voice input.
    pub enabled: bool,
    /// ALSA/PulseAudio input device. `None` uses the system default.
    pub mic_device: Option<String>,
    /// Push-to-talk hold hotkey (e.g. `"Super+Space"`). Empty disables it;
    /// the PTT IPC commands still work.
    pub hotkey: String,
    /// Max PTT recording length in seconds; audio past it is dropped.
    pub max_recording_secs: NonZeroU32,
    pub transcription: TranscriptionConfig,
    pub continuous: ContinuousListenConfig,
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

/// Whisper speech-to-text settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TranscriptionConfig {
    /// Whisper GGML model as `<owner>/<repo>:<file>`, downloaded on first
    /// use. Validated when voice is enabled.
    pub model: String,
    /// Prefer GPU inference; falls back to CPU when no CUDA device is found
    /// or the `cuda` feature is off.
    pub prefer_gpu: bool,
    /// CPU threads. `None` lets whisper.cpp choose.
    pub threads: Option<NonZeroU32>,
    /// Decoding beams; `1` is greedy.
    pub beams: NonZeroU32,
    /// Trim silence with Silero VAD before decoding.
    pub vad_enabled: bool,
    /// VAD GGML model as `<owner>/<repo>:<file>`; used only with `vad_enabled`.
    pub vad_model: String,
    /// Model cache directory. `None` uses `$XDG_CACHE_HOME/assistd/whisper/`
    /// (or `~/.cache/assistd/whisper/`).
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
            model_cache_dir: None,
        }
    }
}

/// Hands-free listening: VAD-segmented utterances are sent as queries.
/// Runs only when both [`VoiceConfig::enabled`] and [`Self::enabled`] are set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ContinuousListenConfig {
    /// Makes continuous listening available; push-to-talk is unaffected.
    pub enabled: bool,
    /// Start listening at launch; otherwise the listener idles until toggled.
    pub start_on_launch: bool,
    /// Hotkey that toggles listening. Empty disables it.
    pub hotkey: String,
    /// Trailing silence, in ms, that ends an utterance.
    pub silence_ms: NonZeroU32,
    /// Seconds after which an utterance is transcribed even mid-speech.
    pub max_utterance_secs: NonZeroU32,
}

impl Default for ContinuousListenConfig {
    fn default() -> Self {
        Self {
            enabled: DEFAULT_LISTEN_ENABLED,
            start_on_launch: DEFAULT_LISTEN_START_ON_LAUNCH,
            hotkey: DEFAULT_LISTEN_HOTKEY.to_string(),
            silence_ms: DEFAULT_LISTEN_SILENCE_MS,
            max_utterance_secs: DEFAULT_LISTEN_MAX_UTTERANCE_SECS,
        }
    }
}

/// Piper text-to-speech settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SynthesisConfig {
    /// Enables TTS. When `false`, piper is never spawned and no voice is
    /// downloaded.
    pub enabled: bool,
    /// Piper binary: a path, or a bare name looked up on `$PATH`. Must not
    /// be empty when enabled.
    pub binary_path: PathBuf,
    /// Voice as `<owner>/<repo>:<file>`, `<file>` being the `.onnx` path in
    /// the repo; its `.onnx.json` is fetched alongside.
    pub voice: String,
    /// Voice cache directory. `None` uses `$XDG_CACHE_HOME/assistd/piper/`
    /// (or `~/.cache/assistd/piper/`).
    pub model_cache_dir: Option<PathBuf>,
    /// Speaking-rate scale: `1.0` is natural, lower is faster. Must be
    /// positive and finite.
    pub length_scale: f32,
    /// espeak-ng data directory; set only if piper logs "Failed to load
    /// espeak-ng".
    pub espeak_data_dir: Option<PathBuf>,
    /// Seconds piper may take per utterance before it is killed.
    pub deadline_secs: NonZeroU32,
    /// Sentence length cap; longer text flushes at the last whitespace
    /// before it. At least 50.
    pub max_sentence_chars: NonZeroU32,
    /// Idle gap, in ms, between LLM deltas after which buffered text is
    /// spoken unterminated. `0` disables; suspended during tool calls.
    pub partial_flush_ms: u32,
    pub code_block_mode: CodeBlockMode,
    /// Hotkey that toggles TTS (e.g. `"Super+Shift+M"`). Empty disables it.
    /// Off cancels playback; sentences arriving while off are dropped.
    pub toggle_hotkey: String,
    /// Hotkey that stops playback and drops the current response's queued
    /// sentences. Empty disables it.
    pub skip_hotkey: String,
    /// Pass `--cuda` to piper; needs a piper built against `onnxruntime-gpu`.
    pub use_cuda: bool,
    /// Output device as listed by `aplay -L` (e.g. `"pipewire"`). `None`
    /// uses the audio host's default, which may be a raw ALSA card.
    pub output_device: Option<String>,
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            enabled: DEFAULT_PIPER_ENABLED,
            binary_path: DEFAULT_PIPER_BINARY.into(),
            voice: DEFAULT_PIPER_VOICE.to_string(),
            model_cache_dir: None,
            length_scale: DEFAULT_PIPER_LENGTH_SCALE,
            espeak_data_dir: None,
            deadline_secs: DEFAULT_PIPER_DEADLINE_SECS,
            max_sentence_chars: DEFAULT_PIPER_MAX_SENTENCE_CHARS,
            partial_flush_ms: DEFAULT_PIPER_PARTIAL_FLUSH_MS,
            code_block_mode: CodeBlockMode::Skip,
            toggle_hotkey: DEFAULT_PIPER_TOGGLE_HOTKEY.to_string(),
            skip_hotkey: DEFAULT_PIPER_SKIP_HOTKEY.to_string(),
            use_cuda: false,
            output_device: None,
        }
    }
}

/// How fenced code blocks in a reply are spoken.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodeBlockMode {
    /// Drop them silently.
    #[default]
    Skip,
    /// Drop them but say "Code block in <lang>." (or "Code block.") per fence.
    Summarize,
}
