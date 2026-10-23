use std::num::NonZeroU32;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::defaults::{
    DEFAULT_PIPER_BINARY, DEFAULT_PIPER_DEADLINE_SECS, DEFAULT_PIPER_ENABLED,
    DEFAULT_PIPER_LENGTH_SCALE, DEFAULT_PIPER_MAX_SENTENCE_CHARS, DEFAULT_PIPER_PARTIAL_FLUSH_MS,
    DEFAULT_PIPER_VOICE, DEFAULT_WAKEWORD_BARGE_IN, DEFAULT_WAKEWORD_COMMAND_TIMEOUT_MS,
    DEFAULT_WAKEWORD_ENABLED, DEFAULT_WAKEWORD_FOLLOW_UP_MS, DEFAULT_WAKEWORD_MATCH_SIMILARITY,
    DEFAULT_WAKEWORD_MAX_UTTERANCE_SECS, DEFAULT_WAKEWORD_SILENCE_MS,
    DEFAULT_WAKEWORD_START_ON_LAUNCH, DEFAULT_WHISPER_BEAMS, DEFAULT_WHISPER_MODEL,
    DEFAULT_WHISPER_PREFER_GPU, DEFAULT_WHISPER_VAD_ENABLED, DEFAULT_WHISPER_VAD_MODEL,
};
use crate::errors::UnitIntervalError;
use crate::settings::{DescribeSettings, describe_fields};

/// Voice input and output settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct VoiceConfig {
    /// Enables voice input.
    pub enabled: bool,
    /// ALSA/PulseAudio input device. `None` uses the system default.
    pub mic_device: Option<String>,
    pub transcription: TranscriptionConfig,
    pub wakeword: WakeWordConfig,
    pub synthesis: SynthesisConfig,
}

impl DescribeSettings for VoiceConfig {
    describe_fields! {
        enabled => ("enabled", "Enables voice input"),
        mic_device => ("microphone", "ALSA/PulseAudio input device; unset uses the system default"),
        transcription => nested,
        wakeword => nested,
        synthesis => nested,
    }
}

/// Whisper speech-to-text settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TranscriptionConfig {
    /// Whisper GGML model as `<owner>/<repo>[@<revision>]:<file>`, downloaded
    /// on first use from `revision` (default `main`; a commit hash pins the
    /// content). Validated when voice is enabled.
    pub model: String,
    /// Prefer GPU inference; falls back to CPU when no CUDA device is found
    /// or the `cuda` feature is off.
    pub prefer_gpu: bool,
    /// CPU threads. `None` lets whisper.cpp choose.
    pub threads: Option<NonZeroU32>,
    /// Decoding beams, `1..=8`; `1` is greedy, and whisper.cpp runs at most
    /// eight decoders.
    pub beams: NonZeroU32,
    /// Trim silence with Silero VAD before decoding.
    pub vad_enabled: bool,
    /// Silero VAD GGML model as `<owner>/<repo>[@<revision>]:<file>`; trims
    /// audio with `vad_enabled` and always decides where wake-word listening's
    /// utterances start and end.
    pub vad_model: String,
    /// Model cache directory, absolute or starting with `~/`. `None` uses
    /// `$XDG_CACHE_HOME/assistd/whisper/` (or `~/.cache/assistd/whisper/`).
    #[serde(deserialize_with = "crate::home_path::deserialize_option")]
    pub model_cache_dir: Option<PathBuf>,
}

impl DescribeSettings for TranscriptionConfig {
    describe_fields! {
        model => ("speech-to-text model", "Whisper model, as owner/repo[@revision]:file"),
        prefer_gpu => (
            "prefer GPU if available",
            "Transcribe on the GPU when CUDA is available",
            advanced
        ),
        threads => (
            "CPU threads",
            "CPU threads for transcription; unset lets whisper.cpp choose",
            advanced
        ),
        beams => ("decoding beams", "Decoding beams, 1 to 8; 1 is greedy", advanced),
        vad_enabled => (
            "enable silence trimming",
            "Trim silence with Silero VAD before transcribing",
            advanced
        ),
        vad_model => (
            "voice-detection model",
            "Silero VAD model, as owner/repo[@revision]:file",
            advanced
        ),
        model_cache_dir => (
            "model cache folder",
            "Where Whisper models download; unset uses the user cache",
            advanced
        ),
    }
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

/// Hands-free listening: VAD-segmented speech becomes a query only after the
/// wake phrase, which is enrolled at runtime rather than set here. Runs only
/// when both [`VoiceConfig::enabled`] and [`Self::enabled`] are set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct WakeWordConfig {
    /// Makes wake-word listening available; push-to-talk is unaffected.
    pub enabled: bool,
    /// Start listening at launch; otherwise the listener idles until toggled.
    pub start_on_launch: bool,
    /// Trailing silence, in ms, that ends an utterance.
    pub silence_ms: NonZeroU32,
    /// Seconds after which an utterance is transcribed even mid-speech,
    /// `1..=300`; bounds the audio buffered under constant noise.
    pub max_utterance_secs: NonZeroU32,
    /// Let confirmed speech over a reply to a wake-word command interrupt it
    /// and become the next command; bumps, noise and the reply's own echo
    /// are rejected.
    pub barge_in: bool,
    /// After a wake phrase heard on its own, how long in ms to wait for the
    /// command, `500..=15000`.
    pub command_timeout_ms: NonZeroU32,
    /// After a reply to a wake-word command finishes, including its speech,
    /// how long in ms the next utterance is taken as a command without the
    /// phrase, `0..=30000`; `0` disables follow-ups.
    pub follow_up_ms: u32,
    /// Minimum normalized edit-distance similarity between each leading word
    /// Whisper heard and the same word of the phrase or a learned variant;
    /// `1.0` demands an exact match after normalization, `0.0` accepts anything.
    pub match_similarity: UnitInterval,
}

impl DescribeSettings for WakeWordConfig {
    describe_fields! {
        enabled => ("enabled", "Makes wake-word listening available"),
        start_on_launch => (
            "listen from launch",
            "Start listening at launch instead of waiting for the hotkey"
        ),
        silence_ms => (
            "end-of-speech silence (ms)",
            "Trailing silence that ends an utterance",
            advanced
        ),
        max_utterance_secs => (
            "max utterance (s)",
            "Transcribe an utterance after this long even mid-speech, 1 to 300",
            advanced
        ),
        barge_in => (
            "interrupt replies by talking",
            "Talking over a reply interrupts it and becomes the next command"
        ),
        command_timeout_ms => (
            "wait for command (ms)",
            "Wait for a command after the phrase alone, 500 to 15000",
            advanced
        ),
        follow_up_ms => (
            "follow-up window (ms)",
            "Window after a reply to speak again without the phrase; 0 disables"
        ),
        match_similarity => (
            "phrase match strictness",
            "How closely speech must match the phrase; 1.0 is exact",
            advanced
        ),
    }
}

impl Default for WakeWordConfig {
    fn default() -> Self {
        Self {
            enabled: DEFAULT_WAKEWORD_ENABLED,
            start_on_launch: DEFAULT_WAKEWORD_START_ON_LAUNCH,
            silence_ms: DEFAULT_WAKEWORD_SILENCE_MS,
            max_utterance_secs: DEFAULT_WAKEWORD_MAX_UTTERANCE_SECS,
            barge_in: DEFAULT_WAKEWORD_BARGE_IN,
            command_timeout_ms: DEFAULT_WAKEWORD_COMMAND_TIMEOUT_MS,
            follow_up_ms: DEFAULT_WAKEWORD_FOLLOW_UP_MS,
            match_similarity: DEFAULT_WAKEWORD_MATCH_SIMILARITY,
        }
    }
}

/// An `f32` in `0.0..=1.0`; never NaN.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "f32", into = "f32")]
pub struct UnitInterval(f32);

impl UnitInterval {
    /// Returns `None` when `value` is outside `0.0..=1.0` or NaN.
    pub const fn new(value: f32) -> Option<Self> {
        if value >= 0.0 && value <= 1.0 {
            Some(Self(value))
        } else {
            None
        }
    }

    pub const fn get(self) -> f32 {
        self.0
    }
}

impl TryFrom<f32> for UnitInterval {
    type Error = UnitIntervalError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::new(value).ok_or(UnitIntervalError(value))
    }
}

impl From<UnitInterval> for f32 {
    fn from(value: UnitInterval) -> Self {
        value.0
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
    /// Voice as `<owner>/<repo>[@<revision>]:<file>`, `<file>` being the
    /// `.onnx` path in the repo; its `.onnx.json` is fetched alongside.
    pub voice: String,
    /// Voice cache directory, absolute or starting with `~/`. `None` uses
    /// `$XDG_CACHE_HOME/assistd/piper/` (or `~/.cache/assistd/piper/`).
    #[serde(deserialize_with = "crate::home_path::deserialize_option")]
    pub model_cache_dir: Option<PathBuf>,
    /// Speaking-rate scale: `1.0` is natural, lower is faster. Must be
    /// positive and finite.
    pub length_scale: f32,
    /// espeak-ng data directory, absolute or starting with `~/`; set only if
    /// piper logs "Failed to load espeak-ng".
    #[serde(deserialize_with = "crate::home_path::deserialize_option")]
    pub espeak_data_dir: Option<PathBuf>,
    /// Seconds piper may take per utterance before it is killed.
    pub deadline_secs: NonZeroU32,
    /// Sentence length cap; longer text flushes at the last whitespace
    /// before it. `50..=2000`.
    pub max_sentence_chars: NonZeroU32,
    /// Idle gap, in ms, between LLM deltas after which buffered text is
    /// spoken unterminated. `0` disables; suspended during tool calls.
    pub partial_flush_ms: u32,
    /// Pass `--cuda` to piper; needs a piper built against `onnxruntime-gpu`.
    pub use_cuda: bool,
    /// Output device as listed by `aplay -L` (e.g. `"pipewire"`). `None`
    /// uses the audio host's default, which may be a raw ALSA card.
    pub output_device: Option<String>,
}

impl DescribeSettings for SynthesisConfig {
    describe_fields! {
        enabled => ("enabled", "Enables spoken replies through piper"),
        binary_path => ("piper binary", "piper path, or a name on $PATH", advanced),
        voice => ("voice", "Piper voice, as owner/repo[@revision]:file.onnx"),
        model_cache_dir => (
            "voice cache folder",
            "Where piper voices download; unset uses the user cache",
            advanced
        ),
        length_scale => (
            "speaking pace",
            "Speaking-rate scale; 1.0 is natural, lower is faster",
            advanced
        ),
        espeak_data_dir => (
            "espeak-ng data folder",
            "espeak-ng data folder; set only if piper fails to load it",
            advanced
        ),
        deadline_secs => (
            "per-sentence timeout (s)",
            "Longest piper may take on one sentence before it is killed",
            advanced
        ),
        max_sentence_chars => (
            "max sentence length (chars)",
            "Longer text is split at whitespace, 50 to 2000",
            advanced
        ),
        partial_flush_ms => (
            "speak partial sentence after (ms)",
            "Speak unfinished text after this long without output; 0 disables",
            advanced
        ),
        use_cuda => ("use CUDA", "Run piper with --cuda; needs an onnxruntime-gpu build", advanced),
        output_device => ("speaker", "Output device as listed by aplay -L; unset uses the default"),
    }
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
            use_cuda: false,
            output_device: None,
        }
    }
}

#[cfg(test)]
mod tests;
