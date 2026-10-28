//! Voice input, wake word and speech output configuration.

use std::num::NonZeroU32;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::defaults::{
    DEFAULT_SILERO_VAD_MODEL, DEFAULT_STT_MODEL, DEFAULT_STT_PREFER_GPU, DEFAULT_STT_THREADS,
    DEFAULT_STT_VAD_ENABLED, DEFAULT_TTS_ENABLED, DEFAULT_TTS_LENGTH_SCALE,
    DEFAULT_TTS_MAX_SENTENCE_CHARS, DEFAULT_TTS_PARTIAL_FLUSH_MS, DEFAULT_TTS_VOICE,
    DEFAULT_WAKE_PHRASE, DEFAULT_WAKEWORD_BARGE_IN, DEFAULT_WAKEWORD_COMMAND_TIMEOUT_MS,
    DEFAULT_WAKEWORD_ENABLED, DEFAULT_WAKEWORD_FOLLOW_UP_MS, DEFAULT_WAKEWORD_MAX_UTTERANCE_SECS,
    DEFAULT_WAKEWORD_SILENCE_MS, DEFAULT_WAKEWORD_START_ON_LAUNCH, DEFAULT_WAKEWORD_THRESHOLD,
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

/// Speech-to-text settings: a Parakeet transducer run with sherpa-onnx.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TranscriptionConfig {
    /// The model's encoder `.onnx` as `<owner>/<repo>[@<revision>]:<file>`,
    /// whose name starts with `encoder`; the decoder and joiner with the same
    /// suffix and `tokens.txt` are fetched beside it. Downloaded
    /// on first use from `revision` (default `main`; a commit hash pins the
    /// content). Validated when voice is enabled.
    pub model: String,
    /// Transcribe on CUDA when an NVIDIA GPU is present and the binary ships
    /// sherpa-onnx's CUDA libraries (`libonnxruntime_providers_cuda.so`);
    /// otherwise, and while the GPU is busy or asleep, on the CPU.
    pub prefer_gpu: bool,
    /// CPU threads for transcription on the CPU, `1..`.
    pub threads: NonZeroU32,
    /// Trim silence with Silero VAD before decoding.
    pub vad_enabled: bool,
    /// Silero VAD ONNX model (v4 to v6) as `<owner>/<repo>[@<revision>]:<file>`;
    /// trims audio with `vad_enabled` and always decides where wake-word
    /// listening's utterances start and end.
    pub vad_model: String,
    /// Model cache directory, absolute or starting with `~/`. `None` uses
    /// `$XDG_CACHE_HOME/assistd/stt/` (or `~/.cache/assistd/stt/`).
    #[serde(deserialize_with = "crate::home_path::deserialize_option")]
    pub model_cache_dir: Option<PathBuf>,
}

impl DescribeSettings for TranscriptionConfig {
    describe_fields! {
        model => (
            "speech-to-text model",
            "Parakeet encoder model, as owner/repo[@revision]:encoder*.onnx"
        ),
        prefer_gpu => (
            "prefer GPU if available",
            "Transcribe on the GPU when CUDA is available",
            advanced
        ),
        threads => (
            "CPU threads",
            "CPU threads for transcription on the CPU",
            advanced
        ),
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
            "Where speech-to-text models download; unset uses the user cache",
            advanced
        ),
    }
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_STT_MODEL.to_string(),
            prefer_gpu: DEFAULT_STT_PREFER_GPU,
            threads: DEFAULT_STT_THREADS,
            vad_enabled: DEFAULT_STT_VAD_ENABLED,
            vad_model: DEFAULT_SILERO_VAD_MODEL.to_string(),
            model_cache_dir: None,
        }
    }
}

/// Hands-free listening: VAD-segmented speech becomes a query only after a
/// keyword spotter hears the wake phrase. Runs only when both
/// [`VoiceConfig::enabled`] and [`Self::enabled`] are set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct WakeWordConfig {
    /// Makes wake-word listening available; push-to-talk is unaffected.
    pub enabled: bool,
    /// The wake phrase: one to six words of letters and apostrophes, case
    /// ignored. Refused at startup when the keyword model cannot spell it.
    pub phrase: String,
    /// Keyword-spotter acceptance threshold, `0.0..=1.0`: the average token
    /// probability the phrase must reach; lower wakes more easily and more
    /// often by mistake.
    pub threshold: UnitInterval,
    /// Start listening at launch; otherwise the listener idles until toggled.
    pub start_on_launch: bool,
    /// Trailing silence, in ms, that ends an utterance, `1..=1400`; the
    /// keyword spotter resets itself after 1.5 s without speech.
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
}

impl DescribeSettings for WakeWordConfig {
    describe_fields! {
        enabled => ("enabled", "Makes wake-word listening available"),
        phrase => ("wake phrase", "One to six words that wake the listener"),
        threshold => (
            "detection threshold",
            "How sure the spotter must be, 0 to 1; lower wakes more easily",
            advanced
        ),
        start_on_launch => (
            "listen from launch",
            "Start listening at launch instead of waiting to be turned on"
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
            phrase: DEFAULT_WAKE_PHRASE.to_string(),
            threshold: DEFAULT_WAKEWORD_THRESHOLD,
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

/// The sherpa-onnx model family a voice belongs to, which fixes the files
/// fetched beside its `.onnx` and how text becomes phonemes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum VoiceFamily {
    /// A Piper voice re-exported with sherpa-onnx metadata (k2-fsa
    /// `vits-piper-*`), with `tokens.txt` beside it; text is phonemized by
    /// running `espeak-ng`, which must be on `$PATH`.
    Vits,
    /// Kokoro, with `voices.bin`, `tokens.txt` and `lexicon-us-en.txt` beside
    /// it; words missing from the lexicon are skipped.
    Kokoro,
}

/// Text-to-speech settings: a sherpa-onnx voice.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SynthesisConfig {
    /// Enables TTS. When `false`, no voice is downloaded or loaded.
    pub enabled: bool,
    /// Voice model as `<owner>/<repo>[@<revision>]:<file>`, `<file>` being
    /// the `.onnx` path in the repo; the files `family` needs are fetched
    /// from the same directory.
    pub voice: String,
    /// Model family of `voice`.
    pub family: VoiceFamily,
    /// Speaker index for multi-speaker voices, from `0`; must be below the
    /// voice's speaker count, which is `1` for single-speaker voices.
    pub speaker_id: u32,
    /// Voice cache directory, absolute or starting with `~/`. `None` uses
    /// `$XDG_CACHE_HOME/assistd/tts/` (or `~/.cache/assistd/tts/`).
    #[serde(deserialize_with = "crate::home_path::deserialize_option")]
    pub model_cache_dir: Option<PathBuf>,
    /// Speaking-rate scale: `1.0` is natural, lower is faster. Must be
    /// positive and finite.
    pub length_scale: f32,
    /// Sentence length cap in Unicode chars; longer text flushes at the
    /// last whitespace before it, or is hard-cut without one. `50..=2000`.
    pub max_sentence_chars: NonZeroU32,
    /// Idle gap, in ms, between LLM deltas after which buffered text is
    /// spoken unterminated. `0` disables; suspended during tool calls.
    pub partial_flush_ms: u32,
    /// Output device as listed by `aplay -L` (e.g. `"pipewire"`). `None`
    /// uses the audio host's default, which may be a raw ALSA card.
    pub output_device: Option<String>,
}

impl DescribeSettings for SynthesisConfig {
    describe_fields! {
        enabled => ("enabled", "Enables spoken replies"),
        voice => ("voice", "Voice model, as owner/repo[@revision]:file.onnx"),
        family => ("voice family", "Model family of the voice: vits or kokoro"),
        speaker_id => (
            "speaker",
            "Speaker index for multi-speaker voices; 0 for single-speaker voices"
        ),
        model_cache_dir => (
            "voice cache folder",
            "Where voices download; unset uses the user cache",
            advanced
        ),
        length_scale => (
            "speaking pace",
            "Speaking-rate scale; 1.0 is natural, lower is faster",
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
        output_device => ("speaker device", "Output device as listed by aplay -L; unset uses the default"),
    }
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            enabled: DEFAULT_TTS_ENABLED,
            voice: DEFAULT_TTS_VOICE.to_string(),
            family: VoiceFamily::Vits,
            speaker_id: 0,
            model_cache_dir: None,
            length_scale: DEFAULT_TTS_LENGTH_SCALE,
            max_sentence_chars: DEFAULT_TTS_MAX_SENTENCE_CHARS,
            partial_flush_ms: DEFAULT_TTS_PARTIAL_FLUSH_MS,
            output_device: None,
        }
    }
}

#[cfg(test)]
mod tests;
