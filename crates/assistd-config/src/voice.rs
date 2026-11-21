//! Voice input, wake word and speech output configuration.

use std::fmt;
use std::num::NonZeroU32;
use std::str::FromStr;

use assistd_utils::hf::HfFileId;
use serde::{Deserialize, Serialize};

use crate::defaults::{bounded, nz32, unit};
use crate::errors::{SttEncoderError, UnitIntervalError, WakePhraseError};
use crate::settings::{DescribeSettings, choice_setting, describe_fields};
use crate::values::{AbsoluteDir, Bounded, HfFile, LengthScale};

const STT_ENCODER_PREFIX: &str = "encoder";
const MAX_WAKE_PHRASE_WORDS: usize = 6;

/// Voice input and output settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct VoiceConfig {
    /// Enables voice input.
    pub enabled: bool,
    /// ALSA/PulseAudio input device. `None` uses the system default. A named
    /// device that is unplugged, at startup or while listening, is polled for
    /// until it appears.
    pub mic_device: Option<String>,
    /// After a reply finishes, including its speech, how long in seconds a
    /// spoken prompt still continues the active session while no chat window
    /// is open, `0..=86400`; later it begins a new session. `0` never begins one.
    pub session_hold_secs: Bounded<0, 86_400>,
    pub transcription: TranscriptionConfig,
    pub wakeword: WakeWordConfig,
    pub synthesis: SynthesisConfig,
}

impl DescribeSettings for VoiceConfig {
    describe_fields! {
        enabled => ("enabled", "Enables voice input"),
        mic_device => ("microphone", "ALSA/PulseAudio input device; unset uses the system default"),
        session_hold_secs => (
            "session hold (s)",
            "Quiet time a spoken session is held before a spoken prompt starts a new one; 0 holds it for good"
        ),
        transcription => nested,
        wakeword => nested,
        synthesis => nested,
    }
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mic_device: None,
            session_hold_secs: bounded(16),
            transcription: TranscriptionConfig::default(),
            wakeword: WakeWordConfig::default(),
            synthesis: SynthesisConfig::default(),
        }
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
    /// content).
    pub model: SttEncoder,
    /// Transcribe on CUDA when an NVIDIA GPU is present and the binary ships
    /// sherpa-onnx's CUDA libraries (`libonnxruntime_providers_cuda.so`),
    /// holding the model's VRAM while voice input runs; otherwise on the CPU.
    pub prefer_gpu: bool,
    /// CPU threads for transcription on the CPU, `1..`.
    pub threads: NonZeroU32,
    /// Trim silence with Silero VAD before decoding.
    pub vad_enabled: bool,
    /// Model cache directory, absolute or starting with `~/`. `None` uses
    /// `$XDG_CACHE_HOME/assistd/stt/` (or `~/.cache/assistd/stt/`).
    pub model_cache_dir: Option<AbsoluteDir>,
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
            model: "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16\
                    @a6cc4e24ed264a93a17575dad416dacc4238497a:encoder.fp16.onnx"
                .parse()
                .expect("the default speech-to-text model parses"),
            prefer_gpu: false,
            threads: nz32(4),
            vad_enabled: true,
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
    pub phrase: WakePhrase,
    /// Keyword-spotter acceptance threshold, `0.0..=1.0`: the average token
    /// probability the phrase must reach; lower wakes more easily and more
    /// often by mistake.
    pub threshold: UnitInterval,
    /// Start listening at launch; otherwise the listener idles until toggled.
    pub start_on_launch: bool,
    /// Trailing silence, in ms, that ends an utterance, `1..=1400`; the
    /// keyword spotter resets itself after 1.5 s without speech.
    pub silence_ms: Bounded<1, 1400>,
    /// Seconds after which an utterance is transcribed even mid-speech,
    /// `1..=300`; bounds the audio buffered under constant noise.
    pub max_utterance_secs: Bounded<1, 300>,
    /// Let confirmed speech over a reply to a wake-word command interrupt it
    /// and become the next command; bumps, noise and the reply's own echo
    /// are rejected.
    pub barge_in: bool,
    /// After a wake phrase heard on its own, how long in ms to wait for the
    /// command, `500..=15000`.
    pub command_timeout_ms: Bounded<500, 15_000>,
    /// After a reply to a wake-word command finishes, including its speech,
    /// how long in ms the next utterance is taken as a command without the
    /// phrase, `0..=30000`; `0` disables follow-ups.
    pub follow_up_ms: Bounded<0, 30_000>,
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
            enabled: false,
            phrase: "hey computer"
                .parse()
                .expect("the default wake phrase parses"),
            threshold: unit(0.25),
            start_on_launch: false,
            silence_ms: bounded(800),
            max_utterance_secs: bounded(30),
            barge_in: true,
            command_timeout_ms: bounded(4000),
            follow_up_ms: bounded(8000),
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

/// A speech-to-text model's encoder: a Hugging Face `.onnx` file whose name
/// starts with `encoder`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct SttEncoder(HfFileId);

impl SttEncoder {
    pub fn id(&self) -> &HfFileId {
        &self.0
    }

    /// What follows `encoder` in the file's name, which the decoder's and
    /// joiner's names end with too.
    pub fn suffix(&self) -> &str {
        file_name(&self.0)
            .strip_prefix(STT_ENCODER_PREFIX)
            .unwrap_or_default()
    }
}

impl fmt::Display for SttEncoder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl FromStr for SttEncoder {
    type Err = SttEncoderError;

    fn from_str(id: &str) -> Result<Self, Self::Err> {
        let file: HfFile = id.parse()?;
        if file_name(file.id()).starts_with(STT_ENCODER_PREFIX) {
            Ok(Self(file.id().clone()))
        } else {
            Err(SttEncoderError::NotAnEncoder(id.to_owned()))
        }
    }
}

impl TryFrom<String> for SttEncoder {
    type Error = SttEncoderError;

    fn try_from(id: String) -> Result<Self, Self::Error> {
        id.parse()
    }
}

impl From<SttEncoder> for String {
    fn from(encoder: SttEncoder) -> Self {
        encoder.0.to_string()
    }
}

/// A wake phrase: one to six words of letters and apostrophes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct WakePhrase(String);

impl WakePhrase {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for WakePhrase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl FromStr for WakePhrase {
    type Err = WakePhraseError;

    fn from_str(phrase: &str) -> Result<Self, Self::Err> {
        Self::try_from(phrase.to_owned())
    }
}

impl TryFrom<String> for WakePhrase {
    type Error = WakePhraseError;

    fn try_from(phrase: String) -> Result<Self, Self::Error> {
        let words = phrase.split_whitespace().count();
        let letters_only = phrase
            .chars()
            .all(|c| c.is_alphabetic() || c.is_whitespace() || c == '\'');
        if (1..=MAX_WAKE_PHRASE_WORDS).contains(&words) && letters_only {
            Ok(Self(phrase))
        } else {
            Err(WakePhraseError(phrase))
        }
    }
}

impl From<WakePhrase> for String {
    fn from(phrase: WakePhrase) -> Self {
        phrase.0
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

choice_setting!(VoiceFamily {
    Vits => "vits",
    Kokoro => "kokoro",
});

/// Text-to-speech settings: a sherpa-onnx voice.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SynthesisConfig {
    /// Enables TTS. When `false`, no voice is downloaded or loaded.
    pub enabled: bool,
    /// Voice model as `<owner>/<repo>[@<revision>]:<file>`, `<file>` being
    /// the `.onnx` path in the repo; the files `family` needs are fetched
    /// from the same directory.
    pub voice: HfFile,
    /// Model family of `voice`.
    pub family: VoiceFamily,
    /// Speaker index for multi-speaker voices, from `0`; must be below the
    /// voice's speaker count, which is `1` for single-speaker voices.
    pub speaker_id: u32,
    /// Voice cache directory, absolute or starting with `~/`. `None` uses
    /// `$XDG_CACHE_HOME/assistd/tts/` (or `~/.cache/assistd/tts/`).
    pub model_cache_dir: Option<AbsoluteDir>,
    /// Speaking-rate scale: `1.0` is natural, lower is faster. Positive
    /// and finite.
    pub length_scale: LengthScale,
    /// Sentence length cap in Unicode chars; longer text flushes at the
    /// last whitespace before it, or is hard-cut without one. `50..=2000`.
    pub max_sentence_chars: Bounded<50, 2000>,
    /// Idle gap, in ms, between LLM deltas after which buffered text is
    /// spoken unterminated. `0` disables; suspended during tool calls.
    pub partial_flush_ms: u32,
    /// Output device as listed by `aplay -L` (e.g. `"pipewire"`). `None`
    /// uses the audio host's default, which may be a raw ALSA card.
    pub output_device: Option<String>,
    /// Ask the model to phrase a reply that will be read aloud as short spoken
    /// sentences, to say what it is about to do before slow work, and to open
    /// detail that does not suit speech on screen instead of reading it out.
    pub conversational: bool,
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
        conversational => (
            "conversational replies",
            "Ask for short spoken sentences, a word before slow work, and detail on screen, while replies are read aloud"
        ),
    }
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            voice: "csukuangfj/vits-piper-en_US-lessac-medium\
                    @83f7470750b36549037551ef7007fa3b4b8697e0:en_US-lessac-medium.onnx"
                .parse()
                .expect("the default voice parses"),
            family: VoiceFamily::Vits,
            speaker_id: 0,
            model_cache_dir: None,
            length_scale: LengthScale::new(1.0).expect("the default length scale is positive"),
            max_sentence_chars: bounded(400),
            partial_flush_ms: 750,
            output_device: None,
            conversational: true,
        }
    }
}

fn file_name(id: &HfFileId) -> &str {
    id.file().rsplit('/').next().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_of_range_threshold_is_refused() {
        for (edge, expected) in [("0.0", 0.0), ("1.0", 1.0)] {
            let parsed: WakeWordConfig = toml::from_str(&format!("threshold = {edge}\n"))
                .expect("in-range threshold must parse");
            assert_eq!(parsed.threshold, unit(expected));
        }

        for bad in ["1.5", "-0.1", "nan"] {
            let source = format!("threshold = {bad}\n");
            assert!(
                toml::from_str::<WakeWordConfig>(&source).is_err(),
                "threshold = {bad} must be refused"
            );
        }
    }
}
