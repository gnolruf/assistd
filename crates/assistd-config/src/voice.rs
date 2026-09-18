use std::num::NonZeroU32;
use std::path::PathBuf;

use crate::defaults::{
    DEFAULT_LISTEN_ENABLED, DEFAULT_LISTEN_HOTKEY, DEFAULT_LISTEN_MAX_UTTERANCE_SECS,
    DEFAULT_LISTEN_SILENCE_MS, DEFAULT_LISTEN_START_ON_LAUNCH, DEFAULT_PIPER_BINARY,
    DEFAULT_PIPER_DEADLINE_SECS, DEFAULT_PIPER_ENABLED, DEFAULT_PIPER_LENGTH_SCALE,
    DEFAULT_PIPER_MAX_SENTENCE_CHARS, DEFAULT_PIPER_PARTIAL_FLUSH_MS, DEFAULT_PIPER_SKIP_HOTKEY,
    DEFAULT_PIPER_TOGGLE_HOTKEY, DEFAULT_PIPER_VOICE, DEFAULT_VOICE_HOTKEY,
    DEFAULT_VOICE_MAX_RECORDING_SECS, DEFAULT_WHISPER_BEAMS, DEFAULT_WHISPER_MODEL,
    DEFAULT_WHISPER_PREFER_GPU, DEFAULT_WHISPER_VAD_ENABLED, DEFAULT_WHISPER_VAD_MODEL,
};
use serde::{Deserialize, Serialize};

/// Voice input settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct VoiceConfig {
    /// Whether voice input is enabled.
    pub enabled: bool,
    /// ALSA/PulseAudio device name. `None` = system default.
    pub mic_device: Option<String>,
    /// Hotkey to hold for push-to-talk recording (e.g. "Super+Space").
    /// Empty disables the in-daemon/TUI global hotkey listener; the PTT
    /// IPC commands (`assistd ptt-start` / `ptt-stop`) still work.
    pub hotkey: String,
    /// Upper bound on a single PTT recording, in seconds. The ring
    /// buffer drops newer samples past this length; transcription still
    /// runs on whatever was captured.
    pub max_recording_secs: NonZeroU32,
    /// Speech-to-text transcription settings.
    pub transcription: TranscriptionConfig,
    /// Hands-free continuous listening (VAD-gated). Disabled by default;
    /// when `enabled = true` the daemon keeps the mic open, segments
    /// utterances with `webrtc-vad`, and auto-dispatches each transcript
    /// to the agent loop.
    pub continuous: ContinuousListenConfig,
    /// Text-to-speech synthesis (Piper). Disabled by default; when
    /// `enabled = true` the daemon spawns piper per utterance and plays
    /// LLM responses aloud through the default audio output.
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

/// Whisper transcription settings. Every field has a default; the full
/// section can be omitted from the TOML.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct TranscriptionConfig {
    /// HuggingFace identifier for the Whisper GGML model, formatted as
    /// `<owner>/<repo>:<file>`. Downloaded on first use and cached under
    /// `model_cache_dir` (or `$XDG_CACHE_HOME/assistd/whisper/`).
    pub model: String,
    /// Prefer GPU inference when available. Falls back to CPU with a
    /// warning log if no CUDA device is detected or whisper-rs was built
    /// without the `cuda` feature.
    pub prefer_gpu: bool,
    /// CPU thread count. `None` lets whisper.cpp choose.
    pub threads: Option<NonZeroU32>,
    /// Number of beams for decoding. `1` = greedy; larger values improve
    /// accuracy at the cost of latency.
    pub beams: NonZeroU32,
    /// Enable Silero VAD to trim silence before decoding.
    pub vad_enabled: bool,
    /// HuggingFace identifier for the VAD GGML model. Only used when
    /// `vad_enabled = true`.
    pub vad_model: String,
    /// Override for the on-disk model cache directory. `None` uses
    /// `$XDG_CACHE_HOME/assistd/whisper/` (or `~/.cache/assistd/whisper/`).
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

/// Continuous (hands-free) listening settings. Runs only when
/// [`VoiceConfig::enabled`] and [`Self::enabled`] are both true.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ContinuousListenConfig {
    /// Master switch for the feature. When false, no listener task is
    /// built even if `voice.enabled` is true; the PTT pipeline is
    /// unaffected.
    pub enabled: bool,
    /// Start listening automatically on daemon launch. When false the
    /// listener is built but idle; flip it on via the hotkey or the
    /// `assistd listen-start` IPC command.
    pub start_on_launch: bool,
    /// Optional global hotkey that toggles listening on/off. Empty
    /// disables the hotkey binding; the IPC commands still work.
    pub hotkey: String,
    /// Trailing silence required to mark the end of an utterance, in
    /// milliseconds. Shorter values respond faster; longer values
    /// tolerate mid-sentence pauses.
    pub silence_ms: NonZeroU32,
    /// Force-flush a utterance to whisper after this many seconds even
    /// if the user keeps speaking. Bounds memory use.
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

/// Piper text-to-speech settings. The full section can be omitted from
/// the TOML; everything has a default. `enabled = false` means the
/// daemon won't spawn piper, won't download voice models, and any LLM
/// response stays silent.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct SynthesisConfig {
    /// Master switch. When false, the daemon substitutes a silent
    /// `NoVoiceOutput` placeholder and skips Piper startup entirely.
    pub enabled: bool,
    /// Path to (or name of) the piper binary. Looked up via `$PATH`
    /// when the value is a bare command name.
    pub binary_path: PathBuf,
    /// HuggingFace identifier for the Piper voice ONNX, formatted as
    /// `<owner>/<repo>:<file>` where `<file>` is the path of the
    /// `.onnx` file inside the repo. The matching `.onnx.json` is
    /// downloaded alongside it. Cached under `model_cache_dir` (or
    /// `$XDG_CACHE_HOME/assistd/piper/`).
    pub voice: String,
    /// Override for the on-disk voice cache directory. `None` uses
    /// `$XDG_CACHE_HOME/assistd/piper/` (or `~/.cache/assistd/piper/`).
    pub model_cache_dir: Option<PathBuf>,
    /// Speaking-rate scale. `1.0` is the voice's natural rate; lower
    /// values speak faster, higher values speak slower.
    pub length_scale: f32,
    /// Optional override for Piper's espeak-ng data directory. Most
    /// distro packages set this themselves; only set when piper logs
    /// "Failed to load espeak-ng".
    pub espeak_data_dir: Option<PathBuf>,
    /// Per-utterance synthesis deadline in seconds. The piper child is
    /// killed if it hasn't returned PCM by this point.
    pub deadline_secs: NonZeroU32,
    /// Maximum sentence length fed to Piper. The sentence buffer flushes
    /// at the last whitespace before this cap when no terminator appears
    /// within the limit.
    pub max_sentence_chars: NonZeroU32,
    /// Idle gap (ms) between LLM deltas after which the sentence buffer
    /// is flushed even without a terminator. `0` disables the timeout
    /// flush; only the terminal `Done`-based flush is used. Inhibited
    /// while a tool call is in flight.
    pub partial_flush_ms: u32,
    /// How fenced code blocks in the LLM response are spoken aloud.
    pub code_block_mode: CodeBlockMode,
    /// Global hotkey (e.g. `"Super+Shift+M"`) that flips TTS on/off
    /// mid-session. Empty disables the binding. Turning off cancels
    /// in-flight playback; turning back on resumes for the next
    /// sentence delivered by the LLM (sentences arriving while off are
    /// silently dropped).
    pub toggle_hotkey: String,
    /// Global hotkey (e.g. `"Super+Shift+S"`) that aborts the current
    /// response: stops playback, drops any queued sentences for the
    /// in-flight query, but does not start recording. Empty disables.
    pub skip_hotkey: String,
    /// Pass `--cuda` to piper, routing ONNX inference through the
    /// CUDA execution provider. Requires a piper binary linked
    /// against `onnxruntime-gpu`; CPU-only builds will exit with
    /// "CUDA execution provider not available". Default false to
    /// preserve the existing CPU behaviour.
    pub use_cuda: bool,
    /// Override cpal's default output device by name. When `None`,
    /// rodio opens whatever `cpal::default_host().default_output_device()`
    /// returns (usually the right thing), but on Sway+PipeWire systems
    /// where the user's default sink is a Bluetooth or other virtual
    /// PipeWire sink, cpal's default may pick a raw ALSA hardware card
    /// (HDMI, an unused analog port) and the audio goes nowhere.
    /// Common values to try: `"pipewire"`, `"pulse"`, or `"default"`;
    /// match a name from `aplay -L`.
    pub output_device: Option<String>,
}

/// How the sentence buffer treats fenced code blocks in the LLM response.
///
/// Triple-backtick fences are detected per-character as the stream comes
/// in. The selected mode decides what (if anything) is spoken.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodeBlockMode {
    /// Drop fenced content silently. The default: code in a chat
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
