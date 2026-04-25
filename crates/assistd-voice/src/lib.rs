//! Voice subsystem: input (speech-to-text) and output (text-to-speech) traits.
//!
//! Milestone 5 ships voice input: [`WhisperTranscriber`] (whisper-rs) and
//! [`MicVoiceInput`] (cpal + ring buffer + rubato) for push-to-talk, plus
//! [`MicContinuousListener`] (webrtc-vad) for hands-free capture. These
//! sit behind the `whisper`, `mic`, and `listen` cargo features
//! respectively. Milestone 6 will add a `Piper` implementation of
//! [`VoiceOutput`]; the trait lives here already so the daemon can hold
//! it as `Arc<dyn VoiceOutput>` from the moment the feature lands,
//! without touching the protocol or handler shape.

use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::watch;

pub use assistd_ipc::VoiceCaptureState;

#[cfg(feature = "whisper")]
pub mod gpu;
#[cfg(feature = "whisper")]
pub mod model_cache;
#[cfg(feature = "whisper")]
pub mod transcribe;
#[cfg(feature = "whisper")]
pub mod whisper;

#[cfg(feature = "mic")]
pub mod mic;

#[cfg(feature = "listen")]
pub mod listen;

#[cfg(feature = "tts")]
pub mod piper;

pub mod controller;
pub mod sentence;

pub use controller::{SpeakDecision, VoiceOutputController};
#[cfg(feature = "listen")]
pub use listen::{ContinuousListener, MicContinuousListener, NoContinuousListener};
#[cfg(feature = "mic")]
pub use mic::{MicVoiceInput, VoiceInputError, capture::validate as mic_validate};
#[cfg(feature = "tts")]
pub use piper::{PiperError, PiperVoiceOutput};
pub use sentence::SentenceBuffer;
#[cfg(feature = "whisper")]
pub use transcribe::{
    BusyProbe, CpuFallbackFactory, NullBusyProbe, QueueConfig, QueuedTranscriber, Transcriber,
    TranscriptionError,
};
#[cfg(feature = "whisper")]
pub use whisper::{WhisperTranscriber, WhisperTranscriberBuilder, build_cpu_fallback};

/// Push-to-talk voice capture. Implementors buffer mic audio between
/// [`start_recording`](VoiceInput::start_recording) and
/// [`stop_and_transcribe`](VoiceInput::stop_and_transcribe), then run
/// whisper on the buffered samples and return the transcribed text.
///
/// The three-state lifecycle is visible to callers via [`state`](Self::state)
/// and the watch receiver from [`subscribe`](Self::subscribe), so TUIs can
/// render "idle / recording / transcribing" without polling.
#[async_trait]
pub trait VoiceInput: Send + Sync + 'static {
    /// Open the capture device and begin buffering audio. Returns once
    /// the cpal stream has started; the caller must eventually call
    /// [`stop_and_transcribe`](Self::stop_and_transcribe) or the
    /// recording will run up to the configured cap.
    async fn start_recording(&self) -> Result<()>;

    /// Stop the capture, transcribe the buffered audio, and return the
    /// text. Returns `Ok("")` when VAD trimmed the audio down to silence
    /// — callers should treat empty strings as "no speech detected",
    /// not as an error.
    async fn stop_and_transcribe(&self) -> Result<String>;

    /// Current capture state — cheap synchronous snapshot.
    fn state(&self) -> VoiceCaptureState;

    /// Subscribe to state transitions. The initial value is the current
    /// state, and each subsequent change is published at most once per
    /// transition.
    fn subscribe(&self) -> watch::Receiver<VoiceCaptureState>;
}

/// Speak the given text aloud.
///
/// Designed for streaming pipelines where the daemon hands utterances to
/// TTS one sentence at a time. The implementation is expected to keep an
/// internal playback queue so sequential `speak` calls produce
/// back-to-back audio with no audible gap.
#[async_trait]
pub trait VoiceOutput: Send + Sync + 'static {
    /// Synthesize `text` and append the resulting audio to the playback
    /// queue. Returns once the audio is enqueued — **not** once playback
    /// finishes. Sequential calls produce back-to-back audio because the
    /// playback queue is FIFO. Use [`wait_idle`](Self::wait_idle) to await
    /// queue drain.
    async fn speak(&self, text: String) -> Result<()>;

    /// Block until the playback queue drains. Used at end-of-query so
    /// the worker isn't torn down before the last utterance plays.
    /// Default impl is a no-op for output backends that have no queue.
    async fn wait_idle(&self) -> Result<()> {
        Ok(())
    }

    /// Drop pending audio (mid-utterance interrupt). Idempotent and
    /// infallible — used by future "shut up" / barge-in handling.
    /// Default impl is a no-op.
    async fn cancel(&self) {}
}

/// Placeholder [`VoiceInput`] used when voice is disabled in config or
/// built without the `mic` feature. All methods refuse capture or report
/// the `Idle` state.
pub struct NoVoiceInput {
    state_tx: watch::Sender<VoiceCaptureState>,
}

impl Default for NoVoiceInput {
    fn default() -> Self {
        Self::new()
    }
}

impl NoVoiceInput {
    pub fn new() -> Self {
        let (state_tx, _) = watch::channel(VoiceCaptureState::Idle);
        Self { state_tx }
    }
}

#[async_trait]
impl VoiceInput for NoVoiceInput {
    async fn start_recording(&self) -> Result<()> {
        anyhow::bail!("voice input is not enabled in this build")
    }

    async fn stop_and_transcribe(&self) -> Result<String> {
        anyhow::bail!("voice input is not enabled in this build")
    }

    fn state(&self) -> VoiceCaptureState {
        VoiceCaptureState::Idle
    }

    fn subscribe(&self) -> watch::Receiver<VoiceCaptureState> {
        self.state_tx.subscribe()
    }
}

/// Placeholder [`VoiceOutput`] used when TTS is disabled in config or
/// built without the `tts` feature. Drops every request silently and
/// inherits the trait's no-op `wait_idle` / `cancel` defaults.
pub struct NoVoiceOutput;

#[async_trait]
impl VoiceOutput for NoVoiceOutput {
    async fn speak(&self, _text: String) -> Result<()> {
        Ok(())
    }
}

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn no_voice_input_start_errors() {
        assert!(NoVoiceInput::new().start_recording().await.is_err());
    }

    #[tokio::test]
    async fn no_voice_input_stop_errors() {
        assert!(NoVoiceInput::new().stop_and_transcribe().await.is_err());
    }

    #[tokio::test]
    async fn no_voice_input_state_is_idle() {
        assert_eq!(NoVoiceInput::new().state(), VoiceCaptureState::Idle);
    }

    #[tokio::test]
    async fn no_voice_output_is_silent_success() {
        NoVoiceOutput.speak("hi".into()).await.unwrap();
    }

    #[tokio::test]
    async fn no_voice_output_wait_idle_returns_ok() {
        NoVoiceOutput.wait_idle().await.unwrap();
    }

    #[tokio::test]
    async fn no_voice_output_cancel_does_not_panic() {
        NoVoiceOutput.cancel().await;
    }

    #[test]
    fn version_is_not_empty() {
        assert!(!version().is_empty());
    }
}
