#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::print_stdout,
        clippy::print_stderr
    )
)]

//! Voice subsystem: the [`VoiceInput`] (speech-to-text) and
//! [`VoiceOutput`] (text-to-speech) traits, with Whisper, cpal, and
//! Piper implementations behind cargo features.

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
#[cfg(all(feature = "whisper", any(test, feature = "test-support")))]
pub use transcribe::StubTranscriber;
#[cfg(feature = "whisper")]
pub use transcribe::{
    BusyProbe, CpuFallbackFactory, NullBusyProbe, QueueConfig, QueuedTranscriber, Transcriber,
    TranscriptionError,
};
#[cfg(feature = "whisper")]
pub use whisper::{WhisperTranscriber, WhisperTranscriberBuilder, build_cpu_fallback};

/// Push-to-talk voice capture: buffer mic audio between
/// [`start_recording`](VoiceInput::start_recording) and
/// [`stop_and_transcribe`](VoiceInput::stop_and_transcribe), then
/// transcribe it.
#[async_trait]
pub trait VoiceInput: Send + Sync + 'static {
    /// Open the capture device and begin buffering. Recording runs
    /// until [`stop_and_transcribe`](Self::stop_and_transcribe) or the
    /// configured cap.
    async fn start_recording(&self) -> Result<()>;

    /// Stop capture and transcribe. `Ok("")` means no speech was
    /// detected, not an error.
    async fn stop_and_transcribe(&self) -> Result<String>;

    /// Current capture state; cheap synchronous snapshot.
    fn state(&self) -> VoiceCaptureState;

    /// Subscribe to state transitions. The initial value is the
    /// current state.
    fn subscribe(&self) -> watch::Receiver<VoiceCaptureState>;
}

/// Text-to-speech with a FIFO playback queue, so sequential `speak`
/// calls produce back-to-back audio.
#[async_trait]
pub trait VoiceOutput: Send + Sync + 'static {
    /// Synthesize `text` and enqueue the audio. Returns once enqueued,
    /// not once played; use [`wait_idle`](Self::wait_idle) for that.
    async fn speak(&self, text: String) -> Result<()>;

    /// Block until the playback queue drains.
    async fn wait_idle(&self) -> Result<()> {
        Ok(())
    }

    /// Drop pending audio. Idempotent.
    async fn cancel(&self) {}
}

/// Placeholder [`VoiceInput`] that refuses capture and reports `Idle`.
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

/// Placeholder [`VoiceOutput`] that drops every request silently.
pub struct NoVoiceOutput;

#[async_trait]
impl VoiceOutput for NoVoiceOutput {
    async fn speak(&self, _text: String) -> Result<()> {
        Ok(())
    }
}

/// The crate version.
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
