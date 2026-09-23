//! Voice subsystem: the [`VoiceInput`] (speech-to-text) and
//! [`VoiceOutput`] (text-to-speech) traits, with Whisper, cpal, and
//! Piper implementations behind cargo features.

use async_trait::async_trait;
use tokio::sync::watch;

pub use assistd_ipc::VoiceCaptureState;

#[cfg(feature = "whisper")]
pub mod gpu;
#[cfg(any(feature = "whisper", feature = "tts"))]
pub mod hf_download;
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
pub mod error;
pub mod sentence;

pub use controller::{SpeakDecision, VoiceOutputController};
pub use error::{VoiceInputError, VoiceOutputError};
#[cfg(feature = "listen")]
pub use listen::{ContinuousListener, ListenError, MicContinuousListener, NoContinuousListener};
#[cfg(feature = "mic")]
pub use mic::{
    AudioCaptureError, DeviceValidationError, MicVoiceInput, capture::validate as mic_validate,
};
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
    async fn start_recording(&self) -> Result<(), VoiceInputError>;

    /// Stop capture and transcribe. `Ok("")` means no speech was
    /// detected, not an error.
    async fn stop_and_transcribe(&self) -> Result<String, VoiceInputError>;

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
    async fn speak(&self, text: String) -> Result<(), VoiceOutputError>;

    /// Block until the playback queue drains.
    async fn wait_idle(&self) -> Result<(), VoiceOutputError> {
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
    async fn start_recording(&self) -> Result<(), VoiceInputError> {
        Err(VoiceInputError::Disabled)
    }

    async fn stop_and_transcribe(&self) -> Result<String, VoiceInputError> {
        Err(VoiceInputError::Disabled)
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
    async fn speak(&self, _text: String) -> Result<(), VoiceOutputError> {
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
    async fn no_voice_input_refuses_capture_and_stays_idle() {
        let input = NoVoiceInput::new();
        assert!(matches!(
            input.start_recording().await,
            Err(VoiceInputError::Disabled)
        ));
        assert!(matches!(
            input.stop_and_transcribe().await,
            Err(VoiceInputError::Disabled)
        ));
        assert_eq!(input.state(), VoiceCaptureState::Idle);
        assert_eq!(*input.subscribe().borrow(), VoiceCaptureState::Idle);
    }

    #[tokio::test]
    async fn no_voice_output_accepts_and_drops_speech() {
        NoVoiceOutput.speak("hi".into()).await.unwrap();
        NoVoiceOutput.wait_idle().await.unwrap();
    }
}
