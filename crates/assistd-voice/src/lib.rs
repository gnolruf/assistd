//! Voice subsystem: the [`VoiceInput`] (speech-to-text) and [`VoiceOutput`] (text-to-speech)
//! traits, with sherpa-onnx and cpal implementations behind cargo features.

use std::fmt;

use async_trait::async_trait;
use tokio::sync::watch;

pub use assistd_ipc::VoiceCaptureState;

#[cfg(feature = "stt")]
pub mod gpu;
#[cfg(feature = "stt")]
pub mod stt;
#[cfg(feature = "stt")]
pub mod transcribe;
#[cfg(feature = "stt")]
pub mod vad;

#[cfg(feature = "mic")]
pub mod mic;

#[cfg(feature = "listen")]
pub mod listen;

#[cfg(feature = "tts")]
pub mod tts;

pub mod controller;
pub mod error;
#[cfg(feature = "listen")]
pub mod manager;
pub mod sentence;
#[cfg(all(feature = "listen", feature = "tts"))]
pub mod worker;

pub use controller::{SpeakDecision, SpeakingGuard, VoiceOutputController};
pub use error::{VoiceInputError, VoiceOutputError, VoiceWorkerError};
#[cfg(feature = "listen")]
pub use listen::{
    ContinuousListener, Heard, ListenError, MicContinuousListener, NoContinuousListener, SileroVad,
    SpotterError, WakeTest, WakeWordSpotter,
};
#[cfg(feature = "listen")]
pub use manager::{CaptureUnavailable, VoiceCapture, VoiceManager};
#[cfg(feature = "mic")]
pub use mic::{AudioCaptureError, MicVoiceInput};
pub use sentence::SentenceBuffer;
#[cfg(feature = "stt")]
pub use stt::SherpaTranscriber;
#[cfg(feature = "stt")]
pub use transcribe::{
    BusyProbe, CpuFallbackFactory, NullBusyProbe, QueuedTranscriber, Transcriber,
    TranscriptionError,
};
#[cfg(feature = "tts")]
pub use tts::{SherpaVoiceOutput, SpeechError};

/// Push-to-talk voice capture: buffer mic audio between
/// [`start_recording`](VoiceInput::start_recording) and
/// [`stop_and_transcribe`](VoiceInput::stop_and_transcribe), then transcribe it.
#[async_trait]
pub trait VoiceInput: fmt::Debug + Send + Sync + 'static {
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
pub trait VoiceOutput: fmt::Debug + Send + Sync + 'static {
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
