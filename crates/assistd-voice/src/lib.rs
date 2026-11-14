//! Voice subsystem: the [`VoiceInput`] (speech-to-text) and [`VoiceOutput`] (text-to-speech)
//! traits, with their sherpa-onnx, cpal and rodio implementations.

use std::fmt;

use async_trait::async_trait;
use tokio::sync::watch;

pub use assistd_ipc::VoiceCaptureState;

pub mod controller;
mod device;
pub mod error;
pub mod gpu;
pub mod listen;
pub mod manager;
pub mod mic;
pub mod sentence;
pub mod stt;
pub mod transcribe;
pub mod tts;
pub mod vad;
pub mod worker;

pub use controller::{SpeakDecision, SpeakingGuard, VoiceOutputController};
pub use error::{VoiceInputError, VoiceOutputError, VoiceWorkerError};
pub use listen::{
    ContinuousListener, Heard, ListenError, MicContinuousListener, NoContinuousListener, SileroVad,
    SpotterError, WakeTest, WakeWordSpotter,
};
pub use manager::{CaptureUnavailable, VoiceCapture, VoiceManager};
pub use mic::{AudioCaptureError, MicVoiceInput};
pub use sentence::SentenceBuffer;
pub use stt::SherpaTranscriber;
pub use transcribe::{
    BusyProbe, CpuFallbackFactory, NullBusyProbe, QueuedTranscriber, Transcriber,
    TranscriptionError,
};
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

    /// Block until the playback queue drains. Errors when the audio output
    /// stops playing with speech still queued.
    async fn wait_idle(&self) -> Result<(), VoiceOutputError> {
        Ok(())
    }

    /// Drop pending audio. Idempotent.
    async fn cancel(&self) {}
}
