//! Typed errors for the [`VoiceInput`](crate::VoiceInput) and
//! [`VoiceOutput`](crate::VoiceOutput) traits and for calls into the voice worker.

use std::time::Duration;

use thiserror::Error;

/// Errors surfaced by [`VoiceInput`](crate::VoiceInput) implementations.
#[derive(Debug, Error)]
pub enum VoiceInputError {
    /// This build or configuration has no voice input.
    #[error("voice input is not enabled in this build")]
    Disabled,

    #[cfg(feature = "mic")]
    #[error("audio capture error: {0}")]
    Capture(#[from] crate::mic::AudioCaptureError),

    #[cfg(feature = "stt")]
    #[error("transcription error: {0}")]
    Transcription(#[from] crate::transcribe::TranscriptionError),

    #[error("capture task panicked: {0}")]
    ConsumerPanic(#[source] tokio::task::JoinError),

    #[error(transparent)]
    Worker(#[from] VoiceWorkerError),
}

/// Errors surfaced by [`VoiceOutput`](crate::VoiceOutput) implementations.
#[derive(Debug, Error)]
pub enum VoiceOutputError {
    #[cfg(feature = "tts")]
    #[error("speech synthesis failed: {0}")]
    Synthesis(#[source] crate::tts::SpeechError),

    #[cfg(feature = "tts")]
    #[error("speech playback enqueue failed: {0}")]
    Playback(#[source] crate::tts::SpeechError),

    #[error(transparent)]
    Worker(#[from] VoiceWorkerError),
}

/// A call into the voice worker process that did not succeed.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VoiceWorkerError {
    /// No worker is running to take the call.
    #[error("the voice worker is not running")]
    NotRunning,
    /// The worker exited before it answered.
    #[error("the voice worker exited")]
    Exited,
    /// The worker missed the call's deadline and is being restarted.
    #[error("the voice worker did not answer within {deadline:?}; restarting it")]
    Unresponsive { deadline: Duration },
    /// The worker answered with this failure.
    #[error("{0}")]
    Failed(String),
}
