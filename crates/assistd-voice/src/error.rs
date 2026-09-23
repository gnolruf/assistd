//! Typed errors for the [`VoiceInput`](crate::VoiceInput) and
//! [`VoiceOutput`](crate::VoiceOutput) traits.

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

    #[cfg(feature = "whisper")]
    #[error("transcription error: {0}")]
    Transcription(#[from] crate::transcribe::TranscriptionError),

    #[error("capture task panicked: {0}")]
    ConsumerPanic(#[source] tokio::task::JoinError),

    /// The request to the daemon that owns the microphone failed.
    #[error("daemon IPC error: {0}")]
    Ipc(#[from] assistd_ipc::IpcClientError),
}

/// Errors surfaced by [`VoiceOutput`](crate::VoiceOutput) implementations.
#[derive(Debug, Error)]
pub enum VoiceOutputError {
    #[cfg(feature = "tts")]
    #[error("piper synthesis failed: {0}")]
    Synthesis(#[source] crate::piper::PiperError),

    #[cfg(feature = "tts")]
    #[error("piper playback enqueue failed: {0}")]
    Playback(#[source] crate::piper::PiperError),
}
