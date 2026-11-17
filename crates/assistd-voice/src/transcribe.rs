//! The [`Transcriber`] trait and its error type.

use std::fmt;
use std::io;

use async_trait::async_trait;

/// Transcribe 16 kHz mono PCM audio into text.
#[async_trait]
pub trait Transcriber: fmt::Debug + Send + Sync + 'static {
    /// Transcribe signed 16-bit mono samples at 16 kHz. Returns trimmed text;
    /// an empty string means the input was silence, not an error.
    async fn transcribe(&self, pcm_i16_16k_mono: &[i16]) -> Result<String, TranscriptionError>;
}

/// Errors surfaced by a [`Transcriber`] implementation.
#[derive(Debug, thiserror::Error)]
pub enum TranscriptionError {
    #[error("empty audio buffer")]
    EmptyAudio,

    #[error(transparent)]
    Download(#[from] assistd_utils::hf::download::DownloadError),

    #[error("failed to load the speech-to-text model: {0}")]
    ModelLoad(String),

    #[error("failed to hand sherpa-onnx the CUDA provider options: {0}")]
    CudaOptions(#[source] io::Error),

    #[error("transcription failed: {0}")]
    Inference(String),

    #[error("failed to load Silero VAD: {0}")]
    VadInit(String),

    #[error("tokio join error: {0}")]
    Join(#[from] tokio::task::JoinError),
}
