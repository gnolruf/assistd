//! Speech-to-text primitive. The `Transcriber` trait is the stable
//! boundary: callers hand over a buffer of 16 kHz mono PCM samples and
//! get back a transcribed string, regardless of which backend is
//! running underneath.

use std::path::PathBuf;

use async_trait::async_trait;

/// Transcribe 16 kHz mono PCM audio into text.
#[async_trait]
pub trait Transcriber: Send + Sync + 'static {
    /// `pcm_i16_16k_mono` must be signed 16-bit samples at 16 kHz, single
    /// channel. Returns the transcribed text with leading/trailing
    /// whitespace trimmed. An empty string is returned when the input
    /// contains only silence (per VAD) — it is not an error.
    async fn transcribe(&self, pcm_i16_16k_mono: &[i16]) -> Result<String, TranscriptionError>;
}

/// Errors surfaced by a [`Transcriber`] implementation.
#[derive(Debug, thiserror::Error)]
pub enum TranscriptionError {
    #[error("empty audio buffer")]
    EmptyAudio,

    #[error("invalid model identifier {id:?}: {reason}")]
    ModelParse { id: String, reason: String },

    #[error("failed to download model from {url}: {source}")]
    ModelDownload {
        url: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("model download returned HTTP {status} from {url}")]
    ModelHttp { url: String, status: u16 },

    #[error("model I/O error at {path}: {source}")]
    ModelIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to initialize whisper context: {0}")]
    WhisperInit(String),

    #[error("whisper inference failed: {0}")]
    WhisperInference(String),

    #[error("tokio join error: {0}")]
    Join(#[from] tokio::task::JoinError),
}
