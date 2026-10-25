use thiserror::Error;

/// Errors produced by the text-to-speech subsystem.
#[derive(Debug, Error)]
pub enum SpeechError {
    #[error(transparent)]
    Download(#[from] assistd_utils::hf::download::DownloadError),

    #[error(
        "voice {voice} has no tokens.txt beside it; stock rhasspy/piper-voices files do not \
         load in sherpa-onnx, so use a k2-fsa re-export such as \
         csukuangfj/vits-piper-en_US-lessac-medium"
    )]
    NotSherpaExport { voice: String },

    #[error("could not load the voice: {reason}")]
    LoadFailed { reason: String },

    #[error("could not speak the sentence: {reason}")]
    Rejected { reason: String },

    #[error("speech task failed: {0}")]
    Join(#[from] tokio::task::JoinError),

    #[error("rodio failed to open default audio output: {0}")]
    Audio(String),
}
