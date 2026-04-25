use std::path::PathBuf;
use std::process::ExitStatus;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PiperError {
    #[error(
        "invalid voice identifier '{id}': {reason} \
         (expected '<owner>/<repo>:<file>')"
    )]
    VoiceParse { id: String, reason: String },

    #[error("voice download failed for {url}: {source}")]
    Download {
        url: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("voice download for {url} returned HTTP {status}")]
    Http { url: String, status: u16 },

    #[error("voice file at {path} is not valid JSON: starts with {prefix:?}")]
    JsonShape { path: PathBuf, prefix: String },

    #[error("voice config at {path} did not parse: {source}")]
    JsonParse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("voice file I/O at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("piper binary '{binary}' not found on PATH")]
    BinaryMissing { binary: String },

    #[error("failed to spawn piper at {binary}: {source}")]
    Spawn {
        binary: String,
        #[source]
        source: std::io::Error,
    },

    #[error("piper exited with {status} after producing {bytes} bytes; last stderr: {stderr_tail}")]
    SynthFailed {
        status: ExitStatus,
        bytes: usize,
        stderr_tail: String,
    },

    #[error("piper produced odd byte count {bytes}; not a valid 16-bit PCM stream")]
    OddPcmLength { bytes: usize },

    #[error("piper synthesis exceeded deadline of {secs}s")]
    Deadline { secs: u64 },

    #[error("rodio failed to open default audio output: {0}")]
    Audio(String),

    #[error("playback worker channel closed; rodio thread has exited")]
    PlaybackClosed,

    #[error("piper service is degraded: {0}")]
    Degraded(String),
}
