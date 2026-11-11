//! Line-delimited JSON spoken between the daemon and its voice worker over
//! the worker's stdin and stdout.

use std::time::Duration;

use assistd_config::VoiceConfig;
use assistd_ipc::{ListenPhase, VoiceCaptureState};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;

use crate::error::VoiceWorkerError;
use crate::listen::{Heard, ListenError, WakeTest};

/// How long the worker has to open, close or stop the mic.
const CAPTURE_DEADLINE: Duration = Duration::from_secs(15);

/// How long the worker has to transcribe a push-to-talk recording.
const TRANSCRIPTION_DEADLINE: Duration = Duration::from_secs(120);

/// How long the worker has to synthesize and queue one sentence.
const SYNTHESIS_DEADLINE: Duration = Duration::from_secs(60);

/// A line from the other side that is not a message this build knows.
#[derive(Debug, Error)]
#[error("malformed voice-worker message: {0}")]
pub struct ProtocolError(#[from] serde_json::Error);

/// Daemon → worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum ToWorker {
    /// The first line: the voice settings to build the stack from.
    Configure {
        config: Box<VoiceConfig>,
    },
    /// A request the worker answers with a [`FromWorker::Reply`] of the same id.
    Call {
        id: u64,
        request: Request,
    },
    SetReplying {
        replying: bool,
    },
    OpenFollowUp,
    /// Whether the GPU is free for transcription, re-sent whenever it changes.
    Gpu {
        presence_active: bool,
        foreign_busy: bool,
    },
}

/// What a [`ToWorker::Call`] asks for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Request {
    Capture(CaptureRequest),
    Speech(SpeechRequest),
}

impl Request {
    /// How long the worker may take to answer; `None` for requests that last
    /// as long as the user or the queued audio does.
    pub(super) fn deadline(&self) -> Option<Duration> {
        match self {
            Self::Capture(CaptureRequest::WakeTestStart)
            | Self::Speech(SpeechRequest::WaitIdle) => None,
            Self::Capture(CaptureRequest::PttStop) => Some(TRANSCRIPTION_DEADLINE),
            Self::Capture(_) | Self::Speech(SpeechRequest::Cancel) => Some(CAPTURE_DEADLINE),
            Self::Speech(SpeechRequest::Speak { .. }) => Some(SYNTHESIS_DEADLINE),
        }
    }
}

/// A request for the mic: push-to-talk, listening, or a wake-word test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub(super) enum CaptureRequest {
    PttStart,
    /// Answered with [`Outcome::Transcript`].
    PttStop,
    ListenStart,
    ListenStop,
    ListenStopForPushToTalk,
    /// Answered once the test ends; its reports come as [`FromWorker::WakeTest`].
    WakeTestStart,
    WakeTestStop,
}

/// A request for the speaker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub(super) enum SpeechRequest {
    Speak { text: String },
    Cancel,
    WaitIdle,
}

/// Worker → daemon.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum FromWorker {
    CaptureReadiness { startup: Startup },
    SpeechReadiness { startup: Startup },
    CaptureState { state: VoiceCaptureState },
    ListenPhase { phase: ListenPhase },
    Heard { heard: Heard },
    WakeTest { report: WakeTest },
    Reply { id: u64, outcome: Outcome },
}

/// The one field read from a line that is not a [`ToWorker`] this build knows.
#[derive(Deserialize)]
struct UndecodedCall {
    id: u64,
}

/// How a component's startup in the worker ended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(super) enum Startup {
    Ready,
    Unavailable { reason: String },
}

/// The answer to one [`Request`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(super) enum Outcome {
    Done,
    Transcript { text: String },
    Failed { failure: Failure },
}

/// Why a [`Request`] failed, typed where the daemon acts on the reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "message", rename_all = "snake_case")]
pub(super) enum Failure {
    ListenDisabled,
    ListenTesting,
    ListenListening,
    Other(String),
}

impl Failure {
    pub(super) fn other(error: impl std::fmt::Display) -> Self {
        Self::Other(error.to_string())
    }

    pub(super) fn into_listen_error(self) -> ListenError {
        match self {
            Self::ListenDisabled => ListenError::Disabled,
            Self::ListenTesting => ListenError::Testing,
            Self::ListenListening => ListenError::Listening,
            Self::Other(message) => VoiceWorkerError::Failed(message).into(),
        }
    }

    pub(super) fn into_worker_error(self) -> VoiceWorkerError {
        match self {
            Self::Other(message) => VoiceWorkerError::Failed(message),
            listen => VoiceWorkerError::Failed(listen.into_listen_error().to_string()),
        }
    }
}

impl From<ListenError> for Failure {
    fn from(error: ListenError) -> Self {
        match error {
            ListenError::Disabled => Self::ListenDisabled,
            ListenError::Testing => Self::ListenTesting,
            ListenError::Listening => Self::ListenListening,
            other => Self::Other(other.to_string()),
        }
    }
}

/// `message` as one line, newline included.
pub(super) fn encode(message: &impl Serialize) -> String {
    let mut line = serde_json::to_string(message).expect("voice-worker messages always serialize");
    line.push('\n');
    line
}

pub(super) fn decode<T: DeserializeOwned>(line: &str) -> Result<T, ProtocolError> {
    Ok(serde_json::from_str(line)?)
}

/// The reply failing a line that did not decode for `error`, when the line
/// still carries a call id for the daemon to match it to.
pub(super) fn refusal(line: &str, error: &ProtocolError) -> Option<FromWorker> {
    let UndecodedCall { id } = serde_json::from_str(line).ok()?;
    Some(FromWorker::Reply {
        id,
        outcome: Outcome::Failed {
            failure: Failure::other(error),
        },
    })
}

#[cfg(test)]
mod tests;
