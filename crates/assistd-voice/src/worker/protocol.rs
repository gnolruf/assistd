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
mod tests {
    use super::*;

    fn round_trips<T>(messages: &[T])
    where
        T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        for message in messages {
            let line = encode(message);
            assert!(line.ends_with('\n'), "{line}");
            assert_eq!(
                line.matches('\n').count(),
                1,
                "one line per message: {line}"
            );
            let decoded: T = decode(line.trim_end()).unwrap_or_else(|e| panic!("{line}: {e}"));
            assert_eq!(&decoded, message, "{line}");
        }
    }

    #[test]
    fn every_daemon_message_round_trips() {
        let requests = [
            Request::Capture(CaptureRequest::PttStart),
            Request::Capture(CaptureRequest::PttStop),
            Request::Capture(CaptureRequest::ListenStart),
            Request::Capture(CaptureRequest::ListenStop),
            Request::Capture(CaptureRequest::ListenStopForPushToTalk),
            Request::Capture(CaptureRequest::WakeTestStart),
            Request::Capture(CaptureRequest::WakeTestStop),
            Request::Speech(SpeechRequest::Speak {
                text: "Line one.\nLine \"two\".".into(),
            }),
            Request::Speech(SpeechRequest::Cancel),
            Request::Speech(SpeechRequest::WaitIdle),
        ];
        let mut messages: Vec<ToWorker> = requests
            .into_iter()
            .enumerate()
            .map(|(id, request)| ToWorker::Call {
                id: id as u64,
                request,
            })
            .collect();
        messages.extend([
            ToWorker::Configure {
                config: Box::default(),
            },
            ToWorker::SetReplying { replying: true },
            ToWorker::OpenFollowUp,
        ]);
        round_trips(&messages);
    }

    #[test]
    fn every_worker_message_round_trips() {
        let outcomes = [
            Outcome::Done,
            Outcome::Transcript {
                text: "turn it up".into(),
            },
            Outcome::Failed {
                failure: Failure::ListenDisabled,
            },
            Outcome::Failed {
                failure: Failure::ListenTesting,
            },
            Outcome::Failed {
                failure: Failure::ListenListening,
            },
            Outcome::Failed {
                failure: Failure::Other("no mic".into()),
            },
        ];
        let mut messages: Vec<FromWorker> = outcomes
            .into_iter()
            .zip(7..)
            .map(|(outcome, id)| FromWorker::Reply { id, outcome })
            .collect();
        messages.extend([
            FromWorker::CaptureReadiness {
                startup: Startup::Ready,
            },
            FromWorker::SpeechReadiness {
                startup: Startup::Unavailable {
                    reason: "no voice".into(),
                },
            },
            FromWorker::CaptureState {
                state: VoiceCaptureState::Transcribing,
            },
            FromWorker::ListenPhase {
                phase: ListenPhase::Capturing,
            },
            FromWorker::Heard {
                heard: Heard::WakePhrase,
            },
            FromWorker::Heard {
                heard: Heard::Command("what time is it".into()),
            },
            FromWorker::Heard {
                heard: Heard::BargeIn,
            },
            FromWorker::WakeTest {
                report: WakeTest::Speaking { speaking: true },
            },
            FromWorker::WakeTest {
                report: WakeTest::Utterance { matched: false },
            },
        ]);
        round_trips(&messages);
    }

    #[test]
    fn unknown_or_malformed_lines_are_errors() {
        let lines = [
            "",
            "not json",
            r#"{"type":"reply","id":1"#,
            r#"{"type":"teleport"}"#,
            r#"{"type":"call","id":1,"request":{"capture":{"op":"self_destruct"}}}"#,
            r#"{"type":"call","id":1,"request":{"op":"ptt_start"}}"#,
            r#"{"type":"call","id":-1,"request":{"capture":{"op":"ptt_start"}}}"#,
            r#"{"type":"reply","id":1,"outcome":{"status":"done"}}"#,
        ];
        for line in lines {
            assert!(decode::<ToWorker>(line).is_err(), "{line:?}");
        }
        assert!(decode::<FromWorker>(r#"{"type":"heard","heard":{"kind":"shout"}}"#).is_err());
    }

    #[test]
    fn an_undecodable_line_is_refused_only_when_it_carries_a_call_id() {
        let refused = |line: &str| {
            let error = decode::<ToWorker>(line).unwrap_err();
            refusal(line, &error)
        };
        let unknown_request =
            r#"{"type":"call","id":7,"request":{"capture":{"op":"self_destruct"}}}"#;
        assert!(matches!(
            refused(unknown_request),
            Some(FromWorker::Reply {
                id: 7,
                outcome: Outcome::Failed { .. }
            })
        ));
        assert_eq!(refused(r#"{"type":"teleport"}"#), None);
        assert_eq!(refused("not json"), None);
    }
}
