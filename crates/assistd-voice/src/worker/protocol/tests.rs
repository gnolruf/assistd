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
        ToWorker::Gpu {
            presence_active: false,
            foreign_busy: true,
        },
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
