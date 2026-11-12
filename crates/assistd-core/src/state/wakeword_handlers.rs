//! Wake-phrase status and testing. A test is a session that holds the mic,
//! one at a time, for as long as the connection that opened it stays.

use std::time::Duration;

use parking_lot::Mutex as StdMutex;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tracing::warn;

use assistd_ipc::{Event, EventBody, VoiceCaptureState, WakeWordOp};
use assistd_voice::{ListenError, VoiceCapture, WakeTest};

use super::voice_handlers::VoiceService;
use super::{DispatchError, FailedStep};
use crate::bus::EventSink;

/// A session with no utterance for this long is cancelled.
const SESSION_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

/// Mic reports queued while the previous one is still being sent on; each
/// utterance comes with a start and an end of speech.
const AUDIO_BACKLOG: usize = 12;

const STOP_BACKLOG: usize = 1;

/// Why a request could not have the mic, or a wake-word test ended early.
#[derive(Debug, Error)]
pub enum MicError {
    #[error("wake-word test in progress")]
    TestInProgress,

    #[error("wake-word test in progress; finish it first")]
    TestAlreadyOpen,

    #[error("no wake-word test in progress")]
    NoTest,

    #[error("push-to-talk is recording; release it first")]
    PushToTalkRecording,

    #[error("cannot start continuous listening while PTT is recording")]
    ListenDuringPushToTalk,

    #[error("wake-word test cancelled after 2 minutes idle")]
    TestIdle,

    #[error("wake-word listening is off or failed to start; see the daemon log")]
    ListeningOff,

    #[error("audio capture stopped")]
    CaptureStopped,

    #[error("audio capture stopped: {0}")]
    CaptureFailed(#[source] ListenError),
}

/// The open test session, reachable from the request that stops it.
#[derive(Debug)]
pub(in crate::state) struct MicSession {
    stop: mpsc::Sender<()>,
}

/// Frees the session slot when the session ends, however it ends.
struct MicSessionClaim<'a> {
    slot: &'a StdMutex<Option<MicSession>>,
}

impl Drop for MicSessionClaim<'_> {
    fn drop(&mut self) {
        *self.slot.lock() = None;
    }
}

/// How a session ended, before its terminal event is sent.
enum SessionEnd {
    Done,
    Idle,
    CaptureStopped,
    Disconnected,
}

/// A session's event stream, stop requests, and mic reports.
struct SessionIo {
    id: String,
    tx: EventSink,
    stop: mpsc::Receiver<()>,
    audio: mpsc::Receiver<WakeTest>,
    idle_deadline: Instant,
}

impl SessionIo {
    /// Whether the next utterance held the wake phrase, after reporting each
    /// start and end of speech before it. A stop request, disconnect, or the
    /// idle timeout ends the session instead.
    async fn next_utterance(&mut self) -> Result<bool, SessionEnd> {
        loop {
            let audio = tokio::select! {
                biased;
                () = self.tx.closed() => Err(SessionEnd::Disconnected),
                _ = self.stop.recv() => Err(SessionEnd::Done),
                audio = self.audio.recv() => audio.ok_or(SessionEnd::CaptureStopped),
                () = tokio::time::sleep_until(self.idle_deadline) => Err(SessionEnd::Idle),
            }?;
            self.idle_deadline = Instant::now() + SESSION_IDLE_TIMEOUT;
            match audio {
                WakeTest::Speaking { speaking } => self.send_speaking(speaking).await,
                WakeTest::Utterance { matched } => return Ok(matched),
            }
        }
    }

    async fn send_speaking(&self, speaking: bool) {
        let _ = self
            .tx
            .send(Event::new(
                self.id.clone(),
                EventBody::WakeWordSpeech { speaking },
            ))
            .await;
    }
}

impl VoiceService {
    pub(super) async fn handle_wake_word(
        &self,
        id: String,
        op: WakeWordOp,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        match op {
            WakeWordOp::Status => self.handle_wake_word_status(id, tx).await,
            WakeWordOp::TestStart => self.handle_wake_word_test(id, tx).await,
            WakeWordOp::TestStop => self.stop_mic_session(id, tx).await,
        }
    }

    /// Whether a test session holds the mic.
    pub(in crate::state) fn mic_session_busy(&self) -> bool {
        self.mic_session.lock().is_some()
    }

    async fn handle_wake_word_status(
        &self,
        id: String,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let capture = self.voice.capture().context("wake-word status failed")?;
        let _ = tx
            .send(Event::new(
                id.clone(),
                EventBody::WakeWordStatus {
                    phrase: self.config.voice.wakeword.phrase.to_string(),
                    listening: capture.listener.is_active(),
                },
            ))
            .await;
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
        Ok(())
    }

    async fn handle_wake_word_test(&self, id: String, tx: EventSink) -> Result<(), DispatchError> {
        let capture = self.voice.capture().context("wake-word test failed")?;
        self.run_mic_session(id, tx, &capture, drive_test).await
    }

    async fn stop_mic_session(&self, id: String, tx: EventSink) -> Result<(), DispatchError> {
        let stop = self
            .mic_session
            .lock()
            .as_ref()
            .map(|session| session.stop.clone());
        let stopped = match stop {
            Some(stop) => stop.send(()).await.is_ok(),
            None => false,
        };
        if !stopped {
            return Err(MicError::NoTest.into());
        }
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
        Ok(())
    }

    /// Claim the mic, pause listening, and run `drive` against the test's
    /// reports until it ends; then resume listening and send the
    /// terminal event.
    async fn run_mic_session<F, Fut>(
        &self,
        id: String,
        tx: EventSink,
        capture: &VoiceCapture,
        drive: F,
    ) -> Result<(), DispatchError>
    where
        F: FnOnce(SessionIo) -> Fut,
        Fut: Future<Output = SessionEnd>,
    {
        let (claim, stop) = self.claim_mic_session().ok_or(MicError::TestAlreadyOpen)?;
        if self.ptt_busy(capture).await {
            return Err(MicError::PushToTalkRecording.into());
        }
        let resume_listening = capture.listener.is_active();
        if resume_listening {
            capture
                .listener
                .stop()
                .await
                .context("failed to pause listening")?;
        }
        let (audio_tx, audio) = mpsc::channel(AUDIO_BACKLOG);
        let io = SessionIo {
            id: id.clone(),
            tx: tx.clone(),
            stop,
            audio,
            idle_deadline: Instant::now() + SESSION_IDLE_TIMEOUT,
        };
        let (captured, end) = tokio::join!(capture.listener.test_wake_word(audio_tx), drive(io));
        drop(claim);
        if resume_listening && let Err(e) = capture.listener.start().await {
            warn!(target: "assistd::wakeword", "failed to resume listening: {e}");
        }
        match end {
            SessionEnd::Done => {
                let _ = tx.send(Event::new(id, EventBody::Done)).await;
                Ok(())
            }
            SessionEnd::Disconnected => Ok(()),
            SessionEnd::Idle => Err(MicError::TestIdle.into()),
            SessionEnd::CaptureStopped => Err(capture_stop(captured).into()),
        }
    }

    fn claim_mic_session(&self) -> Option<(MicSessionClaim<'_>, mpsc::Receiver<()>)> {
        let mut slot = self.mic_session.lock();
        if slot.is_some() {
            return None;
        }
        let (stop_tx, stop_rx) = mpsc::channel(STOP_BACKLOG);
        *slot = Some(MicSession { stop: stop_tx });
        Some((
            MicSessionClaim {
                slot: &self.mic_session,
            },
            stop_rx,
        ))
    }

    async fn ptt_busy(&self, capture: &VoiceCapture) -> bool {
        capture.input.state() != VoiceCaptureState::Idle || self.ptt_capture.lock().await.is_some()
    }
}

/// Report, for each utterance, whether the wake phrase was spotted in it.
async fn drive_test(mut io: SessionIo) -> SessionEnd {
    loop {
        let matched = match io.next_utterance().await {
            Ok(matched) => matched,
            Err(end) => return end,
        };
        let _ = io
            .tx
            .send(Event::new(
                io.id.clone(),
                EventBody::WakeWordTest { matched },
            ))
            .await;
    }
}

/// Why the capture behind a test stopped, from how its listener returned.
fn capture_stop(captured: Result<(), ListenError>) -> MicError {
    match captured {
        Err(ListenError::Disabled) => MicError::ListeningOff,
        Err(e) => MicError::CaptureFailed(e),
        Ok(()) => MicError::CaptureStopped,
    }
}
