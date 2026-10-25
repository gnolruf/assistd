//! Wake-phrase status and testing. A test is a session that holds the mic,
//! one at a time, for as long as the connection that opened it stays.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex as StdMutex;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tracing::warn;

use assistd_ipc::{Event, VoiceCaptureState, WakeWordOp};
use assistd_voice::{CapturedUtterance, ListenError, SessionAudio, VoiceCapture, WakeWordSpotter};

use super::{AppState, DispatchError, send_error};

/// A session with no utterance for this long is cancelled.
const SESSION_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

/// Mic reports queued while the previous utterance is still being
/// spotted; each utterance comes with a start and an end of speech.
const AUDIO_BACKLOG: usize = 12;

const STOP_BACKLOG: usize = 1;

/// Why another request cannot use the mic while a test runs.
pub(in crate::state) const TEST_IN_PROGRESS: &str = "wake-word test in progress";

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
    Failed(String),
    CaptureStopped,
    Disconnected,
}

/// A session's event stream, stop requests, and mic reports.
struct SessionIo {
    id: String,
    tx: mpsc::Sender<Event>,
    stop: mpsc::Receiver<()>,
    audio: mpsc::Receiver<SessionAudio>,
    idle_deadline: Instant,
}

impl SessionIo {
    /// The next utterance, after reporting each start and end of speech
    /// before it. A stop request, disconnect, or the idle timeout ends the
    /// session instead.
    async fn next_utterance(&mut self) -> Result<CapturedUtterance, SessionEnd> {
        loop {
            let audio = tokio::select! {
                biased;
                () = self.tx.closed() => Err(SessionEnd::Disconnected),
                _ = self.stop.recv() => Err(SessionEnd::Done),
                audio = self.audio.recv() => audio.ok_or(SessionEnd::CaptureStopped),
                () = tokio::time::sleep_until(self.idle_deadline) => Err(SessionEnd::Failed(
                    "wake-word test cancelled after 2 minutes idle".into(),
                )),
            }?;
            self.idle_deadline = Instant::now() + SESSION_IDLE_TIMEOUT;
            match audio {
                SessionAudio::Speaking(speaking) => self.send_speaking(speaking).await,
                SessionAudio::Utterance(utterance) => return Ok(utterance),
            }
        }
    }

    async fn send_speaking(&self, speaking: bool) {
        let _ = self
            .tx
            .send(Event::WakeWordSpeech {
                id: self.id.clone(),
                speaking,
            })
            .await;
    }
}

impl AppState {
    pub(super) async fn handle_wake_word(
        self: Arc<Self>,
        id: String,
        op: WakeWordOp,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        match op {
            WakeWordOp::Status => self.handle_wake_word_status(id, tx).await,
            WakeWordOp::TestStart => self.handle_wake_word_test(id, tx).await,
            WakeWordOp::TestStop => {
                self.stop_mic_session(id, tx).await;
                Ok(())
            }
        }
    }

    /// Whether a test session holds the mic.
    pub(in crate::state) fn mic_session_busy(&self) -> bool {
        self.runtime.mic_session.lock().is_some()
    }

    async fn handle_wake_word_status(
        &self,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let capture = self.ready_capture(&id, "wake-word status", &tx).await?;
        let _ = tx
            .send(Event::WakeWordStatus {
                id: id.clone(),
                phrase: self.config.voice.wakeword.phrase.clone(),
                listening: capture.listener.is_active(),
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    async fn handle_wake_word_test(
        &self,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let capture = self.ready_capture(&id, "wake-word test", &tx).await?;
        let Some(spotter) = capture.spotter.clone() else {
            let message = "wake-word listening is off or failed to start; see the daemon log";
            send_error(&tx, id, message.into()).await;
            return Ok(());
        };
        self.run_mic_session(id, tx, &capture, |io| drive_test(io, spotter))
            .await
    }

    async fn stop_mic_session(&self, id: String, tx: mpsc::Sender<Event>) {
        let stop = self
            .runtime
            .mic_session
            .lock()
            .as_ref()
            .map(|session| session.stop.clone());
        let stopped = match stop {
            Some(stop) => stop.send(()).await.is_ok(),
            None => false,
        };
        if stopped {
            let _ = tx.send(Event::Done { id }).await;
        } else {
            send_error(&tx, id, "no wake-word test in progress".into()).await;
        }
    }

    /// Claim the mic, pause listening, and run `drive` against the captured
    /// utterances until it ends; then resume listening and send the
    /// terminal event.
    async fn run_mic_session<F, Fut>(
        &self,
        id: String,
        tx: mpsc::Sender<Event>,
        capture: &VoiceCapture,
        drive: F,
    ) -> Result<(), DispatchError>
    where
        F: FnOnce(SessionIo) -> Fut,
        Fut: Future<Output = SessionEnd>,
    {
        let Some((claim, stop)) = self.claim_mic_session() else {
            send_error(&tx, id, format!("{TEST_IN_PROGRESS}; finish it first")).await;
            return Ok(());
        };
        if self.ptt_busy(capture).await {
            send_error(
                &tx,
                id,
                "push-to-talk is recording; release it first".into(),
            )
            .await;
            return Ok(());
        }
        let resume_listening = capture.listener.is_active();
        if resume_listening && let Err(e) = capture.listener.stop().await {
            send_error(&tx, id, format!("failed to pause listening: {e}")).await;
            return Err(e.into());
        }
        let (audio_tx, audio) = mpsc::channel(AUDIO_BACKLOG);
        let io = SessionIo {
            id: id.clone(),
            tx: tx.clone(),
            stop,
            audio,
            idle_deadline: Instant::now() + SESSION_IDLE_TIMEOUT,
        };
        let (captured, end) =
            tokio::join!(capture.listener.capture_utterances(audio_tx), drive(io));
        drop(claim);
        if resume_listening && let Err(e) = capture.listener.start().await {
            warn!(target: "assistd::wakeword", "failed to resume listening: {e}");
        }
        send_session_end(&tx, id, end, captured).await;
        Ok(())
    }

    fn claim_mic_session(&self) -> Option<(MicSessionClaim<'_>, mpsc::Receiver<()>)> {
        let mut slot = self.runtime.mic_session.lock();
        if slot.is_some() {
            return None;
        }
        let (stop_tx, stop_rx) = mpsc::channel(STOP_BACKLOG);
        *slot = Some(MicSession { stop: stop_tx });
        Some((
            MicSessionClaim {
                slot: &self.runtime.mic_session,
            },
            stop_rx,
        ))
    }

    async fn ptt_busy(&self, capture: &VoiceCapture) -> bool {
        capture.input.state() != VoiceCaptureState::Idle
            || self.runtime.ptt_capture.lock().await.is_some()
    }
}

/// Report, for each utterance, whether the wake phrase was spotted in it.
async fn drive_test(mut io: SessionIo, spotter: Arc<WakeWordSpotter>) -> SessionEnd {
    loop {
        let utterance = match io.next_utterance().await {
            Ok(utterance) => utterance,
            Err(end) => return end,
        };
        let spotter = Arc::clone(&spotter);
        let matched =
            match tokio::task::spawn_blocking(move || spotter.find(&utterance.pcm).is_some()).await
            {
                Ok(matched) => matched,
                Err(e) => return SessionEnd::Failed(format!("wake-phrase spotting failed: {e}")),
            };
        let _ = io
            .tx
            .send(Event::WakeWordTest {
                id: io.id.clone(),
                matched,
            })
            .await;
    }
}

async fn send_session_end(
    tx: &mpsc::Sender<Event>,
    id: String,
    end: SessionEnd,
    captured: Result<(), ListenError>,
) {
    match end {
        SessionEnd::Done => {
            let _ = tx.send(Event::Done { id }).await;
        }
        SessionEnd::Failed(message) => send_error(tx, id, message).await,
        SessionEnd::CaptureStopped => {
            let message = match captured {
                Err(e) => format!("audio capture stopped: {e}"),
                Ok(()) => "audio capture stopped".to_string(),
            };
            send_error(tx, id, message).await;
        }
        SessionEnd::Disconnected => {}
    }
}
