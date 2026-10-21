//! Wake-phrase status, enrollment, testing and reset. Enrollment and
//! testing are sessions that hold the mic, one at a time, for as long as the
//! connection that opened them stays.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex as StdMutex;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;
use tracing::warn;

use assistd_ipc::{
    EnrollEvent, Event, SampleRejection, VoiceCaptureState, WakePhraseText, WakeWordOp,
};
use assistd_voice::listen::enrollment::MIN_SAMPLES;
use assistd_voice::listen::wake_phrase::DEFAULT_WAKE_PHRASE;
use assistd_voice::{
    CapturedUtterance, Enrollment, ListenError, VoiceCapture, WakePhrase, WakeWordProfile,
};

use super::{AppState, DispatchError, send_error};

/// A session with no utterance or command for this long is cancelled.
const SESSION_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

/// Utterances captured while the previous one is still being transcribed.
const UTTERANCE_BACKLOG: usize = 4;

const CONTROL_BACKLOG: usize = 4;

/// What an open wake-word session is doing with the mic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::state) enum MicSessionKind {
    Enroll,
    Test,
}

impl MicSessionKind {
    /// Why another request cannot use the mic right now.
    pub(in crate::state) fn in_progress(self) -> &'static str {
        match self {
            MicSessionKind::Enroll => "wake-word enrollment in progress",
            MicSessionKind::Test => "wake-word test in progress",
        }
    }

    fn not_open(self) -> &'static str {
        match self {
            MicSessionKind::Enroll => "no wake-word enrollment in progress",
            MicSessionKind::Test => "no wake-word test in progress",
        }
    }
}

/// The open session, reachable from the requests that steer it.
#[derive(Debug)]
pub(in crate::state) struct MicSession {
    kind: MicSessionKind,
    control: mpsc::Sender<SessionControl>,
}

#[derive(Debug)]
enum SessionControl {
    /// Save the enrollment, answering whether it was saved.
    Finish(oneshot::Sender<Result<(), String>>),
    Stop,
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

enum SessionInput {
    Utterance(CapturedUtterance),
    Control(SessionControl),
    Ended(SessionEnd),
}

/// A session's event stream, steering requests, and captured utterances.
struct SessionIo {
    id: String,
    tx: mpsc::Sender<Event>,
    control: mpsc::Receiver<SessionControl>,
    utterances: mpsc::Receiver<CapturedUtterance>,
    idle_deadline: Instant,
}

impl SessionIo {
    /// The next thing the session must act on; disconnect and the idle
    /// timeout end it.
    async fn next_input(&mut self) -> SessionInput {
        let input = tokio::select! {
            biased;
            () = self.tx.closed() => SessionInput::Ended(SessionEnd::Disconnected),
            control = self.control.recv() => {
                control.map_or(SessionInput::Ended(SessionEnd::Done), SessionInput::Control)
            }
            utterance = self.utterances.recv() => {
                utterance.map_or(SessionInput::Ended(SessionEnd::CaptureStopped), SessionInput::Utterance)
            }
            () = tokio::time::sleep_until(self.idle_deadline) => SessionInput::Ended(
                SessionEnd::Failed("wake-word session cancelled after 2 minutes idle".into()),
            ),
        };
        self.idle_deadline = Instant::now() + SESSION_IDLE_TIMEOUT;
        input
    }

    async fn send_enrollment(&self, event: EnrollEvent) {
        let _ = self
            .tx
            .send(Event::WakeWordEnrollment {
                id: self.id.clone(),
                event,
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
            WakeWordOp::EnrollStart { phrase } => self.handle_enroll_start(id, phrase, tx).await,
            WakeWordOp::EnrollFinish => {
                self.handle_enroll_finish(id, tx).await;
                Ok(())
            }
            WakeWordOp::EnrollCancel => {
                self.stop_mic_session(id, MicSessionKind::Enroll, tx).await;
                Ok(())
            }
            WakeWordOp::TestStart => self.handle_wake_word_test(id, tx).await,
            WakeWordOp::TestStop => {
                self.stop_mic_session(id, MicSessionKind::Test, tx).await;
                Ok(())
            }
            WakeWordOp::Reset => self.handle_wake_word_reset(id, tx).await,
        }
    }

    pub(in crate::state) fn mic_session_kind(&self) -> Option<MicSessionKind> {
        self.runtime
            .mic_session
            .lock()
            .as_ref()
            .map(|session| session.kind)
    }

    async fn handle_wake_word_status(
        &self,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let capture = self.ready_capture(&id, "wake-word status", &tx).await?;
        let profile = self.load_profile(&id, &capture, &tx).await?;
        let (phrase, variants, samples) = match profile {
            Some(profile) => (profile.phrase, profile.variants, profile.samples.len()),
            None => (DEFAULT_WAKE_PHRASE.to_string(), Vec::new(), 0),
        };
        let _ = tx
            .send(Event::WakeWordStatus {
                id: id.clone(),
                phrase,
                variants,
                samples: u32::try_from(samples).unwrap_or(u32::MAX),
                listening: capture.listener.is_active(),
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    async fn handle_enroll_start(
        &self,
        id: String,
        phrase: WakePhraseText,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let capture = self.ready_capture(&id, "wake-word enrollment", &tx).await?;
        let enrollment = Enrollment::new(phrase.as_str());
        self.run_mic_session(id, tx, MicSessionKind::Enroll, &capture, |io| {
            self.drive_enrollment(io, &capture, enrollment)
        })
        .await
    }

    async fn handle_wake_word_test(
        &self,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let capture = self.ready_capture(&id, "wake-word test", &tx).await?;
        let wake_phrase = self.saved_wake_phrase(&id, &capture, &tx).await?;
        self.run_mic_session(id, tx, MicSessionKind::Test, &capture, |io| {
            drive_test(io, &capture, wake_phrase)
        })
        .await
    }

    async fn handle_enroll_finish(&self, id: String, tx: mpsc::Sender<Event>) {
        let Some(control) = self.session_control(MicSessionKind::Enroll) else {
            send_error(&tx, id, MicSessionKind::Enroll.not_open().into()).await;
            return;
        };
        let (reply_tx, reply_rx) = oneshot::channel();
        let saved = match control.send(SessionControl::Finish(reply_tx)).await {
            Ok(()) => reply_rx
                .await
                .unwrap_or_else(|_| Err("the enrollment ended before it was saved".into())),
            Err(_) => Err(MicSessionKind::Enroll.not_open().into()),
        };
        match saved {
            Ok(()) => {
                let _ = tx.send(Event::Done { id }).await;
            }
            Err(message) => send_error(&tx, id, message).await,
        }
    }

    async fn stop_mic_session(&self, id: String, kind: MicSessionKind, tx: mpsc::Sender<Event>) {
        let stopped = match self.session_control(kind) {
            Some(control) => control.send(SessionControl::Stop).await.is_ok(),
            None => false,
        };
        if stopped {
            let _ = tx.send(Event::Done { id }).await;
        } else {
            send_error(&tx, id, kind.not_open().into()).await;
        }
    }

    async fn handle_wake_word_reset(
        &self,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let capture = self.ready_capture(&id, "wake-word reset", &tx).await?;
        if let Some(kind) = self.mic_session_kind() {
            send_error(&tx, id, kind.in_progress().into()).await;
            return Ok(());
        }
        if let Err(e) = capture.profiles.reset().await {
            send_error(&tx, id, format!("wake-word reset failed: {e}")).await;
            return Err(e.into());
        }
        let builtin = WakePhrase::builtin(self.config.voice.wakeword.match_similarity);
        if let Err(e) = apply_wake_phrase(&capture, builtin).await {
            send_error(&tx, id, format!("wake-word reset failed: {e}")).await;
            return Err(e.into());
        }
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    /// Claim the mic for a session of `kind`, pause listening, and run
    /// `drive` against the captured utterances until it ends; then resume
    /// listening and send the terminal event.
    async fn run_mic_session<F, Fut>(
        &self,
        id: String,
        tx: mpsc::Sender<Event>,
        kind: MicSessionKind,
        capture: &VoiceCapture,
        drive: F,
    ) -> Result<(), DispatchError>
    where
        F: FnOnce(SessionIo) -> Fut,
        Fut: Future<Output = SessionEnd>,
    {
        let Some((claim, control)) = self.claim_mic_session(kind) else {
            let busy = self.mic_session_kind().unwrap_or(kind);
            send_error(&tx, id, format!("{}; finish it first", busy.in_progress())).await;
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
        let (utterance_tx, utterances) = mpsc::channel(UTTERANCE_BACKLOG);
        let io = SessionIo {
            id: id.clone(),
            tx: tx.clone(),
            control,
            utterances,
            idle_deadline: Instant::now() + SESSION_IDLE_TIMEOUT,
        };
        let (captured, end) =
            tokio::join!(capture.listener.capture_utterances(utterance_tx), drive(io));
        drop(claim);
        if resume_listening && let Err(e) = capture.listener.start().await {
            warn!(target: "assistd::wakeword", "failed to resume listening: {e}");
        }
        send_session_end(&tx, id, end, captured).await;
        Ok(())
    }

    fn claim_mic_session(
        &self,
        kind: MicSessionKind,
    ) -> Option<(MicSessionClaim<'_>, mpsc::Receiver<SessionControl>)> {
        let mut slot = self.runtime.mic_session.lock();
        if slot.is_some() {
            return None;
        }
        let (control_tx, control_rx) = mpsc::channel(CONTROL_BACKLOG);
        *slot = Some(MicSession {
            kind,
            control: control_tx,
        });
        Some((
            MicSessionClaim {
                slot: &self.runtime.mic_session,
            },
            control_rx,
        ))
    }

    fn session_control(&self, kind: MicSessionKind) -> Option<mpsc::Sender<SessionControl>> {
        self.runtime
            .mic_session
            .lock()
            .as_ref()
            .filter(|session| session.kind == kind)
            .map(|session| session.control.clone())
    }

    async fn ptt_busy(&self, capture: &VoiceCapture) -> bool {
        capture.input.state() != VoiceCaptureState::Idle
            || self.runtime.ptt_capture.lock().await.is_some()
    }

    async fn drive_enrollment(
        &self,
        mut io: SessionIo,
        capture: &VoiceCapture,
        mut enrollment: Enrollment,
    ) -> SessionEnd {
        io.send_enrollment(enrollment.prompt()).await;
        loop {
            match io.next_input().await {
                SessionInput::Utterance(utterance) => {
                    if let Err(end) = enroll_sample(&io, capture, &mut enrollment, utterance).await
                    {
                        return end;
                    }
                }
                SessionInput::Control(SessionControl::Finish(reply)) if !enrollment.is_ready() => {
                    let _ = reply.send(Err(format!(
                        "at least {MIN_SAMPLES} accepted samples are needed before saving"
                    )));
                }
                SessionInput::Control(SessionControl::Finish(reply)) => {
                    return self.save_enrollment(&io, capture, enrollment, reply).await;
                }
                SessionInput::Control(SessionControl::Stop) => return SessionEnd::Done,
                SessionInput::Ended(end) => return end,
            }
        }
    }

    async fn save_enrollment(
        &self,
        io: &SessionIo,
        capture: &VoiceCapture,
        enrollment: Enrollment,
        reply: oneshot::Sender<Result<(), String>>,
    ) -> SessionEnd {
        let profile = match capture.profiles.save(enrollment).await {
            Ok(profile) => profile,
            Err(e) => {
                let message = format!("failed to save the wake-word profile: {e}");
                let _ = reply.send(Err(message.clone()));
                return SessionEnd::Failed(message);
            }
        };
        capture
            .listener
            .set_wake_phrase(profile.wake_phrase(self.config.voice.wakeword.match_similarity));
        let _ = reply.send(Ok(()));
        io.send_enrollment(EnrollEvent::Saved {
            phrase: profile.phrase,
            variants: profile.variants,
        })
        .await;
        SessionEnd::Done
    }

    async fn saved_wake_phrase(
        &self,
        id: &str,
        capture: &VoiceCapture,
        tx: &mpsc::Sender<Event>,
    ) -> Result<WakePhrase, DispatchError> {
        let similarity = self.config.voice.wakeword.match_similarity;
        let profile = self.load_profile(id, capture, tx).await?;
        Ok(profile.map_or_else(
            || WakePhrase::builtin(similarity),
            |profile| profile.wake_phrase(similarity),
        ))
    }

    async fn load_profile(
        &self,
        id: &str,
        capture: &VoiceCapture,
        tx: &mpsc::Sender<Event>,
    ) -> Result<Option<WakeWordProfile>, DispatchError> {
        match capture.profiles.load().await {
            Ok(profile) => Ok(profile),
            Err(e) => {
                send_error(tx, id.to_string(), e.to_string()).await;
                Err(e.into())
            }
        }
    }
}

/// Screen, transcribe and offer one utterance to `enrollment`, reporting
/// the outcome and the next prompt. A transcription failure ends the session.
async fn enroll_sample(
    io: &SessionIo,
    capture: &VoiceCapture,
    enrollment: &mut Enrollment,
    utterance: CapturedUtterance,
) -> Result<(), SessionEnd> {
    let outcome = match Enrollment::screen(&utterance) {
        Err(reason) => EnrollEvent::SampleRejected {
            reason,
            heard: None,
        },
        Ok(()) => {
            let heard = capture
                .transcriber
                .transcribe(&utterance.pcm, Some(enrollment.phrase()))
                .await
                .map_err(|e| SessionEnd::Failed(format!("transcription failed: {e}")))?;
            offer_sample(enrollment, utterance.pcm, heard.trim())
        }
    };
    io.send_enrollment(outcome).await;
    io.send_enrollment(enrollment.prompt()).await;
    Ok(())
}

fn offer_sample(enrollment: &mut Enrollment, pcm: Vec<i16>, heard: &str) -> EnrollEvent {
    match enrollment.accept(pcm, heard) {
        Ok(index) => EnrollEvent::SampleAccepted {
            index,
            heard: heard.to_string(),
        },
        Err(reason) => EnrollEvent::SampleRejected {
            reason,
            heard: (reason == SampleRejection::DidNotMatch && !heard.is_empty())
                .then(|| heard.to_string()),
        },
    }
}

/// Report what each utterance was heard as and whether `wake_phrase` matched it.
async fn drive_test(
    mut io: SessionIo,
    capture: &VoiceCapture,
    wake_phrase: WakePhrase,
) -> SessionEnd {
    loop {
        match io.next_input().await {
            SessionInput::Utterance(utterance) => {
                let heard = match capture
                    .transcriber
                    .transcribe(&utterance.pcm, Some(wake_phrase.phrase()))
                    .await
                {
                    Ok(heard) => heard.trim().to_string(),
                    Err(e) => return SessionEnd::Failed(format!("transcription failed: {e}")),
                };
                if heard.is_empty() {
                    continue;
                }
                let matched = wake_phrase.match_prefix(&heard).is_some();
                let _ = io
                    .tx
                    .send(Event::WakeWordTest {
                        id: io.id.clone(),
                        heard,
                        matched,
                    })
                    .await;
            }
            SessionInput::Control(SessionControl::Finish(reply)) => {
                let _ = reply.send(Err(MicSessionKind::Enroll.not_open().into()));
            }
            SessionInput::Control(SessionControl::Stop) => return SessionEnd::Done,
            SessionInput::Ended(end) => return end,
        }
    }
}

/// Make `capture`'s listener gate on `phrase`, restarting it if it is on.
async fn apply_wake_phrase(capture: &VoiceCapture, phrase: WakePhrase) -> Result<(), ListenError> {
    capture.listener.set_wake_phrase(phrase);
    if capture.listener.is_active() {
        capture.listener.stop().await?;
        capture.listener.start().await?;
    }
    Ok(())
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
