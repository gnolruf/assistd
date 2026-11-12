//! Voice-input (PTT, listen) and voice-output (TTS) request handlers.

use std::sync::Arc;

use parking_lot::Mutex as StdMutex;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{Instrument, debug, warn};

use assistd_config::Config;
use assistd_ipc::{Event, EventBody, ListenPhase, PlanMode, VoiceCaptureState};
use assistd_voice::{VoiceCapture, VoiceManager};

use super::query::{Query, TurnOrigin, finish_interrupted_before_start};
use super::turn_admission::TurnAdmission;
use super::wakeword_handlers::{MicSession, TEST_IN_PROGRESS};
use super::{DispatchError, send_error};
use crate::bus::EventSink;
use crate::presence::PresenceManager;
use crate::recovery::{Component, spawn_supervised};

/// What a push-to-talk press hands to its release.
#[derive(Debug)]
pub(super) struct PttCapture {
    /// Presence warmup started by the press.
    warmup: JoinHandle<()>,
    /// The turn the release will run, so an interrupt mid-capture drops it.
    cancel: CancellationToken,
    /// Listening was handed over for the press and resumes on release.
    resume_listening: bool,
}

/// A transcribed push-to-talk press, ready to run as a turn unless `cancel` fires first.
#[derive(Debug)]
pub(super) struct HeardPrompt {
    pub(super) query: Query,
    pub(super) cancel: CancellationToken,
}

/// Handles push-to-talk, listening, wake-word and speech requests; owns the press
/// awaiting its release and the open wake-word test session.
#[derive(Debug)]
pub struct VoiceService {
    pub(super) config: Arc<Config>,
    pub(super) voice: Arc<VoiceManager>,
    presence: Arc<PresenceManager>,
    admission: Arc<TurnAdmission>,
    /// The push-to-talk press awaiting its release.
    pub(super) ptt_capture: Mutex<Option<PttCapture>>,
    /// The open wake-word test session, which holds the mic.
    pub(super) mic_session: StdMutex<Option<MicSession>>,
}

impl VoiceService {
    pub(super) fn new(
        config: Arc<Config>,
        voice: Arc<VoiceManager>,
        presence: Arc<PresenceManager>,
        admission: Arc<TurnAdmission>,
    ) -> Self {
        Self {
            config,
            voice,
            presence,
            admission,
            ptt_capture: Mutex::new(None),
            mic_session: StdMutex::new(None),
        }
    }

    /// Holds the capture slot until the mic is open, so a `ptt_stop` sent
    /// while this press is still starting waits for it. A repeat while a
    /// press is open, as key autorepeat sends, leaves that press alone.
    pub(super) async fn handle_ptt_start(
        &self,
        id: String,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let mut ptt_capture = self.ptt_capture.lock().await;
        if ptt_capture.is_none() {
            match self.open_ptt_capture(&id, &tx).await? {
                Some(capture) => *ptt_capture = Some(capture),
                None => return Ok(()),
            }
        }
        drop(ptt_capture);
        let _ = tx
            .send(Event::new(
                id.clone(),
                EventBody::VoiceState {
                    state: VoiceCaptureState::Recording,
                },
            ))
            .await;
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
        Ok(())
    }

    /// Stop speech and open the mic for a press; `None`, already reported,
    /// when a wake-word test holds the mic.
    async fn open_ptt_capture(
        &self,
        id: &str,
        tx: &EventSink,
    ) -> Result<Option<PttCapture>, DispatchError> {
        self.voice.speech().interrupt().await;
        let capture = self.ready_capture(id, "ptt_start", tx).await?;
        if self.mic_session_busy() {
            send_error(tx, id.to_string(), TEST_IN_PROGRESS.into()).await;
            return Ok(None);
        }
        if let Err(e) = capture.input.start_recording().await {
            send_error(tx, id.to_string(), format!("ptt_start failed: {e}")).await;
            return Err(e.into());
        }
        let resume_listening = take_mic_from_listening(&capture).await;
        Ok(Some(PttCapture {
            warmup: self.spawn_presence_warmup(),
            cancel: self.admission.turn_cancellation(),
            resume_listening,
        }))
    }

    /// Transcribe the press being released; `None`, with the request already answered,
    /// when it heard nothing or was interrupted.
    #[tracing::instrument(skip_all, fields(correlation_id = %id))]
    pub(super) async fn handle_ptt_stop(
        &self,
        id: String,
        tx: &EventSink,
    ) -> Result<Option<HeardPrompt>, DispatchError> {
        debug!(
            target: "assistd::voice::latency",
            stage = "audio_capture_stop",
            "voice latency stage"
        );
        let _ = tx
            .send(Event::new(
                id.clone(),
                EventBody::VoiceState {
                    state: VoiceCaptureState::Transcribing,
                },
            ))
            .await;

        let (warmup, cancel, resume_listening) = match self.ptt_capture.lock().await.take() {
            Some(PttCapture {
                warmup,
                cancel,
                resume_listening,
            }) => (Some(warmup), cancel, resume_listening),
            None => (None, self.admission.turn_cancellation(), false),
        };
        let transcription = self.transcribe_while_warming(warmup, &cancel).await;
        if resume_listening {
            self.resume_listening_after_ptt().await;
        }
        debug!(
            target: "assistd::voice::latency",
            stage = "ensure_active_done",
            "voice latency stage"
        );
        let text = match transcription {
            Ok(text) => text,
            Err(e) => {
                let _ = tx
                    .send(Event::new(
                        id.clone(),
                        EventBody::VoiceState {
                            state: VoiceCaptureState::Idle,
                        },
                    ))
                    .await;
                send_error(tx, id, format!("ptt_stop failed: {e}")).await;
                return Err(e);
            }
        };

        let _ = tx
            .send(Event::new(
                id.clone(),
                EventBody::VoiceState {
                    state: VoiceCaptureState::Idle,
                },
            ))
            .await;
        if cancel.is_cancelled() {
            finish_interrupted_before_start(tx, id).await?;
            return Ok(None);
        }
        let _ = tx
            .send(Event::new(
                id.clone(),
                EventBody::Transcription { text: text.clone() },
            ))
            .await;

        if text.trim().is_empty() {
            let _ = tx.send(Event::new(id, EventBody::Done)).await;
            return Ok(None);
        }

        let query = Query {
            id,
            text,
            attachments: Vec::new(),
            origin: TurnOrigin::Voice,
            plan: PlanMode::Off,
        };
        Ok(Some(HeardPrompt { query, cancel }))
    }

    async fn transcribe_while_warming(
        &self,
        warmup: Option<JoinHandle<()>>,
        cancel: &CancellationToken,
    ) -> Result<String, DispatchError> {
        let capture = self.voice.capture()?;
        let warmed_or_cancelled = async {
            if let Some(warmup) = warmup {
                tokio::select! {
                    biased;
                    () = cancel.cancelled() => {}
                    _ = warmup => {}
                }
            }
        };
        let (transcription, ()) =
            tokio::join!(capture.input.stop_and_transcribe(), warmed_or_cancelled);
        Ok(transcription?)
    }

    async fn resume_listening_after_ptt(&self) {
        let resumed = match self.voice.capture() {
            Ok(capture) => capture.listener.start().await,
            Err(e) => {
                warn!(target: "assistd::state", "cannot resume listening after PTT: {e}");
                return;
            }
        };
        if let Err(e) = resumed {
            warn!(target: "assistd::state", "failed to resume listening after PTT: {e}");
        }
    }

    /// Whether a press holds the mic that listening will get back on release.
    async fn listening_paused_for_ptt(&self) -> bool {
        self.ptt_capture
            .lock()
            .await
            .as_ref()
            .is_some_and(|capture| capture.resume_listening)
    }

    fn spawn_presence_warmup(&self) -> JoinHandle<()> {
        let presence = self.presence.clone();
        spawn_supervised(
            "ptt_warmup",
            Component::Llm,
            async move {
                if let Err(e) = presence.ensure_active().await {
                    warn!(
                        target: "assistd::state",
                        error = %e,
                        "presence warmup failed; query path will retry"
                    );
                }
            }
            .in_current_span(),
        )
    }

    pub(super) async fn handle_listen_start(
        &self,
        id: String,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let capture = self.ready_capture(&id, "listen_start", &tx).await?;
        if self.mic_session_busy() {
            send_error(&tx, id, TEST_IN_PROGRESS.into()).await;
            return Ok(());
        }
        if capture.input.state() != VoiceCaptureState::Idle {
            send_error(
                &tx,
                id,
                "cannot start continuous listening while PTT is recording".into(),
            )
            .await;
            return Ok(());
        }
        match capture.listener.start().await {
            Ok(()) => {
                let _ = tx
                    .send(Event::new(
                        id.clone(),
                        EventBody::ListenState {
                            phase: self.voice.listen_phase(),
                        },
                    ))
                    .await;
                let _ = tx.send(Event::new(id, EventBody::Done)).await;
                Ok(())
            }
            Err(e) => {
                send_error(&tx, id, format!("listen_start failed: {e}")).await;
                Err(e.into())
            }
        }
    }

    pub(super) async fn handle_listen_stop(
        &self,
        id: String,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        if let Some(capture) = self.ptt_capture.lock().await.as_mut() {
            capture.resume_listening = false;
        }
        let stopped = match self.voice.capture() {
            Ok(capture) => capture.listener.stop().await,
            Err(_) => Ok(()),
        };
        match stopped {
            Ok(()) => {
                let _ = tx
                    .send(Event::new(
                        id.clone(),
                        EventBody::ListenState {
                            phase: ListenPhase::Off,
                        },
                    ))
                    .await;
                let _ = tx.send(Event::new(id, EventBody::Done)).await;
                Ok(())
            }
            Err(e) => {
                send_error(&tx, id, format!("listen_stop failed: {e}")).await;
                Err(e.into())
            }
        }
    }

    pub(super) async fn handle_listen_toggle(
        &self,
        id: String,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        if self.voice.listening() || self.listening_paused_for_ptt().await {
            self.handle_listen_stop(id, tx).await
        } else {
            self.handle_listen_start(id, tx).await
        }
    }

    pub(super) async fn handle_get_listen_state(&self, id: String, tx: EventSink) {
        let phase = self.voice.listen_phase();
        let _ = tx
            .send(Event::new(id.clone(), EventBody::ListenState { phase }))
            .await;
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
    }

    pub(super) async fn handle_voice_toggle(&self, id: String, tx: EventSink) {
        let speech = self.voice.speech();
        let new_state = !speech.enabled();
        speech.set_enabled(new_state).await;
        let _ = tx
            .send(Event::new(
                id.clone(),
                EventBody::VoiceOutputState { enabled: new_state },
            ))
            .await;
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
    }

    pub(super) async fn handle_voice_skip(&self, id: String, tx: EventSink) {
        let speech = self.voice.speech();
        speech.skip().await;
        let _ = tx
            .send(Event::new(
                id.clone(),
                EventBody::VoiceOutputState {
                    enabled: speech.enabled(),
                },
            ))
            .await;
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
    }

    /// Cancel every agent turn already received, including ones still
    /// waking the model or queued, and drop queued TTS audio. Idempotent.
    pub async fn interrupt_turns(&self) {
        self.admission.interrupt_turns();
        self.voice.speech().skip().await;
    }

    pub(super) async fn handle_interrupt_turn(&self, id: String, tx: EventSink) {
        self.interrupt_turns().await;
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
    }

    pub(super) async fn handle_get_voice_state(&self, id: String, tx: EventSink) {
        let voice = &self.voice;
        let _ = tx
            .send(Event::new(
                id.clone(),
                EventBody::VoiceOutputState {
                    enabled: voice.speech().enabled(),
                },
            ))
            .await;
        for (component, state) in voice.readiness() {
            let _ = tx
                .send(Event::new(
                    id.clone(),
                    EventBody::Readiness { component, state },
                ))
                .await;
        }
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
    }

    /// The capture handles, or an error event on `tx` saying why `request`
    /// cannot use them.
    pub(super) async fn ready_capture(
        &self,
        id: &str,
        request: &str,
        tx: &EventSink,
    ) -> Result<VoiceCapture, DispatchError> {
        match self.voice.capture() {
            Ok(capture) => Ok(capture),
            Err(e) => {
                send_error(tx, id.to_string(), format!("{request} failed: {e}")).await;
                Err(e.into())
            }
        }
    }
}

/// Hand the mic from listening to a press that just started recording;
/// true when listening was on and should resume on release.
async fn take_mic_from_listening(capture: &VoiceCapture) -> bool {
    if !capture.listener.is_active() {
        return false;
    }
    if let Err(e) = capture.listener.stop_for_push_to_talk().await {
        warn!(target: "assistd::state", "failed to pause listening for PTT: {e}");
    }
    true
}
