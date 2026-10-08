//! Voice-input (PTT, listen) and voice-output (TTS) request handlers.

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{Instrument, debug, warn};

use assistd_ipc::{Event, VoiceCaptureState};
use assistd_voice::VoiceCapture;

use super::query::{TurnOrigin, finish_interrupted_before_start};
use super::runtime::PttCapture;
use super::{AppState, DispatchError, send_error};
use crate::recovery::{Component, spawn_supervised};

impl AppState {
    pub(super) async fn handle_ptt_start(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        self.subsystems.voice.speech().interrupt().await;
        let capture = self.ready_capture(&id, "ptt_start", &tx).await?;
        if capture.listener.is_active() {
            send_error(
                &tx,
                id,
                "continuous listening is active; disable it before using PTT".into(),
            )
            .await;
            return Ok(());
        }
        match capture.input.start_recording().await {
            Ok(()) => {
                let capture = PttCapture {
                    warmup: self.spawn_presence_warmup(),
                    cancel: self.runtime.turn_cancellation(),
                };
                *self.runtime.ptt_capture.lock().await = Some(capture);
                let _ = tx
                    .send(Event::VoiceState {
                        id: id.clone(),
                        state: VoiceCaptureState::Recording,
                    })
                    .await;
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                send_error(&tx, id, format!("ptt_start failed: {e}")).await;
                Err(e.into())
            }
        }
    }

    #[tracing::instrument(skip_all, fields(correlation_id = %id))]
    pub(super) async fn handle_ptt_stop(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        debug!(
            target: "assistd::voice::latency",
            stage = "audio_capture_stop",
            "voice latency stage"
        );
        let _ = tx
            .send(Event::VoiceState {
                id: id.clone(),
                state: VoiceCaptureState::Transcribing,
            })
            .await;

        let (warmup, cancel) = match self.runtime.ptt_capture.lock().await.take() {
            Some(PttCapture { warmup, cancel }) => (Some(warmup), cancel),
            None => (None, self.runtime.turn_cancellation()),
        };
        let transcription = self.transcribe_while_warming(warmup, &cancel).await;
        debug!(
            target: "assistd::voice::latency",
            stage = "ensure_active_done",
            "voice latency stage"
        );
        let text = match transcription {
            Ok(text) => text,
            Err(e) => {
                let _ = tx
                    .send(Event::VoiceState {
                        id: id.clone(),
                        state: VoiceCaptureState::Idle,
                    })
                    .await;
                send_error(&tx, id, format!("ptt_stop failed: {e}")).await;
                return Err(e);
            }
        };

        let _ = tx
            .send(Event::VoiceState {
                id: id.clone(),
                state: VoiceCaptureState::Idle,
            })
            .await;
        if cancel.is_cancelled() {
            return finish_interrupted_before_start(&tx, id).await;
        }
        let _ = tx
            .send(Event::Transcription {
                id: id.clone(),
                text: text.clone(),
            })
            .await;

        if text.trim().is_empty() {
            let _ = tx.send(Event::Done { id }).await;
            return Ok(());
        }

        self.run_query(id, text, Vec::new(), TurnOrigin::Voice, tx, cancel)
            .await
    }

    async fn transcribe_while_warming(
        &self,
        warmup: Option<JoinHandle<()>>,
        cancel: &CancellationToken,
    ) -> Result<String, DispatchError> {
        let capture = self.subsystems.voice.capture()?;
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

    fn spawn_presence_warmup(&self) -> JoinHandle<()> {
        let presence = self.subsystems.presence.clone();
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
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let capture = self.ready_capture(&id, "listen_start", &tx).await?;
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
                    .send(Event::ListenState {
                        id: id.clone(),
                        active: true,
                    })
                    .await;
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                send_error(&tx, id, format!("listen_start failed: {e}")).await;
                Err(e.into())
            }
        }
    }

    pub(super) async fn handle_listen_stop(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let stopped = match self.subsystems.voice.capture() {
            Ok(capture) => capture.listener.stop().await,
            Err(_) => Ok(()),
        };
        match stopped {
            Ok(()) => {
                let _ = tx
                    .send(Event::ListenState {
                        id: id.clone(),
                        active: false,
                    })
                    .await;
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                send_error(&tx, id, format!("listen_stop failed: {e}")).await;
                Err(e.into())
            }
        }
    }

    pub(super) async fn handle_listen_toggle(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        if self.subsystems.voice.listening() {
            self.handle_listen_stop(id, tx).await
        } else {
            self.handle_listen_start(id, tx).await
        }
    }

    pub(super) async fn handle_get_listen_state(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) {
        let active = self.subsystems.voice.listening();
        let _ = tx
            .send(Event::ListenState {
                id: id.clone(),
                active,
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
    }

    pub(super) async fn handle_voice_toggle(self: Arc<Self>, id: String, tx: mpsc::Sender<Event>) {
        let speech = self.subsystems.voice.speech();
        let new_state = !speech.enabled();
        speech.set_enabled(new_state).await;
        let _ = tx
            .send(Event::VoiceOutputState {
                id: id.clone(),
                enabled: new_state,
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
    }

    pub(super) async fn handle_voice_skip(self: Arc<Self>, id: String, tx: mpsc::Sender<Event>) {
        let speech = self.subsystems.voice.speech();
        speech.skip().await;
        let _ = tx
            .send(Event::VoiceOutputState {
                id: id.clone(),
                enabled: speech.enabled(),
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
    }

    /// Cancel every agent turn already received, including ones still
    /// waking the model or queued, and drop queued TTS audio. Idempotent.
    pub(super) async fn handle_interrupt_turn(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) {
        self.runtime.interrupt_turns();
        self.subsystems.voice.speech().skip().await;
        let _ = tx.send(Event::Done { id }).await;
    }

    pub(super) async fn handle_get_voice_state(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) {
        let voice = &self.subsystems.voice;
        let _ = tx
            .send(Event::VoiceOutputState {
                id: id.clone(),
                enabled: voice.speech().enabled(),
            })
            .await;
        let _ = tx.send(voice.readiness_event(id.clone())).await;
        let _ = tx.send(Event::Done { id }).await;
    }

    /// The capture handles, or an error event on `tx` saying why `request`
    /// cannot use them.
    async fn ready_capture(
        &self,
        id: &str,
        request: &str,
        tx: &mpsc::Sender<Event>,
    ) -> Result<VoiceCapture, DispatchError> {
        match self.subsystems.voice.capture() {
            Ok(capture) => Ok(capture),
            Err(e) => {
                send_error(tx, id.to_string(), format!("{request} failed: {e}")).await;
                Err(e.into())
            }
        }
    }
}
