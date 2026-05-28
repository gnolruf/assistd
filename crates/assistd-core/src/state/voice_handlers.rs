//! Voice-input (PTT, listen) and voice-output (TTS) request handlers.

use super::AppState;
use anyhow::Result;
use assistd_ipc::{Event, VoiceCaptureState};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::Instrument;

impl AppState {
    pub(super) async fn handle_ptt_start(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        self.subsystems.voice_output.interrupt().await;
        if self.subsystems.listener.is_active() {
            let _ = tx
                .send(Event::Error {
                    id,
                    message: "continuous listening is active; disable it before using PTT".into(),
                })
                .await;
            return Ok(());
        }
        match self.subsystems.voice.start_recording().await {
            Ok(()) => {
                let presence = self.subsystems.presence.clone();
                let warm =
                    tokio::spawn(async move { presence.ensure_active().await }.in_current_span());
                *self.runtime.warmup_handle.lock().await = Some(warm);
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
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("ptt_start failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    pub(super) async fn handle_listen_start(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        if self.subsystems.voice.state() != VoiceCaptureState::Idle {
            let _ = tx
                .send(Event::Error {
                    id,
                    message: "cannot start continuous listening while PTT is recording".into(),
                })
                .await;
            return Ok(());
        }
        match self.subsystems.listener.start().await {
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
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("listen_start failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    pub(super) async fn handle_listen_stop(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        match self.subsystems.listener.stop().await {
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
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("listen_stop failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    pub(super) async fn handle_listen_toggle(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        if self.subsystems.listener.is_active() {
            self.handle_listen_stop(id, tx).await
        } else {
            self.handle_listen_start(id, tx).await
        }
    }

    pub(super) async fn handle_get_listen_state(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        let active = self.subsystems.listener.is_active();
        let _ = tx
            .send(Event::ListenState {
                id: id.clone(),
                active,
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    pub(super) async fn handle_voice_toggle(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        let new_state = !self.subsystems.voice_output.enabled();
        self.subsystems.voice_output.set_enabled(new_state).await;
        let _ = tx
            .send(Event::VoiceOutputState {
                id: id.clone(),
                enabled: new_state,
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    pub(super) async fn handle_voice_skip(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        self.subsystems.voice_output.skip().await;
        let _ = tx
            .send(Event::VoiceOutputState {
                id: id.clone(),
                enabled: self.subsystems.voice_output.enabled(),
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    /// Cancel the in-flight agent turn (if any) and drop queued TTS
    /// audio. Idempotent.
    pub(super) async fn handle_interrupt_turn(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        if let Some(token) = self.runtime.current_cancel.lock().await.take() {
            token.cancel();
        }
        self.subsystems.voice_output.skip().await;
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    pub(super) async fn handle_get_voice_state(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        let _ = tx
            .send(Event::VoiceOutputState {
                id: id.clone(),
                enabled: self.subsystems.voice_output.enabled(),
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    #[tracing::instrument(skip_all, fields(correlation_id = %id))]
    pub(super) async fn handle_ptt_stop(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        tracing::debug!(
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

        let warmup = self.runtime.warmup_handle.lock().await.take();
        let (text_res, warm_res) =
            tokio::join!(self.subsystems.voice.stop_and_transcribe(), async {
                match warmup {
                    Some(h) => h.await.unwrap_or_else(|e| Err(anyhow::anyhow!(e))),
                    None => Ok(()),
                }
            },);
        if let Err(e) = &warm_res {
            tracing::warn!(
                target: "assistd::state",
                error = %e,
                "presence warmup failed; query path will retry"
            );
        }
        tracing::debug!(
            target: "assistd::voice::latency",
            stage = "ensure_active_done",
            "voice latency stage"
        );
        let text = match text_res {
            Ok(t) => t,
            Err(e) => {
                let _ = tx
                    .send(Event::VoiceState {
                        id: id.clone(),
                        state: VoiceCaptureState::Idle,
                    })
                    .await;
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("ptt_stop failed: {e:#}"),
                    })
                    .await;
                return Err(e);
            }
        };

        let _ = tx
            .send(Event::VoiceState {
                id: id.clone(),
                state: VoiceCaptureState::Idle,
            })
            .await;
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

        self.handle_query(id, text, Vec::new(), tx).await
    }
}
