//! Handlers for the voice-input (PTT, continuous listen) and
//! voice-output (TTS toggle/skip) variants of `Request`.

use super::AppState;
use anyhow::Result;
use assistd_ipc::{Event, VoiceCaptureState};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::Instrument;

impl AppState {
    /// Begin a push-to-talk recording. Returns immediately on success
    /// so the CLI client exits cleanly; the daemon holds the capture
    /// session open until a matching `PttStop` arrives on a separate
    /// connection. Rejects when continuous listening currently owns
    /// the mic; the two modes can't share one cpal stream.
    pub(super) async fn handle_ptt_start(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        // Barge-in: stop any in-flight TTS playback before opening the
        // mic. Fire unconditionally; even if recording is rejected
        // below, the user signaled "shut up", which we should honor.
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
                // Kick off the presence wake in parallel with capture so a
                // Drowsy → Active model load runs during Whisper inference.
                // `ensure_active` is idempotent; the second call from
                // `acquire_request_guard` short-circuits to a single RwLock read.
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

    /// Enable continuous listening. Rejects if PTT currently holds the
    /// mic; the two modes are mutually exclusive because cpal cannot
    /// share one input stream.
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

    /// Disable continuous listening. Idempotent.
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

    /// Flip continuous listening state. Routes to start or stop
    /// depending on the current value of `listener.is_active()`.
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

    /// Report whether continuous listening is currently active.
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

    /// Flip TTS on/off at runtime. Off cancels currently-queued audio;
    /// on is a pure flag flip. Subsequent sentences from the in-flight
    /// query speak again as soon as the toggle returns. Emits the
    /// post-toggle `VoiceOutputState` + `Done`.
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

    /// Abort the current TTS response: drops queued audio and any
    /// pending sentences for active speech workers. Does not start
    /// recording. Emits `VoiceOutputState` + `Done`; the enabled flag
    /// is unchanged.
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

    /// Report whether TTS is currently enabled.
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

    /// End the push-to-talk recording, transcribe, and (if the
    /// transcription has content) dispatch it internally as a Query
    /// so the streaming LLM response flows back on the same
    /// connection before the terminal `Done`.
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

        // The warmup was spawned at PTT-start; joining it concurrently with
        // Whisper means a Drowsy → Active model load runs during transcription
        // rather than serially after it. A failed warmup is non-fatal; the
        // request guard in `handle_query` will retry the wake.
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
            // VAD trimmed to silence; don't dispatch an empty user message.
            let _ = tx.send(Event::Done { id }).await;
            return Ok(());
        }

        self.handle_query(id, text, Vec::new(), tx).await
    }
}
