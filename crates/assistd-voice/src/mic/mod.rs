//! Microphone capture for push-to-talk voice input.
//!
//! `MicVoiceInput` wires cpal (cross-platform audio I/O), a lock-free
//! SPSC ring buffer (`ringbuf`), and a consumer task that resamples and
//! converts PCM for the existing `WhisperTranscriber`. The three pieces
//! are split across submodules so the audio-thread hot path and the
//! off-thread drain can be read independently.
//!
//! cpal's `Stream` is `!Send` on ALSA, so the stream lives entirely
//! inside a single `spawn_blocking` worker. The outer `MicVoiceInput`
//! holds only the atomics and the `JoinHandle` — all trivially `Send`.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, anyhow};
use assistd_config::VoiceConfig;
use async_trait::async_trait;
use thiserror::Error;
use tokio::sync::{Mutex, watch};
use tokio::task::JoinHandle;
use tracing::{info, warn};

use crate::VoiceCaptureState;
use crate::VoiceInput;
use crate::transcribe::{Transcriber, TranscriptionError};
use crate::whisper::WhisperTranscriberBuilder;

pub mod capture;
pub mod consumer;

pub use capture::AudioCaptureError;

/// Errors surfaced by [`MicVoiceInput`].
#[derive(Debug, Error)]
pub enum VoiceInputError {
    #[error("audio capture error: {0}")]
    Capture(#[from] AudioCaptureError),
    #[error("transcription error: {0}")]
    Transcription(#[from] TranscriptionError),
    #[error("capture task panicked: {0}")]
    ConsumerPanic(String),
}

/// Push-to-talk voice input backed by cpal + a [`Transcriber`] (typically
/// the daemon's [`crate::QueuedTranscriber`] wrapping whisper-rs).
///
/// Construction builds the whisper context (downloads models on first
/// use) but does *not* open the audio device — that happens inside
/// [`start_recording`](VoiceInput::start_recording) so a headless CI
/// server without a mic still builds the daemon successfully.
pub struct MicVoiceInput {
    transcriber: Arc<dyn Transcriber>,
    mic_device: Option<String>,
    max_recording_secs: u32,
    state_tx: watch::Sender<VoiceCaptureState>,
    /// Monotonic counter so a stale transcription from an aborted PTT
    /// press cannot clobber the state of a newer press. Incremented in
    /// `start_recording`; checked in the state-forwarder before each
    /// forwarded transition. Shared in an `Arc` with the forwarder task.
    active_session_id: Arc<AtomicU64>,
    inner: Arc<Mutex<InnerState>>,
}

struct InnerState {
    /// Handles for the currently active capture session, `Some`
    /// between `start_recording` and `stop_and_transcribe`.
    session: Option<capture::CaptureSession>,
    /// JoinHandle for the active state-forwarder task, if any. Kept so
    /// `stop_and_transcribe` can let it exit cleanly without leaking a
    /// subscriber.
    forwarder: Option<JoinHandle<()>>,
}

impl MicVoiceInput {
    /// Build a voice input from the user's config. This async step
    /// downloads the whisper/VAD models on first use and probes GPU
    /// availability, but does NOT enumerate or open the audio device.
    /// The returned input is backed by a bare [`crate::WhisperTranscriber`]
    /// (no queueing, no CPU fallback) — daemon code that wants the
    /// queue/fallback behavior should build a
    /// [`crate::QueuedTranscriber`] and pass it to [`Self::new`].
    pub async fn from_config(cfg: &VoiceConfig) -> Result<Self, VoiceInputError> {
        let transcriber = WhisperTranscriberBuilder::from_config(&cfg.transcription)
            .build()
            .await?;
        Ok(Self::new(
            Arc::new(transcriber),
            cfg.mic_device.clone(),
            cfg.max_recording_secs.max(1),
        ))
    }

    pub fn new(
        transcriber: Arc<dyn Transcriber>,
        mic_device: Option<String>,
        max_recording_secs: u32,
    ) -> Self {
        let (state_tx, _) = watch::channel(VoiceCaptureState::Idle);
        Self {
            transcriber,
            mic_device,
            max_recording_secs,
            state_tx,
            active_session_id: Arc::new(AtomicU64::new(0)),
            inner: Arc::new(Mutex::new(InnerState {
                session: None,
                forwarder: None,
            })),
        }
    }

    /// Send a terminal `Idle` and join the state-forwarder from the
    /// current session so a stale Queued/Transcribing publish from a
    /// still-running transcription (e.g. on a VoiceInputError that
    /// returned before the transcriber finished) cannot overwrite the
    /// final Idle.
    async fn cleanup_forwarder_and_idle(&self, forwarder: Option<JoinHandle<()>>) {
        if let Some(h) = forwarder {
            h.abort();
            let _ = h.await;
        }
        let _ = self.state_tx.send(VoiceCaptureState::Idle);
    }
}

#[async_trait]
impl VoiceInput for MicVoiceInput {
    async fn start_recording(&self) -> Result<()> {
        let mut inner = self.inner.lock().await;
        if inner.session.is_some() {
            // Idempotent re-entry: already recording. A dropped
            // Released event would otherwise leave the user unable to
            // stop — let the first active recording carry on.
            return Ok(());
        }

        let session_id = self.active_session_id.fetch_add(1, Ordering::SeqCst) + 1;

        let session = capture::start(self.mic_device.as_deref(), self.max_recording_secs);
        inner.session = Some(session);

        // Spawn a state-forwarder subscribing to the transcriber's
        // internal state stream (if it exposes one). The forwarder
        // relays Queued / Transcribing transitions into our own
        // state_tx so TUI clients see them. It gates on `session_id`
        // so a late transition from an aborted PTT press cannot
        // clobber a fresh Recording state.
        if let Some(mut rx) = self.transcriber.subscribe_state() {
            let state_tx = self.state_tx.clone();
            let session_id_at_spawn = session_id;
            let counter = Arc::clone(&self.active_session_id);
            let handle = tokio::spawn(async move {
                loop {
                    if rx.changed().await.is_err() {
                        return;
                    }
                    let s = *rx.borrow_and_update();
                    // Drop transitions that belong to a previous PTT.
                    if counter.load(Ordering::SeqCst) != session_id_at_spawn {
                        return;
                    }
                    // Only forward the non-Idle transitions here; the
                    // outer `stop_and_transcribe` owns the terminal
                    // Idle publish so we don't flicker the indicator
                    // between the transcriber's Idle and our own
                    // terminal publish.
                    if matches!(
                        s,
                        VoiceCaptureState::Queued | VoiceCaptureState::Transcribing
                    ) {
                        let _ = state_tx.send(s);
                    }
                }
            });
            inner.forwarder = Some(handle);
        }

        drop(inner);

        let _ = self.state_tx.send(VoiceCaptureState::Recording);
        info!(target: "assistd::voice::mic", session_id, "recording started");
        Ok(())
    }

    async fn stop_and_transcribe(&self) -> Result<String> {
        let (session, forwarder) = {
            let mut inner = self.inner.lock().await;
            match inner.session.take() {
                Some(s) => (s, inner.forwarder.take()),
                None => {
                    // Release without matching press — benign no-op.
                    return Ok(String::new());
                }
            }
        };

        // Publish an initial Transcribing; the forwarder may overwrite
        // it with Queued if the transcriber decides to wait for the
        // GPU. That's fine — the TUI renders the latest value.
        let _ = self.state_tx.send(VoiceCaptureState::Transcribing);

        // Signal stop; the worker will exit its drain loop, drop the
        // cpal stream (on its own thread), and return the final PCM.
        session.stop_flag.store(true, Ordering::SeqCst);

        let pcm = match session.handle.await {
            Ok(Ok(pcm)) => pcm,
            Ok(Err(e)) => {
                self.cleanup_forwarder_and_idle(forwarder).await;
                return Err(anyhow!(VoiceInputError::from(e)));
            }
            Err(join_err) => {
                self.cleanup_forwarder_and_idle(forwarder).await;
                return Err(anyhow!(VoiceInputError::ConsumerPanic(
                    join_err.to_string()
                )));
            }
        };

        let overrun = session.overrun.load(Ordering::Relaxed);
        if overrun > 0 {
            warn!(
                target: "assistd::voice::mic",
                overrun_samples = overrun,
                "ring buffer overrun during PTT capture — audio was truncated"
            );
        }

        if pcm.is_empty() {
            self.cleanup_forwarder_and_idle(forwarder).await;
            return Ok(String::new());
        }

        let duration_secs = pcm.len() as f32 / 16_000.0;
        let peak = pcm.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
        let peak_dbfs = if peak == 0 {
            f32::NEG_INFINITY
        } else {
            20.0 * (peak as f32 / i16::MAX as f32).log10()
        };
        info!(
            target: "assistd::voice::mic",
            pcm_samples = pcm.len(),
            duration_secs,
            peak_dbfs,
            "captured pcm; invoking transcriber"
        );

        let result = self.transcriber.transcribe(&pcm).await;

        // Regardless of result: retire the forwarder and publish Idle.
        self.cleanup_forwarder_and_idle(forwarder).await;

        let text = match result {
            Ok(t) => t,
            Err(e) => return Err(anyhow!(VoiceInputError::from(e))),
        };

        info!(
            target: "assistd::voice::mic",
            pcm_samples = pcm.len(),
            text_chars = text.chars().count(),
            text = %text,
            "transcription complete"
        );
        Ok(text)
    }

    fn state(&self) -> VoiceCaptureState {
        *self.state_tx.borrow()
    }

    fn subscribe(&self) -> watch::Receiver<VoiceCaptureState> {
        self.state_tx.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_capture_state_pins_idle_default() {
        // Guards against accidental reordering of the enum — the
        // TUI treats Idle as the no-indicator default.
        assert_eq!(VoiceCaptureState::Idle as u8, 0);
    }

    #[test]
    fn error_conversions_compile() {
        fn _from_capture(e: AudioCaptureError) -> VoiceInputError {
            e.into()
        }
        fn _from_transcribe(e: TranscriptionError) -> VoiceInputError {
            e.into()
        }
    }
}
