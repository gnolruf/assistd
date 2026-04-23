//! `MicContinuousListener` — cpal + webrtc-vad + whisper-rs driven
//! hands-free utterance transcriber.
//!
//! Architecturally mirrors [`crate::MicVoiceInput`]:
//!   cpal callback → ring buffer → streaming resampler → VAD frames
//!                                                      → whisper → broadcast
//!
//! The cpal stream lives in a `spawn_blocking` worker because it's
//! `!Send` on ALSA. A second async task owns the VAD state machine
//! and drains completed utterances through whisper, publishing
//! transcripts on an internal broadcast channel.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, anyhow};
use assistd_config::{ContinuousListenConfig, VoiceConfig};
use async_trait::async_trait;
use tokio::sync::{Mutex, broadcast, mpsc, watch};
use tokio::task::JoinHandle;
use tracing::{info, warn};

use crate::transcribe::Transcriber;

use super::ContinuousListener;
use super::capture::{self, ListenCaptureSession};
use super::vad::{FRAME_SAMPLES, UtteranceVad, VadEvent, VadTuning};

type CaptureJoin = JoinHandle<Result<(), crate::mic::capture::AudioCaptureError>>;

/// Depth of the utterance broadcast channel. Slow consumers fall
/// behind by this many utterances; faster consumers keep up.
const UTTERANCE_CHANNEL_DEPTH: usize = 16;

/// Depth of the VAD frame channel. Large enough to buffer a few
/// scheduling hiccups without dropping frames — at 20 ms per frame,
/// 128 slots is ~2.5 s of audio.
const FRAME_CHANNEL_DEPTH: usize = 128;

/// Cpal + VAD + whisper implementation of [`ContinuousListener`].
pub struct MicContinuousListener {
    transcriber: Arc<dyn Transcriber>,
    mic_device: Option<String>,
    tuning: VadTuning,
    active: Arc<AtomicBool>,
    state_tx: watch::Sender<bool>,
    utterances: broadcast::Sender<String>,
    inner: Arc<Mutex<InnerState>>,
}

struct InnerState {
    session: Option<ActiveSession>,
}

struct ActiveSession {
    capture_stop: Arc<AtomicBool>,
    capture_handle: CaptureJoin,
    vad_handle: JoinHandle<()>,
}

impl MicContinuousListener {
    /// Build from config, sharing a transcriber with the PTT pipeline
    /// (typically the same [`crate::QueuedTranscriber`] so continuous
    /// listening benefits from the same queue-and-fallback policy).
    /// Does not open the audio device — that happens on [`Self::start`].
    pub fn new(transcriber: Arc<dyn Transcriber>, cfg: &VoiceConfig) -> Self {
        let tuning = tuning_from_config(&cfg.continuous);
        let (state_tx, _) = watch::channel(false);
        let (utterances, _) = broadcast::channel(UTTERANCE_CHANNEL_DEPTH);
        Self {
            transcriber,
            mic_device: cfg.mic_device.clone(),
            tuning,
            active: Arc::new(AtomicBool::new(false)),
            state_tx,
            utterances,
            inner: Arc::new(Mutex::new(InnerState { session: None })),
        }
    }

    /// Accessor used by the daemon's PTT-exclusion guard.
    pub fn active_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.active)
    }
}

fn tuning_from_config(cfg: &ContinuousListenConfig) -> VadTuning {
    VadTuning::from_ms(
        cfg.silence_ms,
        cfg.min_utterance_ms,
        cfg.max_utterance_secs,
        cfg.preroll_ms,
        cfg.onset_confirm_ms,
        cfg.aggressiveness,
    )
}

#[async_trait]
impl ContinuousListener for MicContinuousListener {
    async fn start(&self) -> Result<()> {
        let mut inner = self.inner.lock().await;
        if inner.session.is_some() {
            return Ok(()); // idempotent
        }
        if self
            .active
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(anyhow!(
                "continuous listening is already active on another subsystem"
            ));
        }

        let (frame_tx, frame_rx) = mpsc::channel::<Box<[i16; FRAME_SAMPLES]>>(FRAME_CHANNEL_DEPTH);

        let ListenCaptureSession {
            stop_flag: capture_stop,
            overrun: _,
            handle: capture_handle,
        } = capture::start(self.mic_device.as_deref(), frame_tx);

        let rt_handle = tokio::runtime::Handle::current();
        let tuning = self.tuning;
        let transcriber = self.transcriber.clone();
        let utterances = self.utterances.clone();
        let vad_handle = tokio::task::spawn_blocking(move || {
            run_vad_blocking(tuning, transcriber, utterances, frame_rx, rt_handle);
        });

        inner.session = Some(ActiveSession {
            capture_stop,
            capture_handle,
            vad_handle,
        });
        let _ = self.state_tx.send(true);
        info!(target: "assistd::voice::listen", "continuous listening started");
        Ok(())
    }

    async fn stop(&self) -> Result<()> {
        let session = {
            let mut inner = self.inner.lock().await;
            inner.session.take()
        };
        let Some(session) = session else {
            // Not active — reset the flag defensively and exit.
            self.active.store(false, Ordering::SeqCst);
            return Ok(());
        };

        session.capture_stop.store(true, Ordering::SeqCst);
        match session.capture_handle.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => warn!(target: "assistd::voice::listen", "capture worker error: {e}"),
            Err(e) => warn!(target: "assistd::voice::listen", "capture worker panicked: {e}"),
        }
        // Dropping the capture side closes `frame_tx`; the VAD loop
        // sees `None` on `recv()` and exits.
        let _ = session.vad_handle.await;

        self.active.store(false, Ordering::SeqCst);
        let _ = self.state_tx.send(false);
        info!(target: "assistd::voice::listen", "continuous listening stopped");
        Ok(())
    }

    fn is_active(&self) -> bool {
        self.active.load(Ordering::SeqCst)
    }

    fn subscribe_utterances(&self) -> broadcast::Receiver<String> {
        self.utterances.subscribe()
    }

    fn subscribe_state(&self) -> watch::Receiver<bool> {
        self.state_tx.subscribe()
    }
}

/// Runs inside `spawn_blocking` because `UtteranceVad` (via
/// `webrtc_vad::Vad`) holds a `!Send` C pointer. Transcription jobs
/// are spawned onto the async runtime via the captured handle.
fn run_vad_blocking(
    tuning: VadTuning,
    transcriber: Arc<dyn Transcriber>,
    utterances: broadcast::Sender<String>,
    mut frame_rx: mpsc::Receiver<Box<[i16; FRAME_SAMPLES]>>,
    rt: tokio::runtime::Handle,
) {
    let mut vad = UtteranceVad::new(tuning);
    while let Some(frame) = frame_rx.blocking_recv() {
        let Some(event) = vad.feed(&frame) else {
            continue;
        };
        let pcm = match event {
            VadEvent::UtteranceComplete(p) | VadEvent::Truncated(p) => p,
        };
        let transcriber = transcriber.clone();
        let utterances = utterances.clone();
        rt.spawn(async move {
            match transcriber.transcribe(&pcm).await {
                Ok(text) => {
                    let trimmed = text.trim();
                    if trimmed.is_empty() {
                        // Whisper's Silero VAD trimmed the segment to
                        // silence — drop without dispatching.
                        return;
                    }
                    if utterances.send(trimmed.to_string()).is_err() {
                        tracing::debug!(
                            target: "assistd::voice::listen",
                            "no utterance subscribers; dropping transcript"
                        );
                    }
                }
                Err(e) => {
                    warn!(target: "assistd::voice::listen", "transcription failed: {e:#}");
                }
            }
        });
    }
}
