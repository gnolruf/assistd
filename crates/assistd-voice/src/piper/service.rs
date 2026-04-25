//! `PiperVoiceOutput` — the assembled façade implementing
//! [`crate::VoiceOutput`]. Owns one [`OneShotSynth`] and one
//! [`RodioPlaybackWorker`] for the daemon's lifetime; speak() runs the
//! per-utterance subprocess and queues PCM on the playback worker.
//!
//! Circuit breaker: synthesis failures are timestamped in a small
//! ringbuffer. After 3 failures within 60 seconds the service flips
//! to [`ReadyState::Degraded`] and subsequent speak() calls become
//! no-ops (logged once). This is the practical interpretation of
//! "restarted on crash" for the per-utterance design — a missing
//! binary or broken audio device shouldn't spam the logs forever.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use assistd_config::SynthesisConfig;
use async_trait::async_trait;

use crate::VoiceOutput;
use crate::piper::cache::{default_cache_dir, ensure_voice};
use crate::piper::config::PiperRuntimeConfig;
use crate::piper::error::PiperError;
use crate::piper::playback::RodioPlaybackWorker;
use crate::piper::synth::OneShotSynth;

/// Number of failures within `FAILURE_WINDOW` that flips the breaker.
const FAILURE_THRESHOLD: usize = 3;
/// Sliding window for circuit-breaker accounting.
const FAILURE_WINDOW: Duration = Duration::from_secs(60);

#[derive(Debug, Clone)]
pub enum ReadyState {
    Ready,
    Degraded { reason: String },
}

struct CircuitState {
    ready: ReadyState,
    recent_failures: VecDeque<Instant>,
    /// True after we've logged the "degraded" line once, so we don't
    /// spam the journal on every subsequent speak().
    logged_degraded: bool,
}

pub struct PiperVoiceOutput {
    synth: Arc<OneShotSynth>,
    playback: Arc<RodioPlaybackWorker>,
    state: Arc<Mutex<CircuitState>>,
}

impl PiperVoiceOutput {
    /// Resolve the voice cache, build the runtime config, open the
    /// audio device, and run a tiny health-check synthesis. Any
    /// failure here is reported as `PiperError`; the daemon's startup
    /// logic then logs a warning and substitutes `NoVoiceOutput`.
    pub async fn start(cfg: SynthesisConfig) -> Result<Self, PiperError> {
        // Fail-fast on a missing binary so the error is "binary not
        // found" rather than the more confusing "spawn: file not
        // found" surfaced after voice download.
        which::which(&cfg.binary_path).map_err(|_| PiperError::BinaryMissing {
            binary: cfg.binary_path.clone(),
        })?;

        let cache_dir = cfg
            .model_cache_dir
            .clone()
            .unwrap_or_else(default_cache_dir);
        let voice_files = ensure_voice(&cfg.voice, &cache_dir).await?;
        tracing::info!(
            target: "assistd::voice::piper",
            onnx = %voice_files.onnx.display(),
            sample_rate = voice_files.sample_rate,
            "piper voice resolved"
        );

        let runtime = Arc::new(PiperRuntimeConfig {
            binary_path: cfg.binary_path.clone(),
            voice_files,
            length_scale: cfg.length_scale,
            noise_scale: cfg.noise_scale,
            noise_w: cfg.noise_w,
            sentence_silence_secs: cfg.sentence_silence_secs,
            espeak_data_dir: cfg.espeak_data_dir.clone(),
            deadline: Duration::from_secs(cfg.deadline_secs as u64),
        });

        let synth = Arc::new(OneShotSynth::new(runtime.clone()));
        let playback = Arc::new(RodioPlaybackWorker::start()?);

        // Health-check: a tiny synth surfaces missing-model and
        // device-format errors at startup rather than on the first
        // user-facing utterance.
        synth.health_check().await?;
        tracing::info!(
            target: "assistd::voice::piper",
            "piper health-check passed"
        );

        Ok(Self {
            synth,
            playback,
            state: Arc::new(Mutex::new(CircuitState {
                ready: ReadyState::Ready,
                recent_failures: VecDeque::with_capacity(FAILURE_THRESHOLD),
                logged_degraded: false,
            })),
        })
    }

    pub fn ready_state(&self) -> ReadyState {
        self.state
            .lock()
            .expect("piper state mutex poisoned")
            .ready
            .clone()
    }

    fn record_success(&self) {
        let mut s = self.state.lock().expect("piper state mutex poisoned");
        s.recent_failures.clear();
        // Re-arming after a transient flap is intentional: a transient
        // stutter shouldn't permanently disable speech.
        if matches!(s.ready, ReadyState::Degraded { .. }) {
            tracing::info!(target: "assistd::voice::piper", "piper recovered from degraded");
            s.ready = ReadyState::Ready;
            s.logged_degraded = false;
        }
    }

    fn record_failure(&self, err: &PiperError) {
        let mut s = self.state.lock().expect("piper state mutex poisoned");
        let now = Instant::now();
        // Drop entries older than the window.
        while let Some(&front) = s.recent_failures.front() {
            if now.duration_since(front) > FAILURE_WINDOW {
                s.recent_failures.pop_front();
            } else {
                break;
            }
        }
        s.recent_failures.push_back(now);
        if s.recent_failures.len() >= FAILURE_THRESHOLD && matches!(s.ready, ReadyState::Ready) {
            let reason = err.to_string();
            tracing::warn!(
                target: "assistd::voice::piper",
                %reason,
                threshold = FAILURE_THRESHOLD,
                window_secs = FAILURE_WINDOW.as_secs(),
                "piper synthesis repeatedly failed; entering degraded state"
            );
            s.ready = ReadyState::Degraded { reason };
            s.logged_degraded = true;
        }
    }
}

#[async_trait]
impl VoiceOutput for PiperVoiceOutput {
    async fn speak(&self, text: String) -> Result<()> {
        // Short-circuit when degraded — log once, then silent until a
        // future health check (manual restart for now) clears it.
        {
            let s = self.state.lock().expect("piper state mutex poisoned");
            if let ReadyState::Degraded { ref reason } = s.ready {
                if !s.logged_degraded {
                    tracing::warn!(
                        target: "assistd::voice::piper",
                        %reason,
                        "piper degraded; dropping speak() request"
                    );
                }
                return Ok(());
            }
        }

        if text.trim().is_empty() {
            return Ok(());
        }

        let output = match self.synth.synthesize(&text).await {
            Ok(o) => o,
            Err(e) => {
                tracing::warn!(
                    target: "assistd::voice::piper",
                    error = %e,
                    "piper synthesis failed"
                );
                self.record_failure(&e);
                return Ok(());
            }
        };

        if let Err(e) = self.playback.play(output) {
            tracing::warn!(
                target: "assistd::voice::piper",
                error = %e,
                "piper playback enqueue failed"
            );
            self.record_failure(&e);
            return Ok(());
        }

        // Honor the trait contract: speak() returns when playback
        // finishes. Without this, multiple concurrent speak() calls
        // would interleave audio.
        if let Err(e) = self.playback.drain().await {
            tracing::warn!(
                target: "assistd::voice::piper",
                error = %e,
                "piper playback drain failed"
            );
            self.record_failure(&e);
            return Ok(());
        }

        self.record_success();
        Ok(())
    }
}
