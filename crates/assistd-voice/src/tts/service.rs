//! [`SherpaVoiceOutput`]: sherpa-onnx synthesis plus rodio playback behind
//! a circuit breaker.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use assistd_config::SynthesisConfig;
use async_trait::async_trait;
use parking_lot::Mutex;

use crate::tts::engine::{SpeechEngine, Utterance};
use crate::tts::error::SpeechError;
use crate::tts::playback::RodioPlaybackWorker;
use crate::tts::voice_files::ensure_voice;
use crate::{VoiceOutput, VoiceOutputError};

const FAILURE_THRESHOLD: usize = 3;
const FAILURE_WINDOW: Duration = Duration::from_secs(60);

#[derive(Debug)]
enum ReadyState {
    /// Synthesis is operating normally.
    Ready,
    /// The breaker has tripped; `reason` is the last error.
    Degraded { reason: String },
}

#[derive(Debug)]
struct CircuitState {
    ready: ReadyState,
    recent_failures: VecDeque<Instant>,
    logged_degraded: bool,
}

impl CircuitState {
    fn new() -> Self {
        Self {
            ready: ReadyState::Ready,
            recent_failures: VecDeque::with_capacity(FAILURE_THRESHOLD),
            logged_degraded: false,
        }
    }

    fn admit(&mut self) -> bool {
        let ReadyState::Degraded { reason } = &self.ready else {
            return true;
        };
        let cooled = self
            .recent_failures
            .back()
            .is_none_or(|last| last.elapsed() > FAILURE_WINDOW);
        if cooled {
            self.recent_failures.clear();
            return true;
        }
        if !self.logged_degraded {
            tracing::warn!(
                target: "assistd::voice::tts",
                %reason,
                "speech degraded; dropping speak() request"
            );
            self.logged_degraded = true;
        }
        false
    }

    fn record_success(&mut self) {
        self.recent_failures.clear();
        if matches!(self.ready, ReadyState::Degraded { .. }) {
            tracing::info!(target: "assistd::voice::tts", "speech recovered from degraded");
            self.ready = ReadyState::Ready;
            self.logged_degraded = false;
        }
    }

    fn record_failure(&mut self, err: &SpeechError) {
        let now = Instant::now();
        while self
            .recent_failures
            .front()
            .is_some_and(|&oldest| now.duration_since(oldest) > FAILURE_WINDOW)
        {
            self.recent_failures.pop_front();
        }
        self.recent_failures.push_back(now);
        if self.recent_failures.len() >= FAILURE_THRESHOLD
            && matches!(self.ready, ReadyState::Ready)
        {
            let reason = err.to_string();
            tracing::warn!(
                target: "assistd::voice::tts",
                %reason,
                threshold = FAILURE_THRESHOLD,
                window_secs = FAILURE_WINDOW.as_secs(),
                "speech synthesis repeatedly failed; entering degraded state"
            );
            self.ready = ReadyState::Degraded { reason };
        }
    }
}

/// [`VoiceOutput`] backed by a sherpa-onnx voice and rodio playback. Three failures within a
/// minute degrade it and `speak` silently drops utterances; a minute after the last failure,
/// one utterance is let through to re-arm the service or re-open the breaker.
#[derive(Debug)]
pub struct SherpaVoiceOutput {
    engine: Arc<Mutex<SpeechEngine>>,
    playback: Arc<RodioPlaybackWorker>,
    circuit: Arc<Mutex<CircuitState>>,
}

impl SherpaVoiceOutput {
    /// Fetch and load the voice, open the audio device, and speak a
    /// health-check sentence.
    pub async fn start(config: &SynthesisConfig) -> Result<Self, SpeechError> {
        let voice = ensure_voice(config).await?;
        tracing::info!(target: "assistd::voice::tts", ?voice, "voice resolved");
        let (speaker_id, length_scale) = (config.speaker_id, config.length_scale);
        let engine = tokio::task::spawn_blocking(move || {
            SpeechEngine::load(&voice, speaker_id, length_scale)
        })
        .await??;
        let engine = Arc::new(Mutex::new(engine));
        let playback = RodioPlaybackWorker::start(config.output_device.clone()).await?;

        let probe = synthesize(&engine, "ok".into()).await?;
        if probe.samples.is_empty() {
            return Err(SpeechError::Rejected {
                reason: "health check produced no audio".into(),
            });
        }
        tracing::info!(target: "assistd::voice::tts", "speech health check passed");

        Ok(Self {
            engine,
            playback: Arc::new(playback),
            circuit: Arc::new(Mutex::new(CircuitState::new())),
        })
    }
}

#[async_trait]
impl VoiceOutput for SherpaVoiceOutput {
    async fn speak(&self, text: String) -> Result<(), VoiceOutputError> {
        if !self.circuit.lock().admit() {
            return Ok(());
        }

        if text.trim().is_empty() {
            return Ok(());
        }

        let output = match synthesize(&self.engine, text).await {
            Ok(output) => output,
            Err(err) => {
                tracing::warn!(
                    target: "assistd::voice::tts",
                    error = %err,
                    "speech synthesis failed"
                );
                self.circuit.lock().record_failure(&err);
                return Err(VoiceOutputError::Synthesis(err));
            }
        };

        if let Err(err) = self.playback.play(output).await {
            tracing::warn!(
                target: "assistd::voice::tts",
                error = %err,
                "speech playback enqueue failed"
            );
            self.circuit.lock().record_failure(&err);
            return Err(VoiceOutputError::Playback(err));
        }

        self.circuit.lock().record_success();
        Ok(())
    }

    async fn wait_idle(&self) -> Result<(), VoiceOutputError> {
        let degraded = matches!(self.circuit.lock().ready, ReadyState::Degraded { .. });
        if degraded {
            return Ok(());
        }
        self.playback.drain().await;
        Ok(())
    }

    async fn cancel(&self) {
        self.playback.clear().await;
    }
}

/// Synthesize on the blocking pool, one sentence at a time.
async fn synthesize(
    engine: &Arc<Mutex<SpeechEngine>>,
    sentence: String,
) -> Result<Utterance, SpeechError> {
    let engine = Arc::clone(engine);
    tracing::debug!(
        target: "assistd::voice::latency",
        stage = "tts_synth_start",
        "voice latency stage"
    );
    let utterance =
        tokio::task::spawn_blocking(move || engine.lock().synthesize(&sentence)).await??;
    tracing::debug!(
        target: "assistd::voice::latency",
        stage = "tts_synth_done",
        "voice latency stage"
    );
    Ok(utterance)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err() -> SpeechError {
        SpeechError::Rejected {
            reason: "test".into(),
        }
    }

    fn trip(state: &mut CircuitState) {
        for _ in 0..FAILURE_THRESHOLD {
            state.record_failure(&err());
        }
    }

    fn age_failures(state: &mut CircuitState, by: Duration) {
        for at in &mut state.recent_failures {
            *at = at.checked_sub(by).expect("test clock underflow");
        }
    }

    #[test]
    fn threshold_failures_trip_the_breaker() {
        let mut state = CircuitState::new();
        state.record_failure(&err());
        state.record_failure(&err());
        assert!(state.admit(), "below threshold stays admitting");

        state.record_failure(&err());
        assert!(matches!(state.ready, ReadyState::Degraded { .. }));
        assert!(!state.admit(), "tripped breaker drops utterances");
    }

    #[test]
    fn failures_outside_the_window_do_not_accumulate() {
        let mut state = CircuitState::new();
        state.record_failure(&err());
        state.record_failure(&err());
        age_failures(&mut state, FAILURE_WINDOW + Duration::from_secs(1));

        state.record_failure(&err());
        assert!(matches!(state.ready, ReadyState::Ready));
        assert_eq!(state.recent_failures.len(), 1);
    }

    #[test]
    fn half_open_probe_admits_after_the_window_cools() {
        let mut state = CircuitState::new();
        trip(&mut state);
        assert!(!state.admit());

        age_failures(&mut state, FAILURE_WINDOW + Duration::from_secs(1));
        assert!(state.admit(), "cooled breaker admits a probe");
        assert!(
            matches!(state.ready, ReadyState::Degraded { .. }),
            "probe alone does not re-arm; only its success does"
        );
    }

    #[test]
    fn successful_probe_rearms_the_service() {
        let mut state = CircuitState::new();
        trip(&mut state);
        age_failures(&mut state, FAILURE_WINDOW + Duration::from_secs(1));
        assert!(state.admit());

        state.record_success();
        assert!(matches!(state.ready, ReadyState::Ready));
        assert!(!state.logged_degraded);
        assert!(state.admit());
    }

    #[test]
    fn failed_probe_reopens_the_breaker() {
        let mut state = CircuitState::new();
        trip(&mut state);
        assert!(!state.admit());
        age_failures(&mut state, FAILURE_WINDOW + Duration::from_secs(1));
        assert!(state.admit());

        state.record_failure(&err());
        assert!(matches!(state.ready, ReadyState::Degraded { .. }));
        assert!(!state.admit(), "a failed probe keeps the breaker open");
    }
}
