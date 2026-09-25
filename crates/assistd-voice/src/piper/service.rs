//! [`PiperVoiceOutput`]: Piper synthesis plus rodio playback behind a
//! circuit breaker.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use assistd_config::SynthesisConfig;
use async_trait::async_trait;
use parking_lot::Mutex;

use crate::piper::cache::{VoiceFiles, default_cache_dir, ensure_voice};
use crate::piper::config::{NOISE_SCALE, NOISE_W, PiperRuntimeConfig, SENTENCE_SILENCE_SECS};
use crate::piper::error::PiperError;
use crate::piper::playback::RodioPlaybackWorker;
use crate::piper::synth::OneShotSynth;
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
                target: "assistd::voice::piper",
                %reason,
                "piper degraded; dropping speak() request"
            );
            self.logged_degraded = true;
        }
        false
    }

    fn record_success(&mut self) {
        self.recent_failures.clear();
        if matches!(self.ready, ReadyState::Degraded { .. }) {
            tracing::info!(target: "assistd::voice::piper", "piper recovered from degraded");
            self.ready = ReadyState::Ready;
            self.logged_degraded = false;
        }
    }

    fn record_failure(&mut self, err: &PiperError) {
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
                target: "assistd::voice::piper",
                %reason,
                threshold = FAILURE_THRESHOLD,
                window_secs = FAILURE_WINDOW.as_secs(),
                "piper synthesis repeatedly failed; entering degraded state"
            );
            self.ready = ReadyState::Degraded { reason };
        }
    }
}

/// [`VoiceOutput`] backed by per-utterance piper subprocesses and
/// rodio playback. After three failures within a minute the service is
/// degraded and `speak` drops utterances without error; once a minute
/// has passed since the last failure, one utterance is let through and
/// either re-arms the service or re-opens the breaker.
pub struct PiperVoiceOutput {
    synth: Arc<OneShotSynth>,
    playback: Arc<RodioPlaybackWorker>,
    circuit: Arc<Mutex<CircuitState>>,
}

impl PiperVoiceOutput {
    /// Resolve the voice files, open the audio device, and run a
    /// health-check synthesis.
    pub async fn start(config: SynthesisConfig) -> Result<Self, PiperError> {
        which::which(&config.binary_path).map_err(|_| PiperError::BinaryMissing {
            binary: config.binary_path.clone(),
        })?;

        let cache_dir = config
            .model_cache_dir
            .clone()
            .unwrap_or_else(default_cache_dir);
        let voice_files = ensure_voice(&config.voice, &cache_dir).await?;
        tracing::info!(
            target: "assistd::voice::piper",
            onnx = %voice_files.onnx.display(),
            sample_rate = voice_files.sample_rate,
            "piper voice resolved"
        );

        let runtime = Arc::new(runtime_config(&config, voice_files));
        let synth = Arc::new(OneShotSynth::new(runtime));
        let playback = Arc::new(RodioPlaybackWorker::start(config.output_device.as_deref())?);

        synth.health_check().await?;
        tracing::info!(
            target: "assistd::voice::piper",
            "piper health-check passed"
        );

        Ok(Self {
            synth,
            playback,
            circuit: Arc::new(Mutex::new(CircuitState::new())),
        })
    }
}

#[async_trait]
impl VoiceOutput for PiperVoiceOutput {
    async fn speak(&self, text: String) -> Result<(), VoiceOutputError> {
        if !self.circuit.lock().admit() {
            return Ok(());
        }

        if text.trim().is_empty() {
            return Ok(());
        }

        let output = match self.synth.synthesize(&text).await {
            Ok(output) => output,
            Err(err) => {
                tracing::warn!(
                    target: "assistd::voice::piper",
                    error = %err,
                    "piper synthesis failed"
                );
                self.circuit.lock().record_failure(&err);
                return Err(VoiceOutputError::Synthesis(err));
            }
        };

        if let Err(err) = self.playback.play(output).await {
            tracing::warn!(
                target: "assistd::voice::piper",
                error = %err,
                "piper playback enqueue failed"
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

fn runtime_config(config: &SynthesisConfig, voice_files: VoiceFiles) -> PiperRuntimeConfig {
    PiperRuntimeConfig {
        binary_path: config.binary_path.clone(),
        voice_files,
        length_scale: config.length_scale,
        noise_scale: NOISE_SCALE,
        noise_w: NOISE_W,
        sentence_silence_secs: SENTENCE_SILENCE_SECS,
        espeak_data_dir: config.espeak_data_dir.clone(),
        deadline: Duration::from_secs(u64::from(config.deadline_secs.get())),
        use_cuda: config.use_cuda,
        output_device: config.output_device.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err() -> PiperError {
        PiperError::Deadline { secs: 1 }
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
    fn degraded_drop_is_logged_exactly_once() {
        let mut state = CircuitState::new();
        trip(&mut state);
        assert!(!state.logged_degraded, "tripping must not consume the log");

        assert!(!state.admit());
        assert!(state.logged_degraded, "first dropped speak() logs");
        assert!(!state.admit());
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
