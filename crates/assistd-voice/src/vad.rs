//! Silero voice activity detection through sherpa-onnx, over 16 kHz mono
//! audio in 32 ms windows.

use std::fmt;
use std::ops::Range;
use std::path::{Path, PathBuf};

use assistd_config::TranscriptionConfig;
use assistd_utils::hf::download;
use num_traits::ToPrimitive;
use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};

use crate::transcribe::TranscriptionError;

/// Samples per Silero window at 16 kHz: 32 ms.
pub const WINDOW_SAMPLES: usize = 512;

const SAMPLE_RATE_HZ: i32 = 16_000;

/// Speech probability at or above which a window counts as speech; Silero's
/// own default.
const SPEECH_THRESHOLD: f32 = 0.5;

/// Longest speech segment before the detector splits it, in seconds.
const MAX_SEGMENT_SECS: f32 = 30.0;

/// Seconds of segmented speech the detector can hold before it is popped.
const BUFFER_SECS: f32 = 60.0;

/// A streaming Silero detector; its state carries across calls until
/// [`reset`](Self::reset).
pub struct SpeechDetector {
    vad: VoiceActivityDetector,
}

impl fmt::Debug for SpeechDetector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpeechDetector").finish_non_exhaustive()
    }
}

impl SpeechDetector {
    /// A detector whose segments need `min_speech_secs` of speech to start
    /// and end after `min_silence_secs` of silence; with both `0.0` every
    /// window is judged on its own.
    pub fn load(
        model: &Path,
        min_speech_secs: f32,
        min_silence_secs: f32,
    ) -> Result<Self, TranscriptionError> {
        let config = VadModelConfig {
            silero_vad: SileroVadModelConfig {
                model: Some(model.to_string_lossy().into_owned()),
                threshold: SPEECH_THRESHOLD,
                min_silence_duration: min_silence_secs,
                min_speech_duration: min_speech_secs,
                window_size: i32::try_from(WINDOW_SAMPLES).unwrap_or(i32::MAX),
                max_speech_duration: MAX_SEGMENT_SECS,
            },
            sample_rate: SAMPLE_RATE_HZ,
            num_threads: 1,
            provider: Some("cpu".into()),
            ..VadModelConfig::default()
        };
        VoiceActivityDetector::create(&config, BUFFER_SECS)
            .map(|vad| Self { vad })
            .ok_or_else(|| {
                TranscriptionError::VadInit(format!(
                    "sherpa-onnx could not load {}",
                    model.display()
                ))
            })
    }

    /// Forget all audio seen so far.
    pub fn reset(&mut self) {
        self.vad.reset();
    }

    /// Feed one window and report whether the detector is inside speech.
    pub fn push_window(&mut self, window: &[f32; WINDOW_SAMPLES]) -> bool {
        self.vad.accept_waveform(window);
        let speech = self.vad.detected();
        self.vad.clear();
        speech
    }

    /// The speech in `audio` as sample ranges, from a fresh state.
    pub fn speech_ranges(&mut self, audio: &[f32]) -> Vec<Range<usize>> {
        self.vad.reset();
        for window in audio.chunks(WINDOW_SAMPLES) {
            self.vad.accept_waveform(window);
        }
        self.vad.flush();
        let mut ranges = Vec::new();
        while let Some(segment) = self.vad.front() {
            let start = segment.start().to_usize().unwrap_or(0);
            let len = segment.n().to_usize().unwrap_or(0);
            ranges.push(start..start + len);
            self.vad.pop();
        }
        self.vad.reset();
        ranges
    }
}

/// Download `config.vad_model` if missing and return its path.
pub async fn ensure_model(config: &TranscriptionConfig) -> Result<PathBuf, TranscriptionError> {
    let cache_dir = config
        .model_cache_dir
        .clone()
        .map_or_else(|| download::default_cache_dir("stt"), Ok)?;
    Ok(download::ensure_cached(&config.vad_model, &cache_dir).await?)
}
