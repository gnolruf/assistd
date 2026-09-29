//! Whisper-rs-backed [`Transcriber`].

use std::fmt;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use assistd_config::TranscriptionConfig;
use async_trait::async_trait;
use parking_lot::Mutex;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
    WhisperVadContext, WhisperVadContextParams, WhisperVadParams, WhisperVadSegment,
    convert_integer_to_float_audio,
};

use crate::gpu;
use crate::hf_download;
use crate::transcribe::{Transcriber, TranscriptionError};

/// Minimum trailing silence, in seconds, before Silero VAD trims a
/// segment. Maps to whisper.cpp's `min_silence_duration_ms`.
pub const VAD_SILENCE_SECS: f32 = 0.5;

const SAMPLE_RATE_HZ: usize = 16_000;
const SAMPLES_PER_CENTISECOND: f32 = SAMPLE_RATE_HZ as f32 / 100.0;
/// Trailing audio kept after each non-final speech segment, matching
/// whisper.cpp's `samples_overlap` default.
const SEGMENT_OVERLAP_SECS: f32 = 0.1;
/// Silence inserted between stitched speech segments.
const SEGMENT_GAP_SECS: f32 = 0.1;

#[derive(Debug, Clone)]
struct InferenceConfig {
    threads: Option<u32>,
    beams: u32,
}

#[derive(Debug, Clone)]
struct SileroVadParams {
    model_path: String,
    silence_secs: f32,
}

/// Silero VAD run ahead of decoding, keeping only the speech segments.
/// whisper.cpp only applies its built-in VAD in `whisper_full`, which
/// whisper-rs never calls, so trimming happens here instead.
#[derive(Debug)]
struct SpeechTrimmer {
    context: Mutex<WhisperVadContext>,
    silence_secs: f32,
}

impl SpeechTrimmer {
    fn load(params: &SileroVadParams, use_gpu: bool) -> Result<Self, TranscriptionError> {
        let mut context_params = WhisperVadContextParams::new();
        context_params.set_use_gpu(use_gpu);
        let context = WhisperVadContext::new(&params.model_path, context_params)
            .map_err(|err| TranscriptionError::WhisperInit(err.to_string()))?;
        Ok(Self {
            context: Mutex::new(context),
            silence_secs: params.silence_secs,
        })
    }

    /// Returns only the speech in `audio`, or an empty buffer when none was detected.
    fn trim(&self, audio: &[f32]) -> Result<Vec<f32>, TranscriptionError> {
        let segments = self
            .context
            .lock()
            .segments_from_samples(silero_vad_params(self.silence_secs), audio)
            .map_err(|err| TranscriptionError::WhisperInference(err.to_string()))?
            .collect::<Vec<_>>();
        tracing::debug!(
            target: "assistd::voice::whisper",
            segments = segments.len(),
            "silero vad detected speech segments"
        );
        Ok(stitch_speech_segments(audio, &segments))
    }
}

/// Idle whisper states (KV cache and compute buffers) reused across calls.
/// Each call checks out its own state; only a state whose run succeeded is returned.
struct StatePool<S> {
    idle: Mutex<Vec<S>>,
}

impl<S> fmt::Debug for StatePool<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StatePool").finish_non_exhaustive()
    }
}

impl<S> Default for StatePool<S> {
    fn default() -> Self {
        Self {
            idle: Mutex::new(Vec::new()),
        }
    }
}

impl<S> StatePool<S> {
    fn with_state<R, E>(
        &self,
        create: impl FnOnce() -> Result<S, E>,
        run: impl FnOnce(&mut S) -> Result<R, E>,
    ) -> Result<R, E> {
        let idle = self.idle.lock().pop();
        let mut state = match idle {
            Some(state) => state,
            None => create()?,
        };
        let out = run(&mut state)?;
        self.idle.lock().push(state);
        Ok(out)
    }
}

/// Concrete [`Transcriber`] backed by whisper.cpp via whisper-rs.
#[derive(Debug)]
pub struct WhisperTranscriber {
    ctx: Arc<WhisperContext>,
    states: Arc<StatePool<WhisperState>>,
    inference: InferenceConfig,
    trimmer: Option<Arc<SpeechTrimmer>>,
    is_gpu: bool,
}

impl WhisperTranscriber {
    /// A builder with nothing set; a model is required before `build`.
    pub fn builder() -> WhisperTranscriberBuilder {
        WhisperTranscriberBuilder::default()
    }
}

#[async_trait]
impl Transcriber for WhisperTranscriber {
    fn is_gpu(&self) -> bool {
        self.is_gpu
    }

    async fn transcribe(&self, pcm_i16_16k_mono: &[i16]) -> Result<String, TranscriptionError> {
        if pcm_i16_16k_mono.is_empty() {
            return Err(TranscriptionError::EmptyAudio);
        }
        tracing::debug!(
            target: "assistd::voice::latency",
            stage = "whisper_start",
            "voice latency stage"
        );
        let mut audio_f32 = vec![0.0f32; pcm_i16_16k_mono.len()];
        convert_integer_to_float_audio(pcm_i16_16k_mono, &mut audio_f32)
            .map_err(|err| TranscriptionError::WhisperInference(err.to_string()))?;
        let ctx = self.ctx.clone();
        let states = self.states.clone();
        let inference = self.inference.clone();
        let trimmer = self.trimmer.clone();
        let result = tokio::task::spawn_blocking(move || {
            let speech = match &trimmer {
                Some(trimmer) => trimmer.trim(&audio_f32)?,
                None => audio_f32,
            };
            if speech.is_empty() {
                return Ok(String::new());
            }
            states.with_state(
                || {
                    ctx.create_state()
                        .map_err(|err| TranscriptionError::WhisperInference(err.to_string()))
                },
                |state| run_inference(state, &inference, &speech),
            )
        })
        .await?;
        tracing::debug!(
            target: "assistd::voice::latency",
            stage = "whisper_done",
            "voice latency stage"
        );
        result
    }
}

/// Builder for [`WhisperTranscriber`]. `build()` is async because it
/// may download model files.
#[derive(Debug, Default, Clone)]
pub struct WhisperTranscriberBuilder {
    model: Option<String>,
    cache_dir: Option<PathBuf>,
    prefer_gpu: bool,
    threads: Option<u32>,
    beams: u32,
    vad_enabled: bool,
    vad_model: Option<String>,
    vad_silence_secs: f32,
}

impl WhisperTranscriberBuilder {
    /// HuggingFace identifier for the Whisper GGML model, e.g.
    /// `"ggerganov/whisper.cpp:ggml-large-v3-turbo-q5_0.bin"`.
    pub fn model(mut self, id: impl Into<String>) -> Self {
        self.model = Some(id.into());
        self
    }

    /// Override the on-disk model cache directory.
    pub fn cache_dir(mut self, dir: Option<PathBuf>) -> Self {
        self.cache_dir = dir;
        self
    }

    /// Prefer GPU inference when CUDA is available.
    pub fn prefer_gpu(mut self, prefer: bool) -> Self {
        self.prefer_gpu = prefer;
        self
    }

    /// Override the number of CPU inference threads. `None` lets whisper.cpp choose.
    pub fn threads(mut self, threads: Option<u32>) -> Self {
        self.threads = threads;
        self
    }

    /// Number of beams for beam-search decoding; values ≤ 1 use greedy decoding.
    pub fn beams(mut self, beams: u32) -> Self {
        self.beams = beams.max(1);
        self
    }

    /// Trim non-speech audio with Silero VAD before decoding.
    pub fn vad_enabled(mut self, enabled: bool) -> Self {
        self.vad_enabled = enabled;
        self
    }

    /// HuggingFace identifier for the VAD GGML model, e.g.
    /// `"ggml-org/whisper-vad:ggml-silero-v6.2.0.bin"`. Only consulted when
    /// `vad_enabled` is true.
    pub fn vad_model(mut self, id: impl Into<String>) -> Self {
        self.vad_model = Some(id.into());
        self
    }

    /// Minimum silence duration (seconds) that Silero VAD uses to trim trailing silence.
    pub fn vad_silence_secs(mut self, secs: f32) -> Self {
        self.vad_silence_secs = secs;
        self
    }

    /// Populate from a [`TranscriptionConfig`].
    pub fn from_config(config: &TranscriptionConfig) -> Self {
        Self {
            model: Some(config.model.clone()),
            cache_dir: config.model_cache_dir.clone(),
            prefer_gpu: config.prefer_gpu,
            threads: config.threads.map(NonZeroU32::get),
            beams: config.beams.get(),
            vad_enabled: config.vad_enabled,
            vad_model: Some(config.vad_model.clone()),
            vad_silence_secs: VAD_SILENCE_SECS,
        }
    }

    /// Download any missing model files and initialize the whisper
    /// context. Errors when a required model identifier is missing,
    /// a download fails, or the context cannot be initialized.
    pub async fn build(self) -> Result<WhisperTranscriber, TranscriptionError> {
        whisper_rs::install_logging_hooks();

        let model = self.model.ok_or_else(|| TranscriptionError::ModelParse {
            id: String::new(),
            reason: "model identifier is required".into(),
        })?;
        let cache_dir = self
            .cache_dir
            .map_or_else(|| hf_download::default_cache_dir("whisper"), Ok)?;

        let model_path = hf_download::ensure_cached(&model, &cache_dir).await?;
        let vad = if self.vad_enabled {
            Some(fetch_vad(self.vad_model, self.vad_silence_secs, &cache_dir).await?)
        } else {
            None
        };

        let use_gpu = should_use_gpu(self.prefer_gpu);
        let ctx = load_context(&model_path, use_gpu).await?;
        let trimmer = match vad {
            Some(vad) => Some(Arc::new(load_trimmer(vad, use_gpu).await?)),
            None => None,
        };

        Ok(WhisperTranscriber {
            ctx: Arc::new(ctx),
            states: Arc::default(),
            inference: InferenceConfig {
                threads: self.threads,
                beams: self.beams.max(1),
            },
            trimmer,
            is_gpu: use_gpu,
        })
    }
}

/// Build a CPU-backed [`WhisperTranscriber`] from the same config as
/// the primary, sharing its cached model files.
pub async fn build_cpu_fallback(
    config: &TranscriptionConfig,
    cache_dir_override: Option<PathBuf>,
) -> Result<WhisperTranscriber, TranscriptionError> {
    let cache_dir = cache_dir_override.or_else(|| config.model_cache_dir.clone());
    WhisperTranscriberBuilder::from_config(config)
        .cache_dir(cache_dir)
        .prefer_gpu(false)
        .build()
        .await
}

fn run_inference(
    state: &mut WhisperState,
    inference: &InferenceConfig,
    audio: &[f32],
) -> Result<String, TranscriptionError> {
    let mut params = FullParams::new(sampling_strategy(inference.beams));
    params.set_language(Some("en"));
    params.set_translate(false);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_no_context(true);
    params.set_suppress_blank(true);
    if let Some(threads) = inference.threads {
        params.set_n_threads(threads as i32);
    }

    state
        .full(params, audio)
        .map_err(|err| TranscriptionError::WhisperInference(err.to_string()))?;

    let mut transcript = String::new();
    for index in 0..state.full_n_segments() {
        let Some(segment) = state.get_segment(index) else {
            continue;
        };
        let text = segment
            .to_str_lossy()
            .map_err(|err| TranscriptionError::WhisperInference(err.to_string()))?;
        transcript.push_str(text.as_ref());
    }
    Ok(transcript.trim().to_string())
}

fn sampling_strategy(beams: u32) -> SamplingStrategy {
    if beams <= 1 {
        SamplingStrategy::Greedy { best_of: 1 }
    } else {
        SamplingStrategy::BeamSearch {
            beam_size: beams as i32,
            patience: -1.0,
        }
    }
}

fn silero_vad_params(silence_secs: f32) -> WhisperVadParams {
    let mut vad_params = WhisperVadParams::default();
    let silence_ms = (silence_secs * 1000.0).round().clamp(0.0, i32::MAX as f32) as i32;
    vad_params.set_min_silence_duration(silence_ms);
    vad_params
}

/// Concatenates the audio covered by `segments`, extending every non-final
/// segment by the overlap whisper.cpp uses and separating segments with a
/// short silence, so the decoder sees natural pauses instead of hard cuts.
fn stitch_speech_segments(audio: &[f32], segments: &[WhisperVadSegment]) -> Vec<f32> {
    let overlap_samples = secs_to_samples(SEGMENT_OVERLAP_SECS);
    let gap = vec![0.0f32; secs_to_samples(SEGMENT_GAP_SECS)];
    let last_index = segments.len().saturating_sub(1);
    let mut stitched = Vec::new();
    for (index, segment) in segments.iter().enumerate() {
        let start = centiseconds_to_samples(segment.start).min(audio.len());
        let mut end = centiseconds_to_samples(segment.end);
        if index < last_index {
            end += overlap_samples;
        }
        let end = end.min(audio.len()).max(start);
        if index > 0 && end > start {
            stitched.extend_from_slice(&gap);
        }
        stitched.extend_from_slice(&audio[start..end]);
    }
    stitched
}

fn centiseconds_to_samples(centiseconds: f32) -> usize {
    (centiseconds * SAMPLES_PER_CENTISECOND).round().max(0.0) as usize
}

fn secs_to_samples(secs: f32) -> usize {
    (secs * SAMPLE_RATE_HZ as f32).round().max(0.0) as usize
}

async fn fetch_vad(
    vad_model: Option<String>,
    silence_secs: f32,
    cache_dir: &Path,
) -> Result<SileroVadParams, TranscriptionError> {
    let vad_id = vad_model.ok_or_else(|| TranscriptionError::ModelParse {
        id: String::new(),
        reason: "vad_model identifier is required when vad_enabled".into(),
    })?;
    let vad_path = hf_download::ensure_cached(&vad_id, cache_dir).await?;
    Ok(SileroVadParams {
        model_path: vad_path.to_string_lossy().into_owned(),
        silence_secs: silence_secs.max(0.0),
    })
}

async fn load_trimmer(
    params: SileroVadParams,
    use_gpu: bool,
) -> Result<SpeechTrimmer, TranscriptionError> {
    tokio::task::spawn_blocking(move || SpeechTrimmer::load(&params, use_gpu)).await?
}

async fn load_context(
    model_path: &Path,
    use_gpu: bool,
) -> Result<WhisperContext, TranscriptionError> {
    let model_path = model_path.to_string_lossy().into_owned();
    tokio::task::spawn_blocking(move || {
        let mut params = WhisperContextParameters::new();
        params.use_gpu(use_gpu);
        WhisperContext::new_with_params(&model_path, params)
    })
    .await?
    .map_err(|err| TranscriptionError::WhisperInit(err.to_string()))
}

fn should_use_gpu(prefer: bool) -> bool {
    if !cfg!(feature = "cuda") {
        tracing::info!(
            target: "assistd::voice::whisper",
            "built without CUDA; transcribing on CPU"
        );
        return false;
    }
    if !prefer {
        tracing::info!(
            target: "assistd::voice::whisper",
            "prefer_gpu=false; transcribing on CPU"
        );
        return false;
    }
    if gpu::probe_cuda_available() {
        tracing::info!(
            target: "assistd::voice::whisper",
            "CUDA GPU available; transcribing on GPU"
        );
        true
    } else {
        tracing::warn!(
            target: "assistd::voice::whisper",
            "No CUDA GPU available, falling back to CPU transcription"
        );
        false
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use whisper_rs::WhisperVadSegment;

    use super::{StatePool, stitch_speech_segments};

    fn counting_create(created: &Cell<u32>) -> impl FnOnce() -> Result<u32, &'static str> + '_ {
        move || {
            created.set(created.get() + 1);
            Ok(created.get())
        }
    }

    #[test]
    fn successful_runs_reuse_one_state() {
        let pool = StatePool::default();
        let created = Cell::new(0);
        for _ in 0..3 {
            let id = pool
                .with_state(counting_create(&created), |s| Ok::<_, &str>(*s))
                .unwrap();
            assert_eq!(id, 1);
        }
        assert_eq!(created.get(), 1);
    }

    #[test]
    fn failed_run_discards_its_state() {
        let pool = StatePool::default();
        let created = Cell::new(0);
        let err = pool
            .with_state(counting_create(&created), |_| Err::<(), _>("boom"))
            .unwrap_err();
        assert_eq!(err, "boom");
        let id = pool
            .with_state(counting_create(&created), |s| Ok::<_, &str>(*s))
            .unwrap();
        assert_eq!(id, 2, "a fresh state replaces the failed one");
    }

    #[test]
    fn overlapping_runs_get_distinct_states() {
        let pool = StatePool::default();
        let created = Cell::new(0);
        let (outer, inner) = pool
            .with_state(counting_create(&created), |outer| {
                let inner = pool.with_state(counting_create(&created), |s| Ok::<_, &str>(*s))?;
                Ok((*outer, inner))
            })
            .unwrap();
        assert_ne!(outer, inner);
        for _ in 0..2 {
            pool.with_state(counting_create(&created), |_| Ok::<_, &str>(()))
                .unwrap();
        }
        assert_eq!(created.get(), 2, "both states are pooled for reuse");
    }

    #[test]
    fn create_failure_propagates() {
        let pool: StatePool<u32> = StatePool::default();
        let err = pool.with_state(|| Err("no state"), |_| Ok(())).unwrap_err();
        assert_eq!(err, "no state");
    }

    fn ramp(len: usize) -> Vec<f32> {
        (0..len).map(|i| i as f32).collect()
    }

    fn segment(start_cs: f32, end_cs: f32) -> WhisperVadSegment {
        WhisperVadSegment {
            start: start_cs,
            end: end_cs,
        }
    }

    #[test]
    fn no_segments_yields_empty_audio() {
        assert!(stitch_speech_segments(&ramp(16_000), &[]).is_empty());
    }

    #[test]
    fn single_segment_is_sliced_without_overlap() {
        let audio = ramp(16_000);
        let stitched = stitch_speech_segments(&audio, &[segment(10.0, 30.0)]);
        assert_eq!(stitched, audio[1_600..4_800]);
    }

    #[test]
    fn segments_are_joined_with_overlap_and_a_silence_gap() {
        let audio = ramp(16_000);
        let stitched = stitch_speech_segments(&audio, &[segment(0.0, 20.0), segment(50.0, 60.0)]);
        let first_len = 3_200 + 1_600;
        let gap_len = 1_600;
        assert_eq!(stitched.len(), first_len + gap_len + 1_600);
        assert_eq!(stitched[..first_len], audio[..first_len]);
        assert!(
            stitched[first_len..first_len + gap_len]
                .iter()
                .all(|sample| *sample == 0.0)
        );
        assert_eq!(stitched[first_len + gap_len..], audio[8_000..9_600]);
    }

    #[test]
    fn segment_bounds_are_clamped_to_the_audio() {
        let audio = ramp(8_000);
        let stitched =
            stitch_speech_segments(&audio, &[segment(40.0, 90.0), segment(100.0, 120.0)]);
        assert_eq!(stitched, audio[6_400..]);
    }
}
