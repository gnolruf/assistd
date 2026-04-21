//! Whisper-rs-backed `Transcriber`. Downloads the model on first use,
//! chooses GPU or CPU inference, and wraps whisper.cpp's native Silero
//! VAD for silence trimming.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperVadParams,
    convert_integer_to_float_audio,
};

use crate::gpu;
use crate::model_cache::{self, default_cache_dir};
use crate::transcribe::{Transcriber, TranscriptionError};

/// Runtime-configurable knobs used by each `transcribe` call.
#[derive(Debug, Clone)]
struct InferenceConfig {
    threads: Option<u32>,
    beams: u32,
    vad: Option<VadRuntime>,
}

#[derive(Debug, Clone)]
struct VadRuntime {
    model_path: String,
    silence_secs: f32,
}

/// Concrete [`Transcriber`] backed by whisper.cpp via whisper-rs.
pub struct WhisperTranscriber {
    ctx: Arc<WhisperContext>,
    cfg: InferenceConfig,
}

impl WhisperTranscriber {
    pub fn builder() -> WhisperTranscriberBuilder {
        WhisperTranscriberBuilder::default()
    }
}

#[async_trait]
impl Transcriber for WhisperTranscriber {
    async fn transcribe(&self, pcm_i16_16k_mono: &[i16]) -> Result<String, TranscriptionError> {
        if pcm_i16_16k_mono.is_empty() {
            return Err(TranscriptionError::EmptyAudio);
        }
        let mut audio_f32 = vec![0.0f32; pcm_i16_16k_mono.len()];
        convert_integer_to_float_audio(pcm_i16_16k_mono, &mut audio_f32)
            .map_err(|err| TranscriptionError::WhisperInference(err.to_string()))?;
        let ctx = self.ctx.clone();
        let cfg = self.cfg.clone();
        tokio::task::spawn_blocking(move || run_inference(ctx, cfg, audio_f32)).await?
    }
}

fn run_inference(
    ctx: Arc<WhisperContext>,
    cfg: InferenceConfig,
    audio: Vec<f32>,
) -> Result<String, TranscriptionError> {
    let mut state = ctx
        .create_state()
        .map_err(|err| TranscriptionError::WhisperInference(err.to_string()))?;

    let strategy = if cfg.beams <= 1 {
        SamplingStrategy::Greedy { best_of: 1 }
    } else {
        SamplingStrategy::BeamSearch {
            beam_size: cfg.beams as i32,
            patience: -1.0,
        }
    };
    let mut params = FullParams::new(strategy);
    params.set_language(Some("en"));
    params.set_translate(false);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_no_context(true);
    params.set_suppress_blank(true);
    if let Some(threads) = cfg.threads {
        params.set_n_threads(threads as i32);
    }
    if let Some(vad) = &cfg.vad {
        params.enable_vad(true);
        params.set_vad_model_path(Some(vad.model_path.as_str()));
        let mut vp = WhisperVadParams::default();
        let ms = (vad.silence_secs * 1000.0)
            .round()
            .clamp(0.0, i32::MAX as f32) as i32;
        vp.set_min_silence_duration(ms);
        params.set_vad_params(vp);
    }

    state
        .full(params, &audio)
        .map_err(|err| TranscriptionError::WhisperInference(err.to_string()))?;

    let n = state.full_n_segments();
    let mut out = String::new();
    for i in 0..n {
        let Some(segment) = state.get_segment(i) else {
            continue;
        };
        let text = segment
            .to_str_lossy()
            .map_err(|err| TranscriptionError::WhisperInference(err.to_string()))?;
        out.push_str(text.as_ref());
    }
    Ok(out.trim().to_string())
}

/// Builder for [`WhisperTranscriber`]. `build()` is async because it may
/// download model files and probe NVML.
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

    pub fn prefer_gpu(mut self, prefer: bool) -> Self {
        self.prefer_gpu = prefer;
        self
    }

    pub fn threads(mut self, threads: Option<u32>) -> Self {
        self.threads = threads;
        self
    }

    pub fn beams(mut self, beams: u32) -> Self {
        self.beams = beams.max(1);
        self
    }

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

    pub fn vad_silence_secs(mut self, secs: f32) -> Self {
        self.vad_silence_secs = secs;
        self
    }

    /// Populate from a [`TranscriptionConfig`] for ergonomic wiring from
    /// the daemon.
    pub fn from_config(cfg: &assistd_config::TranscriptionConfig) -> Self {
        Self {
            model: Some(cfg.model.clone()),
            cache_dir: cfg.model_cache_dir.clone(),
            prefer_gpu: cfg.prefer_gpu,
            threads: cfg.threads,
            beams: cfg.beams.max(1),
            vad_enabled: cfg.vad_enabled,
            vad_model: Some(cfg.vad_model.clone()),
            vad_silence_secs: cfg.vad_silence_secs,
        }
    }

    pub async fn build(self) -> Result<WhisperTranscriber, TranscriptionError> {
        let model = self.model.ok_or_else(|| TranscriptionError::ModelParse {
            id: String::new(),
            reason: "model identifier is required".into(),
        })?;
        let cache_dir = self.cache_dir.unwrap_or_else(default_cache_dir);

        let model_path = model_cache::ensure_model(&model, &cache_dir).await?;
        let vad_runtime = if self.vad_enabled {
            let vad_id = self
                .vad_model
                .ok_or_else(|| TranscriptionError::ModelParse {
                    id: String::new(),
                    reason: "vad_model identifier is required when vad_enabled".into(),
                })?;
            let vad_path = model_cache::ensure_model(&vad_id, &cache_dir).await?;
            Some(VadRuntime {
                model_path: vad_path.to_string_lossy().into_owned(),
                silence_secs: self.vad_silence_secs.max(0.0),
            })
        } else {
            None
        };

        let use_gpu = decide_use_gpu(self.prefer_gpu);
        let model_path_str = model_path.to_string_lossy().into_owned();
        let ctx = tokio::task::spawn_blocking(move || {
            let mut params = WhisperContextParameters::new();
            params.use_gpu(use_gpu);
            WhisperContext::new_with_params(&model_path_str, params)
        })
        .await?
        .map_err(|err| TranscriptionError::WhisperInit(err.to_string()))?;

        Ok(WhisperTranscriber {
            ctx: Arc::new(ctx),
            cfg: InferenceConfig {
                threads: self.threads,
                beams: self.beams.max(1),
                vad: vad_runtime,
            },
        })
    }
}

fn decide_use_gpu(prefer: bool) -> bool {
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
