//! [`SherpaTranscriber`]: Parakeet transducer speech-to-text through
//! sherpa-onnx, on CUDA or the CPU.

use std::fmt;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use assistd_config::TranscriptionConfig;
use assistd_utils::hf::download::{self, DownloadError, ensure_cached_id, parse_id};
use async_trait::async_trait;
use parking_lot::Mutex;
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig};

use crate::gpu;
use crate::transcribe::{Transcriber, TranscriptionError};
use crate::vad::{self, SAMPLE_RATE_HZ, SpeechDetector, secs_to_samples};

/// Minimum trailing silence, in seconds, before Silero VAD ends a speech
/// segment.
pub const VAD_SILENCE_SECS: f32 = 0.5;

/// Shortest speech Silero VAD keeps when trimming.
const VAD_MIN_SPEECH_SECS: f32 = 0.25;

/// Trailing audio kept after each non-final speech segment.
const SEGMENT_OVERLAP_SECS: f32 = 0.1;
/// Silence inserted between stitched speech segments.
const SEGMENT_GAP_SECS: f32 = 0.1;

/// How the configured model file's name starts; its siblings swap this
/// for `decoder` and `joiner`.
const ENCODER_PREFIX: &str = "encoder";

/// The files of a Parakeet transducer on disk.
#[derive(Debug, Clone)]
struct ModelFiles {
    encoder: PathBuf,
    decoder: PathBuf,
    joiner: PathBuf,
    tokens: PathBuf,
}

/// Silero VAD run ahead of decoding, keeping only the speech segments.
#[derive(Debug)]
struct SpeechTrimmer {
    detector: Mutex<SpeechDetector>,
}

impl SpeechTrimmer {
    fn load(model: &Path) -> Result<Self, TranscriptionError> {
        Ok(Self {
            detector: Mutex::new(SpeechDetector::load(
                model,
                VAD_MIN_SPEECH_SECS,
                VAD_SILENCE_SECS,
            )?),
        })
    }

    /// Returns only the speech in `audio`, or an empty buffer when none was detected.
    fn trim(&self, audio: &[f32]) -> Vec<f32> {
        let segments = self.detector.lock().speech_ranges(audio);
        tracing::debug!(
            target: "assistd::voice::stt",
            segments = segments.len(),
            "silero vad detected speech segments"
        );
        stitch_speech_segments(audio, &segments)
    }
}

/// [`Transcriber`] backed by a sherpa-onnx offline recognizer.
pub struct SherpaTranscriber {
    recognizer: Arc<OfflineRecognizer>,
    trimmer: Option<Arc<SpeechTrimmer>>,
    is_gpu: bool,
}

impl fmt::Debug for SherpaTranscriber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SherpaTranscriber")
            .field("is_gpu", &self.is_gpu)
            .field("trimmer", &self.trimmer)
            .finish_non_exhaustive()
    }
}

impl SherpaTranscriber {
    /// Download `config`'s models if missing and load them, on CUDA when
    /// `prefer_gpu` is set and CUDA is usable, else on the CPU.
    pub async fn load(config: &TranscriptionConfig) -> Result<Self, TranscriptionError> {
        Self::load_on(config, use_gpu(config.prefer_gpu)).await
    }

    /// The same models on the CPU, for while the GPU is busy or asleep.
    pub async fn load_cpu(config: &TranscriptionConfig) -> Result<Self, TranscriptionError> {
        Self::load_on(config, false).await
    }

    async fn load_on(config: &TranscriptionConfig, gpu: bool) -> Result<Self, TranscriptionError> {
        let cache_dir = config
            .model_cache_dir
            .clone()
            .map_or_else(|| download::default_cache_dir("stt"), Ok)?;
        let files = ensure_model(&config.model, &cache_dir).await?;
        let vad_model = if config.vad_enabled {
            Some(vad::ensure_model(config).await?)
        } else {
            None
        };
        let threads = config.threads.get();
        let (recognizer, trimmer) = tokio::task::spawn_blocking(move || {
            let recognizer = create_recognizer(&files, gpu, threads)?;
            let trimmer = vad_model.as_deref().map(SpeechTrimmer::load).transpose()?;
            Ok::<_, TranscriptionError>((recognizer, trimmer))
        })
        .await??;
        Ok(Self {
            recognizer: Arc::new(recognizer),
            trimmer: trimmer.map(Arc::new),
            is_gpu: gpu,
        })
    }
}

#[async_trait]
impl Transcriber for SherpaTranscriber {
    fn is_gpu(&self) -> bool {
        self.is_gpu
    }

    async fn transcribe(&self, pcm_i16_16k_mono: &[i16]) -> Result<String, TranscriptionError> {
        if pcm_i16_16k_mono.is_empty() {
            return Err(TranscriptionError::EmptyAudio);
        }
        tracing::debug!(
            target: "assistd::voice::latency",
            stage = "stt_start",
            "voice latency stage"
        );
        let audio: Vec<f32> = pcm_i16_16k_mono
            .iter()
            .map(|&sample| f32::from(sample) / 32_768.0)
            .collect();
        let recognizer = Arc::clone(&self.recognizer);
        let trimmer = self.trimmer.clone();
        let transcript = tokio::task::spawn_blocking(move || {
            let speech = match &trimmer {
                Some(trimmer) => trimmer.trim(&audio),
                None => audio,
            };
            decode(&recognizer, &speech)
        })
        .await??;
        tracing::debug!(
            target: "assistd::voice::latency",
            stage = "stt_done",
            "voice latency stage"
        );
        Ok(transcript)
    }
}

/// Use CUDA only when asked to, a GPU is present, and this build ships the
/// CUDA libraries; sherpa-onnx would otherwise fall back to the CPU silently.
fn use_gpu(prefer: bool) -> bool {
    if !prefer {
        tracing::info!(target: "assistd::voice::stt", "prefer_gpu=false; transcribing on CPU");
        return false;
    }
    if !gpu::cuda_provider_installed() {
        tracing::info!(
            target: "assistd::voice::stt",
            "built without sherpa-onnx's CUDA libraries; transcribing on CPU"
        );
        return false;
    }
    if gpu::probe_cuda_available() {
        tracing::info!(target: "assistd::voice::stt", "CUDA GPU available; transcribing on GPU");
        true
    } else {
        tracing::warn!(
            target: "assistd::voice::stt",
            "no CUDA GPU available; transcribing on CPU"
        );
        false
    }
}

fn create_recognizer(
    files: &ModelFiles,
    gpu: bool,
    threads: u32,
) -> Result<OfflineRecognizer, TranscriptionError> {
    let mut config = OfflineRecognizerConfig::default();
    config.model_config.transducer = OfflineTransducerModelConfig {
        encoder: Some(path_string(&files.encoder)),
        decoder: Some(path_string(&files.decoder)),
        joiner: Some(path_string(&files.joiner)),
    };
    config.model_config.tokens = Some(path_string(&files.tokens));
    config.model_config.model_type = Some("nemo_transducer".into());
    config.model_config.num_threads = i32::try_from(threads).unwrap_or(i32::MAX);
    config.model_config.provider = Some(if gpu { "cuda" } else { "cpu" }.into());
    config.decoding_method = Some("greedy_search".into());
    OfflineRecognizer::create(&config).ok_or_else(|| {
        TranscriptionError::ModelLoad(format!(
            "sherpa-onnx could not load {}",
            files.encoder.display()
        ))
    })
}

/// Trimmed text of `speech`; empty audio gives an empty transcript.
fn decode(recognizer: &OfflineRecognizer, speech: &[f32]) -> Result<String, TranscriptionError> {
    if speech.is_empty() {
        return Ok(String::new());
    }
    let stream = recognizer.create_stream();
    stream.accept_waveform(SAMPLE_RATE_HZ, speech);
    recognizer.decode(&stream);
    let result = stream
        .get_result()
        .ok_or_else(|| TranscriptionError::Inference("recognizer returned no result".into()))?;
    Ok(result.text.trim().to_string())
}

/// Download the encoder `model_id` names and the decoder, joiner and tokens
/// beside it.
async fn ensure_model(model_id: &str, cache_dir: &Path) -> Result<ModelFiles, TranscriptionError> {
    let encoder_id = parse_id(model_id)?;
    let encoder_name = encoder_id.file().rsplit('/').next().unwrap_or_default();
    let suffix = encoder_name.strip_prefix(ENCODER_PREFIX).ok_or_else(|| {
        TranscriptionError::ModelLoad(format!(
            "{model_id} must name the encoder, a file starting with `{ENCODER_PREFIX}`"
        ))
    })?;
    let sibling = async |name: String| {
        let id = encoder_id
            .sibling(&name)
            .map_err(|reason| DownloadError::InvalidId {
                id: model_id.to_string(),
                reason,
            })?;
        ensure_cached_id(&id, cache_dir).await
    };
    Ok(ModelFiles {
        tokens: sibling("tokens.txt".into()).await?,
        decoder: sibling(format!("decoder{suffix}")).await?,
        joiner: sibling(format!("joiner{suffix}")).await?,
        encoder: ensure_cached_id(&encoder_id, cache_dir).await?,
    })
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// Concatenates the audio covered by `segments`, extending every non-final
/// segment by a short overlap and separating segments with a short silence,
/// so the decoder sees natural pauses instead of hard cuts.
fn stitch_speech_segments(audio: &[f32], segments: &[Range<usize>]) -> Vec<f32> {
    let overlap_samples = secs_to_samples(SEGMENT_OVERLAP_SECS);
    let gap = vec![0.0f32; secs_to_samples(SEGMENT_GAP_SECS)];
    let last_index = segments.len().saturating_sub(1);
    let mut stitched = Vec::new();
    for (index, segment) in segments.iter().enumerate() {
        let start = segment.start.min(audio.len());
        let mut end = segment.end;
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

#[cfg(test)]
mod tests {
    use super::stitch_speech_segments;

    fn ramp(len: usize) -> Vec<f32> {
        (0..len).map(|i| i as f32).collect()
    }

    #[test]
    fn segments_are_joined_with_overlap_and_a_silence_gap() {
        let audio = ramp(16_000);
        let stitched = stitch_speech_segments(&audio, &[0..3_200, 8_000..9_600]);
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
        let stitched = stitch_speech_segments(&audio, &[6_400..14_400, 16_000..19_200]);
        assert_eq!(stitched, audio[6_400..]);
    }
}
