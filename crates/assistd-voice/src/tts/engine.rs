//! [`SpeechEngine`]: one sherpa-onnx voice loaded in-process, and the
//! frontend that turns text into its input.

use std::num::NonZeroU32;
use std::path::Path;

use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsKokoroModelConfig,
    OfflineTtsModelConfig, OfflineTtsVitsModelConfig,
};

use crate::tts::error::SpeechError;
use crate::tts::phonemize::Phonemizer;
use crate::tts::voice_files::VoiceFiles;

/// ONNX Runtime threads. One thread starts no spinning pool, so synthesis
/// stays off the cores the LLM's CPU layers use.
const ONNX_THREADS: i32 = 1;

/// VITS sampling noise, at the value Piper recommends.
const NOISE_SCALE: f32 = 0.667;
/// VITS phoneme-duration noise, at the value Piper recommends.
const NOISE_SCALE_W: f32 = 0.8;

/// One sentence of mono audio in `-1.0..=1.0`.
#[derive(Debug, Clone)]
pub struct Utterance {
    pub sample_rate: NonZeroU32,
    pub samples: Vec<f32>,
}

/// A loaded voice. Synthesis blocks, so call it off the async runtime.
pub struct SpeechEngine {
    tts: OfflineTts,
    frontend: Frontend,
    generation: GenerationConfig,
    sample_rate: NonZeroU32,
}

impl std::fmt::Debug for SpeechEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeechEngine")
            .field("sample_rate", &self.sample_rate)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
enum Frontend {
    /// Phonemes from `espeak-ng`, for Piper voices.
    Espeak(Phonemizer),
    /// The model's own lexicon, inside sherpa-onnx.
    Lexicon,
}

impl SpeechEngine {
    /// Load `voice` with sherpa-onnx. Fails when sherpa rejects the files,
    /// `speaker_id` is out of range, or a VITS voice finds no `espeak-ng`.
    pub fn load(
        voice: &VoiceFiles,
        speaker_id: u32,
        length_scale: f32,
    ) -> Result<Self, SpeechError> {
        let load_failed = |reason: String| SpeechError::LoadFailed { reason };
        let tts = OfflineTts::create(&tts_config(voice, length_scale)).ok_or_else(|| {
            load_failed("sherpa-onnx rejected the voice files; see the log".into())
        })?;
        let sample_rate = u32::try_from(tts.sample_rate())
            .ok()
            .and_then(NonZeroU32::new)
            .ok_or_else(|| load_failed("voice reports no sample rate".into()))?;
        let speakers = u32::try_from(tts.num_speakers()).unwrap_or(0).max(1);
        if speaker_id >= speakers {
            return Err(load_failed(format!(
                "speaker_id {speaker_id} is out of range; this voice has {speakers} speaker(s)"
            )));
        }
        let frontend = match voice {
            VoiceFiles::Vits { .. } => {
                Frontend::Espeak(Phonemizer::new(tts.lang()).map_err(load_failed)?)
            }
            VoiceFiles::Kokoro { .. } => Frontend::Lexicon,
        };
        let generation = GenerationConfig {
            sid: i32::try_from(speaker_id).unwrap_or(i32::MAX),
            ..GenerationConfig::default()
        };
        Ok(Self {
            tts,
            frontend,
            generation,
            sample_rate,
        })
    }

    /// Synthesize one sentence; text with nothing to speak gives no samples.
    pub fn synthesize(&self, sentence: &str) -> Result<Utterance, SpeechError> {
        let rejected = |reason: String| SpeechError::Rejected { reason };
        let text: String = sentence
            .chars()
            .map(|c| if c.is_control() { ' ' } else { c })
            .collect();
        let generation = match &self.frontend {
            Frontend::Lexicon => self.generation.clone(),
            Frontend::Espeak(phonemizer) => {
                let sentences = phonemizer.sentences(&text).map_err(rejected)?;
                if sentences.is_empty() {
                    return Ok(self.utterance(Vec::new()));
                }
                GenerationConfig {
                    phoneme_codepoints: Some(sentences),
                    ..self.generation.clone()
                }
            }
        };
        let audio = self
            .tts
            .generate_with_config::<fn(&[f32], f32) -> bool>(&text, &generation, None)
            .ok_or_else(|| rejected("sherpa-onnx produced no audio".into()))?;
        Ok(self.utterance(audio.samples().to_vec()))
    }

    fn utterance(&self, samples: Vec<f32>) -> Utterance {
        Utterance {
            sample_rate: self.sample_rate,
            samples,
        }
    }
}

fn tts_config(voice: &VoiceFiles, length_scale: f32) -> OfflineTtsConfig {
    let mut model = OfflineTtsModelConfig {
        num_threads: ONNX_THREADS,
        provider: Some("cpu".into()),
        ..OfflineTtsModelConfig::default()
    };
    match voice {
        VoiceFiles::Vits {
            model: onnx,
            tokens,
        } => {
            model.vits = OfflineTtsVitsModelConfig {
                model: Some(path_string(onnx)),
                tokens: Some(path_string(tokens)),
                noise_scale: NOISE_SCALE,
                noise_scale_w: NOISE_SCALE_W,
                length_scale,
                ..OfflineTtsVitsModelConfig::default()
            };
        }
        VoiceFiles::Kokoro {
            model: onnx,
            voices,
            tokens,
            lexicon,
        } => {
            model.kokoro = OfflineTtsKokoroModelConfig {
                model: Some(path_string(onnx)),
                voices: Some(path_string(voices)),
                tokens: Some(path_string(tokens)),
                lexicon: Some(path_string(lexicon)),
                length_scale,
                ..OfflineTtsKokoroModelConfig::default()
            };
        }
    }
    OfflineTtsConfig {
        model,
        ..OfflineTtsConfig::default()
    }
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}
