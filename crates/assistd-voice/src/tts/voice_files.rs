//! On-disk cache for sherpa-onnx voices: the configured `.onnx` and the
//! files its model family needs from the same directory.

use std::path::PathBuf;

use assistd_config::{SynthesisConfig, VoiceFamily};
use assistd_utils::hf::download::{self, DownloadError, ensure_cached_id, parse_id};

use crate::tts::error::SpeechError;

/// HTTP status Hugging Face returns for a file the repo does not have.
const NOT_FOUND: u16 = 404;

/// A voice's files on disk, by model family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoiceFiles {
    Vits {
        model: PathBuf,
        tokens: PathBuf,
    },
    Kokoro {
        model: PathBuf,
        voices: PathBuf,
        tokens: PathBuf,
        lexicon: PathBuf,
    },
}

/// Download whichever of the voice's files are missing and return their paths.
pub async fn ensure_voice(config: &SynthesisConfig) -> Result<VoiceFiles, SpeechError> {
    let cache_dir = config
        .model_cache_dir
        .clone()
        .map_or_else(|| download::default_cache_dir("tts"), Ok)?;
    let model_id = parse_id(&config.voice)?;
    let sibling = async |name| {
        let id = model_id
            .sibling(name)
            .map_err(|reason| DownloadError::InvalidId {
                id: config.voice.clone(),
                reason,
            })?;
        ensure_cached_id(&id, &cache_dir).await
    };
    let tokens = sibling("tokens.txt")
        .await
        .map_err(|err| match config.family {
            VoiceFamily::Vits => not_found_as_stock_voice(err, &config.voice),
            VoiceFamily::Kokoro => err.into(),
        })?;
    Ok(match config.family {
        VoiceFamily::Vits => VoiceFiles::Vits {
            model: ensure_cached_id(&model_id, &cache_dir).await?,
            tokens,
        },
        VoiceFamily::Kokoro => VoiceFiles::Kokoro {
            voices: sibling("voices.bin").await?,
            lexicon: sibling("lexicon-us-en.txt").await?,
            model: ensure_cached_id(&model_id, &cache_dir).await?,
            tokens,
        },
    })
}

/// A VITS voice without `tokens.txt` is a stock Piper file, which would
/// make sherpa-onnx end the process on load; checked before the model downloads.
fn not_found_as_stock_voice(err: DownloadError, voice: &str) -> SpeechError {
    match err {
        DownloadError::Http {
            status: NOT_FOUND, ..
        } => SpeechError::NotSherpaExport {
            voice: voice.to_string(),
        },
        other => other.into(),
    }
}
