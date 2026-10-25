//! [`WakeWordSpotter`]: sherpa-onnx keyword spotting for the wake phrase,
//! run over one whole utterance at a time.

use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};

use assistd_config::VoiceConfig;
use assistd_utils::hf::download::{self, DownloadError};
use sentencepiece_rust::SentencePieceProcessor;
use sherpa_onnx::{KeywordSpotter, KeywordSpotterConfig};
use thiserror::Error;

use crate::vad::{SAMPLE_RATE_HZ, secs_to_samples};

/// The 3.3M-parameter `GigaSpeech` keyword model, pinned to a mirror whose files match
/// the k2-fsa release byte for byte.
const MODEL_REPO: &str = "mobilebytesensei/sherpa-onnx-kws-zipformer-gigaspeech-3.3M\
     @e8ca74a794ce66e38a100e7c60c0214189ff4100";
const ENCODER: &str = "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx";
const DECODER: &str = "decoder-epoch-12-avg-2-chunk-16-left-64.onnx";
const JOINER: &str = "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx";
const TOKENS: &str = "tokens.txt";
const BPE_MODEL: &str = "bpe.model";

/// sherpa-onnx's spelling of a piece the tokenizer could not place.
const UNKNOWN_PIECE: &str = "<unk>";

/// Seconds after the phrase's last token's start that the token is taken
/// to end, where the command begins.
const LAST_TOKEN_SECS: f32 = 0.15;

/// Silence appended to each utterance so a phrase said last still gets the
/// trailing blank the spotter needs before it reports a match.
const TRAILING_SILENCE_SECS: f32 = 0.3;

/// Why the spotter could not be set up for the configured phrase.
#[derive(Debug, Error)]
pub enum SpotterError {
    #[error(transparent)]
    Download(#[from] DownloadError),

    #[error("could not read the keyword tokenizer: {0}")]
    Tokenizer(String),

    #[error(
        "the keyword model cannot spell voice.wakeword.phrase {phrase:?}: it has no piece {piece:?}"
    )]
    Unspellable { phrase: String, piece: String },

    #[error("sherpa-onnx could not load the keyword model in {}", dir.display())]
    Load { dir: PathBuf },

    #[error("keyword model load task failed: {0}")]
    Join(#[from] tokio::task::JoinError),
}

/// Finds the wake phrase in an utterance.
pub struct WakeWordSpotter {
    spotter: KeywordSpotter,
    phrase: String,
}

impl fmt::Debug for WakeWordSpotter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WakeWordSpotter")
            .field("phrase", &self.phrase)
            .finish_non_exhaustive()
    }
}

impl WakeWordSpotter {
    /// Download the keyword model if missing and load it for
    /// `config.wakeword.phrase`. Fails when the model cannot spell the phrase,
    /// which sherpa-onnx would otherwise answer by ending the process.
    pub async fn load(config: &VoiceConfig) -> Result<Self, SpotterError> {
        let cache_dir = config
            .transcription
            .model_cache_dir
            .clone()
            .map_or_else(|| download::default_cache_dir("stt"), Ok)?;
        let mut dir = None;
        for name in [ENCODER, DECODER, JOINER, TOKENS, BPE_MODEL] {
            let path = download::ensure_cached(&format!("{MODEL_REPO}:{name}"), &cache_dir).await?;
            dir = path.parent().map(Path::to_path_buf);
        }
        let dir = dir.unwrap_or(cache_dir);
        let phrase = config.wakeword.phrase.clone();
        let threshold = config.wakeword.threshold.get();
        tokio::task::spawn_blocking(move || Self::create(&dir, phrase, threshold)).await?
    }

    fn create(dir: &Path, phrase: String, threshold: f32) -> Result<Self, SpotterError> {
        let pieces = spell(dir, &phrase)?;
        let file = |name: &str| Some(dir.join(name).to_string_lossy().into_owned());
        let mut config = KeywordSpotterConfig::default();
        config.model_config.transducer.encoder = file(ENCODER);
        config.model_config.transducer.decoder = file(DECODER);
        config.model_config.transducer.joiner = file(JOINER);
        config.model_config.tokens = file(TOKENS);
        config.model_config.num_threads = 1;
        config.keywords_threshold = threshold;
        config.keywords_buf = Some(format!(
            "{} @{}",
            pieces.join(" "),
            phrase.replace(' ', "_")
        ));
        let spotter = KeywordSpotter::create(&config).ok_or_else(|| SpotterError::Load {
            dir: dir.to_path_buf(),
        })?;
        Ok(Self { spotter, phrase })
    }

    /// The phrase as configured.
    pub fn phrase(&self) -> &str {
        &self.phrase
    }

    /// The sample of 16 kHz `pcm` where the wake phrase ends, or `None` when
    /// the phrase is not in it. Each call starts from a fresh stream.
    pub fn find(&self, pcm: &[i16]) -> Option<usize> {
        let stream = self.spotter.create_stream();
        let mut audio: Vec<f32> = pcm
            .iter()
            .map(|&sample| f32::from(sample) / 32_768.0)
            .collect();
        audio.resize(audio.len() + secs_to_samples(TRAILING_SILENCE_SECS), 0.0);
        stream.accept_waveform(SAMPLE_RATE_HZ, &audio);
        stream.input_finished();
        while self.spotter.is_ready(&stream) {
            self.spotter.decode(&stream);
            if let Some(result) = self.spotter.get_result(&stream)
                && !result.keyword.is_empty()
            {
                let last_token = result.timestamps.last().copied().unwrap_or(0.0);
                return Some(secs_to_samples(last_token + LAST_TOKEN_SECS).min(pcm.len()));
            }
        }
        None
    }
}

/// The model's pieces for `phrase`, refusing any it does not have.
fn spell(dir: &Path, phrase: &str) -> Result<Vec<String>, SpotterError> {
    let tokenizer = SentencePieceProcessor::open(dir.join(BPE_MODEL))
        .map_err(|err| SpotterError::Tokenizer(err.to_string()))?;
    let pieces = tokenizer
        .encode_pieces(&phrase.to_uppercase())
        .map_err(|err| SpotterError::Tokenizer(err.to_string()))?;
    let tokens = std::fs::read_to_string(dir.join(TOKENS))
        .map_err(|err| SpotterError::Tokenizer(err.to_string()))?;
    let vocabulary: HashSet<&str> = tokens
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .collect();
    if let Some(piece) = pieces
        .iter()
        .find(|piece| piece.as_str() == UNKNOWN_PIECE || !vocabulary.contains(piece.as_str()))
    {
        return Err(SpotterError::Unspellable {
            phrase: phrase.to_string(),
            piece: piece.clone(),
        });
    }
    Ok(pieces)
}
