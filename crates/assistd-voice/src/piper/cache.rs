//! On-disk cache for Piper voices: a `.onnx` model and the `.onnx.json`
//! config piper expects beside it.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::hf_download::{self, cached_path, ensure_file, parse_hf_id};
use crate::piper::error::PiperError;

pub fn default_cache_dir() -> PathBuf {
    hf_download::default_cache_dir("piper")
}

/// Resolved on-disk paths for a voice, plus the sample rate read from
/// its `.onnx.json`.
#[derive(Debug, Clone)]
pub struct VoiceFiles {
    pub onnx: PathBuf,
    pub json: PathBuf,
    pub sample_rate: u32,
}

/// Ensure both voice files exist locally, downloading whichever is
/// missing.
pub async fn ensure_voice(hf_id: &str, cache_dir: &Path) -> Result<VoiceFiles, PiperError> {
    let (repo, file) = parse_hf_id(hf_id)?;
    let onnx = cached_path(cache_dir, &repo, &file);
    let json = onnx.with_extension("onnx.json");

    ensure_file(&repo, &file, &onnx).await?;
    ensure_file(&repo, &format!("{file}.json"), &json).await?;

    let sample_rate = read_sample_rate(&json).await?;
    Ok(VoiceFiles {
        onnx,
        json,
        sample_rate,
    })
}

#[derive(Deserialize)]
struct AudioConfig {
    sample_rate: u32,
}

#[derive(Deserialize)]
struct VoiceConfigJson {
    audio: AudioConfig,
}

async fn read_sample_rate(json: &Path) -> Result<u32, PiperError> {
    let body = tokio::fs::read_to_string(json)
        .await
        .map_err(|source| PiperError::Io {
            path: json.to_path_buf(),
            source,
        })?;

    // HuggingFace serves a 200-OK HTML "not found" page for missing files.
    let trimmed = body.trim_start();
    if !trimmed.starts_with('{') {
        let prefix: String = trimmed.chars().take(40).collect();
        return Err(PiperError::JsonShape {
            path: json.to_path_buf(),
            prefix,
        });
    }

    let cfg: VoiceConfigJson =
        serde_json::from_str(&body).map_err(|source| PiperError::JsonParse {
            path: json.to_path_buf(),
            source,
        })?;
    Ok(cfg.audio.sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn read_sample_rate_parses_audio_section() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("voice.onnx.json");
        tokio::fs::write(
            &path,
            r#"{"audio": {"sample_rate": 22050, "quality": "medium"}, "phoneme_id_map": {}}"#,
        )
        .await
        .unwrap();
        assert_eq!(read_sample_rate(&path).await.unwrap(), 22050);
    }

    #[tokio::test]
    async fn read_sample_rate_rejects_html_body() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("voice.onnx.json");
        tokio::fs::write(&path, "<!doctype html><html>not found</html>")
            .await
            .unwrap();
        let err = read_sample_rate(&path).await.unwrap_err();
        assert!(matches!(err, PiperError::JsonShape { .. }), "got {err:?}");
    }
}
