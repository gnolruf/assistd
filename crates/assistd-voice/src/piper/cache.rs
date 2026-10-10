//! On-disk cache for Piper voices: a `.onnx` model and the `.onnx.json`
//! config piper expects beside it.

use std::path::{Path, PathBuf};

use assistd_utils::hf::download::{self, DownloadError, cached_path, ensure_file, parse_id};
use serde::Deserialize;

use crate::piper::error::PiperError;

/// Resolved on-disk paths for a voice, plus the sample rate read from
/// its `.onnx.json`.
#[derive(Debug, Clone)]
pub struct VoiceFiles {
    pub onnx: PathBuf,
    pub json: PathBuf,
    pub sample_rate: u32,
}

#[derive(Deserialize)]
struct AudioConfig {
    sample_rate: u32,
}

#[derive(Deserialize)]
struct VoiceConfigJson {
    audio: AudioConfig,
}

/// The `piper` subdirectory of the shared model cache.
pub fn default_cache_dir() -> Result<PathBuf, DownloadError> {
    download::default_cache_dir("piper")
}

/// Ensure both voice files exist locally, downloading whichever is
/// missing.
pub async fn ensure_voice(hf_id: &str, cache_dir: &Path) -> Result<VoiceFiles, PiperError> {
    let onnx_id = parse_id(hf_id)?;
    let json_id = parse_id(&format!("{onnx_id}.json"))?;
    let onnx = cached_path(cache_dir, &onnx_id);
    let json = cached_path(cache_dir, &json_id);

    ensure_file(&onnx_id, &onnx).await?;
    ensure_file(&json_id, &json).await?;

    let sample_rate = read_sample_rate(&json).await?;
    Ok(VoiceFiles {
        onnx,
        json,
        sample_rate,
    })
}

/// Rejects a body that is not a JSON object up front, since HuggingFace serves
/// a 200-OK HTML "not found" page for missing files.
async fn read_sample_rate(json: &Path) -> Result<u32, PiperError> {
    let body = tokio::fs::read_to_string(json)
        .await
        .map_err(|source| PiperError::Io {
            path: json.to_path_buf(),
            source,
        })?;

    let trimmed = body.trim_start();
    if !trimmed.starts_with('{') {
        let prefix: String = trimmed.chars().take(40).collect();
        return Err(PiperError::JsonShape {
            path: json.to_path_buf(),
            prefix,
        });
    }

    let config: VoiceConfigJson =
        serde_json::from_str(&body).map_err(|source| PiperError::JsonParse {
            path: json.to_path_buf(),
            source,
        })?;
    Ok(config.audio.sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

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
