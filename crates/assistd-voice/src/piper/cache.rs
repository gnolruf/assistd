//! On-disk cache for Piper voices. Each voice is two files: a `.onnx`
//! ONNX model and a matching `.onnx.json` config that carries the
//! sample rate among other things. Both must sit side-by-side in the
//! cache directory because piper looks for `<onnx>.json` next to the
//! `--model` path.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use tokio::io::AsyncWriteExt;

use crate::piper::error::PiperError;

/// `<owner>/<repo>:<file>` parser. Identical contract to the whisper
/// cache version — duplicated locally to keep error types crate-local.
pub fn parse_hf_id(id: &str) -> Result<(String, String), PiperError> {
    let err = |reason: &str| PiperError::VoiceParse {
        id: id.to_string(),
        reason: reason.to_string(),
    };
    let (repo, file) = id.split_once(':').ok_or_else(|| err("missing ':'"))?;
    if file.is_empty() || file.contains(':') {
        return Err(err("file must be non-empty and must not contain ':'"));
    }
    let (owner, name) = repo
        .split_once('/')
        .ok_or_else(|| err("repo must be 'owner/name'"))?;
    if owner.is_empty() || name.is_empty() {
        return Err(err("owner and repo name must both be non-empty"));
    }
    Ok((repo.to_string(), file.to_string()))
}

/// `$XDG_CACHE_HOME/assistd/piper/` or `$HOME/.cache/assistd/piper/`.
pub fn default_cache_dir() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    base.join("assistd").join("piper")
}

/// Resolved on-disk paths for a voice. Sample rate is read from the
/// `.onnx.json`'s `audio.sample_rate` field; everything else piper
/// loads itself from the `.onnx`.
#[derive(Debug, Clone)]
pub struct VoiceFiles {
    pub onnx: PathBuf,
    pub json: PathBuf,
    pub sample_rate: u32,
}

/// Ensure both the `.onnx` and `.onnx.json` for `hf_id` exist locally,
/// downloading whichever is missing. Returns the resolved `VoiceFiles`.
///
/// Atomic writes: each download lands in `<file>.part` and is renamed
/// only after the body is fully flushed, so a crash mid-download leaves
/// no half-baked file masquerading as a complete one.
pub async fn ensure_voice(hf_id: &str, cache_dir: &Path) -> Result<VoiceFiles, PiperError> {
    let (repo, file) = parse_hf_id(hf_id)?;
    let onnx = cached_path(cache_dir, &repo, &file);
    let json = onnx.with_extension("onnx.json");

    ensure_one(&repo, &file, &onnx).await?;
    ensure_one(&repo, &format!("{file}.json"), &json).await?;

    let sample_rate = read_sample_rate(&json).await?;
    Ok(VoiceFiles {
        onnx,
        json,
        sample_rate,
    })
}

async fn ensure_one(repo: &str, file: &str, dest: &Path) -> Result<(), PiperError> {
    if dest.exists() {
        tracing::debug!(
            target: "assistd::voice::piper::cache",
            path = %dest.display(),
            "voice file already cached"
        );
        return Ok(());
    }

    let parent = dest.parent().expect("cache path always has a parent");
    tokio::fs::create_dir_all(parent)
        .await
        .map_err(|source| PiperError::Io {
            path: parent.to_path_buf(),
            source,
        })?;

    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    tracing::info!(
        target: "assistd::voice::piper::cache",
        %url,
        path = %dest.display(),
        "downloading piper voice file"
    );

    let response = reqwest::get(&url)
        .await
        .map_err(|source| PiperError::Download {
            url: url.clone(),
            source,
        })?;
    let status = response.status();
    if !status.is_success() {
        return Err(PiperError::Http {
            url,
            status: status.as_u16(),
        });
    }
    let total = response.content_length();

    let part = dest.with_extension(format!(
        "{}.part",
        dest.extension().and_then(|e| e.to_str()).unwrap_or("bin")
    ));
    let mut out = tokio::fs::File::create(&part)
        .await
        .map_err(|source| PiperError::Io {
            path: part.clone(),
            source,
        })?;

    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut next_tick: u64 = 0;
    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|source| PiperError::Download {
            url: url.clone(),
            source,
        })?;
        out.write_all(&chunk)
            .await
            .map_err(|source| PiperError::Io {
                path: part.clone(),
                source,
            })?;
        downloaded = downloaded.saturating_add(chunk.len() as u64);
        if let Some(total) = total
            && total > 0
            && downloaded >= next_tick
        {
            let pct = downloaded * 100 / total;
            tracing::info!(
                target: "assistd::voice::piper::cache",
                pct,
                downloaded_kib = downloaded / 1024,
                total_kib = total / 1024,
                "voice download progress"
            );
            next_tick = downloaded + total / 20;
        }
    }
    out.flush().await.map_err(|source| PiperError::Io {
        path: part.clone(),
        source,
    })?;
    drop(out);
    tokio::fs::rename(&part, dest)
        .await
        .map_err(|source| PiperError::Io {
            path: dest.to_path_buf(),
            source,
        })?;
    tracing::info!(
        target: "assistd::voice::piper::cache",
        path = %dest.display(),
        "download complete"
    );
    Ok(())
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

    // Defend against HuggingFace returning a 200-OK HTML "file not
    // found" page in place of the JSON body.
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

fn sanitize_repo(repo: &str) -> String {
    repo.replace('/', "__")
}

fn cached_path(cache_dir: &Path, repo: &str, file: &str) -> PathBuf {
    cache_dir.join(sanitize_repo(repo)).join(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_id() {
        let (repo, file) =
            parse_hf_id("rhasspy/piper-voices:en/en_US/lessac/medium/en_US-lessac-medium.onnx")
                .unwrap();
        assert_eq!(repo, "rhasspy/piper-voices");
        assert_eq!(file, "en/en_US/lessac/medium/en_US-lessac-medium.onnx");
    }

    #[test]
    fn rejects_missing_colon() {
        assert!(parse_hf_id("rhasspy/piper-voices").is_err());
    }

    #[test]
    fn rejects_missing_slash() {
        assert!(parse_hf_id("piper:file.onnx").is_err());
    }

    #[test]
    fn rejects_empty_file() {
        assert!(parse_hf_id("owner/repo:").is_err());
    }

    #[test]
    fn rejects_empty_owner() {
        assert!(parse_hf_id("/repo:file.onnx").is_err());
    }

    #[test]
    fn cached_path_sanitizes_slash() {
        let p = cached_path(
            Path::new("/cache"),
            "rhasspy/piper-voices",
            "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        );
        assert_eq!(
            p,
            Path::new(
                "/cache/rhasspy__piper-voices/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
            )
        );
    }

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

    #[test]
    fn default_cache_dir_under_xdg() {
        unsafe {
            std::env::set_var("XDG_CACHE_HOME", "/tmp/xdg-test-piper");
        }
        let p = default_cache_dir();
        assert_eq!(p, Path::new("/tmp/xdg-test-piper/assistd/piper"));
        unsafe {
            std::env::remove_var("XDG_CACHE_HOME");
        }
    }
}
