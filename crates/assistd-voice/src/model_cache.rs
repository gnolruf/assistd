//! On-disk cache for Whisper GGML model files. HuggingFace identifier
//! in, local `PathBuf` out; downloads atomically on first use.

use std::path::{Path, PathBuf};

use tokio::io::AsyncWriteExt;

use crate::transcribe::TranscriptionError;

/// Parse `"<owner>/<repo>:<file>"` into `(repo, file)` where `repo` is
/// `"<owner>/<repo>"`.
pub fn parse_hf_id(id: &str) -> Result<(String, String), TranscriptionError> {
    let err = |reason: &str| TranscriptionError::ModelParse {
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

/// `$XDG_CACHE_HOME/assistd/whisper/` or `$HOME/.cache/assistd/whisper/`.
pub fn default_cache_dir() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    base.join("assistd").join("whisper")
}

fn sanitize_repo(repo: &str) -> String {
    repo.replace('/', "__")
}

fn cached_path(cache_dir: &Path, repo: &str, file: &str) -> PathBuf {
    cache_dir.join(sanitize_repo(repo)).join(file)
}

/// Ensure the file for `hf_id` exists locally, downloading it on first
/// use. Returns the resolved path. Safe against partial writes: the
/// download lands in `<file>.part` and is atomically renamed on success.
pub async fn ensure_model(hf_id: &str, cache_dir: &Path) -> Result<PathBuf, TranscriptionError> {
    let (repo, file) = parse_hf_id(hf_id)?;
    let dest = cached_path(cache_dir, &repo, &file);
    if dest.exists() {
        tracing::debug!(
            target: "assistd::voice::download",
            path = %dest.display(),
            "model already cached"
        );
        return Ok(dest);
    }

    let parent = dest.parent().expect("cache path always has a parent");
    tokio::fs::create_dir_all(parent)
        .await
        .map_err(|source| TranscriptionError::ModelIo {
            path: parent.to_path_buf(),
            source,
        })?;

    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    tracing::info!(
        target: "assistd::voice::download",
        %url,
        path = %dest.display(),
        "downloading whisper model"
    );

    let response =
        reqwest::get(&url)
            .await
            .map_err(|source| TranscriptionError::ModelDownload {
                url: url.clone(),
                source,
            })?;
    let status = response.status();
    if !status.is_success() {
        return Err(TranscriptionError::ModelHttp {
            url,
            status: status.as_u16(),
        });
    }
    let total = response.content_length();

    let part = dest.with_extension(format!(
        "{}.part",
        dest.extension().and_then(|e| e.to_str()).unwrap_or("bin")
    ));
    let mut out =
        tokio::fs::File::create(&part)
            .await
            .map_err(|source| TranscriptionError::ModelIo {
                path: part.clone(),
                source,
            })?;

    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut next_tick: u64 = 0;
    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|source| TranscriptionError::ModelDownload {
            url: url.clone(),
            source,
        })?;
        out.write_all(&chunk)
            .await
            .map_err(|source| TranscriptionError::ModelIo {
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
                target: "assistd::voice::download",
                pct,
                downloaded_mib = downloaded / (1024 * 1024),
                total_mib = total / (1024 * 1024),
                "download progress"
            );
            next_tick = downloaded + total / 20;
        }
    }
    out.flush()
        .await
        .map_err(|source| TranscriptionError::ModelIo {
            path: part.clone(),
            source,
        })?;
    drop(out);
    tokio::fs::rename(&part, &dest)
        .await
        .map_err(|source| TranscriptionError::ModelIo {
            path: dest.clone(),
            source,
        })?;
    tracing::info!(
        target: "assistd::voice::download",
        path = %dest.display(),
        "download complete"
    );
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_id() {
        let (repo, file) = parse_hf_id("ggerganov/whisper.cpp:ggml-tiny.en-q5_1.bin").unwrap();
        assert_eq!(repo, "ggerganov/whisper.cpp");
        assert_eq!(file, "ggml-tiny.en-q5_1.bin");
    }

    #[test]
    fn rejects_missing_colon() {
        assert!(parse_hf_id("ggerganov/whisper.cpp").is_err());
    }

    #[test]
    fn rejects_missing_slash() {
        assert!(parse_hf_id("whisper:file.bin").is_err());
    }

    #[test]
    fn rejects_empty_file() {
        assert!(parse_hf_id("owner/repo:").is_err());
    }

    #[test]
    fn rejects_empty_owner() {
        assert!(parse_hf_id("/repo:file.bin").is_err());
    }

    #[test]
    fn cached_path_sanitizes_slash() {
        let p = cached_path(
            Path::new("/cache"),
            "ggml-org/whisper-vad",
            "ggml-silero-v6.2.0.bin",
        );
        assert_eq!(
            p,
            Path::new("/cache/ggml-org__whisper-vad/ggml-silero-v6.2.0.bin")
        );
    }
}
