//! HuggingFace file downloads with an on-disk cache.

use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use reqwest::Response;
use tokio::io::AsyncWriteExt;

/// Errors from resolving or downloading a HuggingFace file.
#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error("invalid HuggingFace identifier {id:?}: {reason} (expected '<owner>/<repo>:<file>')")]
    InvalidId { id: String, reason: String },

    #[error("failed to download {url}: {source}")]
    Request {
        url: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("download of {url} returned HTTP {status}")]
    Http { url: String, status: u16 },

    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

/// Parse `"<owner>/<repo>:<file>"` into `(repo, file)`.
pub fn parse_hf_id(id: &str) -> Result<(String, String), DownloadError> {
    let err = |reason: &str| DownloadError::InvalidId {
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

/// `$XDG_CACHE_HOME/assistd/<subdir>/`, falling back to
/// `$HOME/.cache/assistd/<subdir>/`, then `/tmp/assistd/<subdir>/`.
pub fn default_cache_dir(subdir: &str) -> PathBuf {
    default_cache_dir_from(
        std::env::var_os("XDG_CACHE_HOME"),
        std::env::var_os("HOME"),
        subdir,
    )
}

fn default_cache_dir_from(
    xdg_cache_home: Option<OsString>,
    home: Option<OsString>,
    subdir: &str,
) -> PathBuf {
    let base = xdg_cache_home
        .map(PathBuf::from)
        .or_else(|| home.map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    base.join("assistd").join(subdir)
}

/// Where `file` from `repo` lives under `cache_dir`.
pub fn cached_path(cache_dir: &Path, repo: &str, file: &str) -> PathBuf {
    cache_dir.join(repo.replace('/', "__")).join(file)
}

/// Path to the cached file for `hf_id`, downloading it first if missing.
pub async fn ensure_cached(hf_id: &str, cache_dir: &Path) -> Result<PathBuf, DownloadError> {
    let (repo, file) = parse_hf_id(hf_id)?;
    let dest = cached_path(cache_dir, &repo, &file);
    ensure_file(&repo, &file, &dest).await?;
    Ok(dest)
}

/// Download `file` from `repo` to `dest` unless it already exists. The
/// download lands in a `.part` sibling and is renamed on success, so a
/// crash never leaves a partial file at `dest`.
pub async fn ensure_file(repo: &str, file: &str, dest: &Path) -> Result<(), DownloadError> {
    if dest.exists() {
        tracing::debug!(
            target: "assistd::voice::download",
            path = %dest.display(),
            "already cached"
        );
        return Ok(());
    }

    let parent = dest.parent().expect("cache path always has a parent");
    tokio::fs::create_dir_all(parent)
        .await
        .map_err(|source| io_error(parent, source))?;

    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    tracing::info!(
        target: "assistd::voice::download",
        %url,
        path = %dest.display(),
        "downloading"
    );

    let response = fetch(&url).await?;
    let part = part_path(dest);
    stream_to_file(response, &url, &part).await?;
    tokio::fs::rename(&part, dest)
        .await
        .map_err(|source| io_error(dest, source))?;
    tracing::info!(
        target: "assistd::voice::download",
        path = %dest.display(),
        "download complete"
    );
    Ok(())
}

async fn fetch(url: &str) -> Result<Response, DownloadError> {
    let response = reqwest::get(url)
        .await
        .map_err(|source| DownloadError::Request {
            url: url.to_string(),
            source,
        })?;
    let status = response.status();
    if !status.is_success() {
        return Err(DownloadError::Http {
            url: url.to_string(),
            status: status.as_u16(),
        });
    }
    Ok(response)
}

fn part_path(dest: &Path) -> PathBuf {
    dest.with_extension(format!(
        "{}.part",
        dest.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("bin")
    ))
}

/// Write the response body to `part`, logging progress every 5% when the length is known.
async fn stream_to_file(response: Response, url: &str, part: &Path) -> Result<(), DownloadError> {
    let total = response.content_length();
    let mut out = tokio::fs::File::create(part)
        .await
        .map_err(|source| io_error(part, source))?;

    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut next_progress_log: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|source| DownloadError::Request {
            url: url.to_string(),
            source,
        })?;
        out.write_all(&chunk)
            .await
            .map_err(|source| io_error(part, source))?;
        downloaded = downloaded.saturating_add(chunk.len() as u64);
        if let Some(total) = total
            && total > 0
            && downloaded >= next_progress_log
        {
            tracing::info!(
                target: "assistd::voice::download",
                pct = downloaded * 100 / total,
                downloaded_mib = downloaded / (1024 * 1024),
                total_mib = total / (1024 * 1024),
                "download progress"
            );
            next_progress_log = downloaded + total / 20;
        }
    }
    out.flush().await.map_err(|source| io_error(part, source))
}

fn io_error(path: &Path, source: io::Error) -> DownloadError {
    DownloadError::Io {
        path: path.to_path_buf(),
        source,
    }
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
    fn rejects_malformed_ids() {
        for id in [
            "ggerganov/whisper.cpp",
            "whisper:file.bin",
            "owner/repo:",
            "owner/repo:a:b",
            "/repo:file.bin",
            "owner/:file.bin",
        ] {
            let err = parse_hf_id(id).expect_err(id);
            assert!(
                matches!(&err, DownloadError::InvalidId { id: got, .. } if got == id),
                "{id}: {err:?}"
            );
        }
    }

    #[test]
    fn cached_path_sanitizes_slash() {
        let path = cached_path(
            Path::new("/cache"),
            "ggml-org/whisper-vad",
            "ggml-silero-v6.2.0.bin",
        );
        assert_eq!(
            path,
            Path::new("/cache/ggml-org__whisper-vad/ggml-silero-v6.2.0.bin")
        );
    }

    #[test]
    fn default_cache_dir_prefers_xdg_then_home_then_tmp() {
        let xdg = Some(OsString::from("/tmp/xdg-test"));
        let home = Some(OsString::from("/home/alice"));
        assert_eq!(
            default_cache_dir_from(xdg, home.clone(), "piper"),
            Path::new("/tmp/xdg-test/assistd/piper")
        );
        assert_eq!(
            default_cache_dir_from(None, home, "piper"),
            Path::new("/home/alice/.cache/assistd/piper")
        );
        assert_eq!(
            default_cache_dir_from(None, None, "whisper"),
            Path::new("/tmp/assistd/whisper")
        );
    }
}
