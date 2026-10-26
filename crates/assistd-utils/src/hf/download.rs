//! Hugging Face file downloads into an on-disk cache.

use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use futures_util::StreamExt;
use reqwest::{Client, ClientBuilder, Response, redirect};
use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;
use tokio::io::AsyncWriteExt;

use super::{HfFileId, InvalidHfId};
use crate::http::remote_client_builder;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
/// Longest wait for the next chunk before a download counts as stalled.
const READ_TIMEOUT: Duration = Duration::from_secs(60);
/// Largest file a download may write; well above any model assistd fetches.
const MAX_DOWNLOAD_BYTES: u64 = 8 * 1024 * 1024 * 1024;
/// Header on a `resolve` redirect carrying the LFS file's SHA-256.
const LINKED_ETAG: &str = "x-linked-etag";

/// Errors from resolving or downloading a HuggingFace file.
#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error(
        "invalid HuggingFace identifier {id:?}: {reason} \
         (expected '<owner>/<repo>[@<revision>]:<file>')"
    )]
    InvalidId { id: String, reason: InvalidHfId },

    #[error("failed to download {url}: {source}")]
    Request {
        url: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("download of {url} returned HTTP {status}")]
    Http { url: String, status: u16 },

    #[error("download of {url} exceeds the {limit_bytes}-byte size limit")]
    TooLarge { url: String, limit_bytes: u64 },

    #[error("download of {url} has SHA-256 {actual}, expected {expected}")]
    ChecksumMismatch {
        url: String,
        expected: String,
        actual: String,
    },

    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("no model cache directory: neither XDG_CACHE_HOME nor a home directory is available")]
    NoCacheDir,
}

/// Parse `"<owner>/<repo>[@<revision>]:<file>"`, rejecting any path that
/// could escape the cache directory.
pub fn parse_id(id: &str) -> Result<HfFileId, DownloadError> {
    HfFileId::parse(id).map_err(|reason| DownloadError::InvalidId {
        id: id.to_string(),
        reason,
    })
}

/// `$XDG_CACHE_HOME/assistd/<subdir>/`, falling back to
/// `~/.cache/assistd/<subdir>/`. Errors when no home directory can be found.
pub fn default_cache_dir(subdir: &str) -> Result<PathBuf, DownloadError> {
    let base = dirs::cache_dir().ok_or(DownloadError::NoCacheDir)?;
    Ok(base.join("assistd").join(subdir))
}

/// Where `id` lives under `cache_dir`; pinned revisions get their own directory.
pub fn cached_path(cache_dir: &Path, id: &HfFileId) -> PathBuf {
    let repo_dir = id.repo().replace('/', "__");
    let repo_dir = match id.revision() {
        Some(revision) => format!("{repo_dir}@{revision}"),
        None => repo_dir,
    };
    cache_dir.join(repo_dir).join(id.file())
}

/// Path to the cached file for `hf_id`, downloading it first if missing.
pub async fn ensure_cached(hf_id: &str, cache_dir: &Path) -> Result<PathBuf, DownloadError> {
    ensure_cached_id(&parse_id(hf_id)?, cache_dir).await
}

/// Path to the cached file for `id`, downloading it first if missing.
pub async fn ensure_cached_id(id: &HfFileId, cache_dir: &Path) -> Result<PathBuf, DownloadError> {
    let dest = cached_path(cache_dir, id);
    ensure_file(id, &dest).await?;
    Ok(dest)
}

/// Download `id` to `dest` unless it already exists. The download lands in
/// a `.part` temp file beside `dest`, deleted unless the download completes
/// (including when this future is dropped), and is synced to disk before the
/// rename. Fails when connecting or any read stalls past its timeout, the
/// file exceeds 8 GiB, or it does not match the SHA-256 Hugging Face reports.
pub async fn ensure_file(id: &HfFileId, dest: &Path) -> Result<(), DownloadError> {
    if tokio::fs::try_exists(dest)
        .await
        .map_err(|source| io_error(dest, source))?
    {
        tracing::debug!(
            target: "assistd::hf::download",
            path = %dest.display(),
            "already cached"
        );
        return Ok(());
    }

    let parent = dest.parent().expect("cache path always has a parent");
    tokio::fs::create_dir_all(parent)
        .await
        .map_err(|source| io_error(parent, source))?;

    let url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        id.repo(),
        id.revision().unwrap_or("main"),
        id.file()
    );
    tracing::info!(
        target: "assistd::hf::download",
        %url,
        path = %dest.display(),
        "downloading"
    );

    let request_error = |source| DownloadError::Request {
        url: url.clone(),
        source,
    };
    let client = download_builder(READ_TIMEOUT)
        .https_only(true)
        .build()
        .map_err(request_error)?;
    let probe_client = download_builder(READ_TIMEOUT)
        .https_only(true)
        .redirect(redirect::Policy::none())
        .build()
        .map_err(request_error)?;
    let expected_sha256 = fetch_expected_sha256(&probe_client, &url).await?;
    let part = create_part_file(parent).await?;
    let response = fetch(&client, &url).await?;
    let actual_sha256 = stream_to_file(response, &url, &part).await?;
    match expected_sha256 {
        Some(expected) if expected != actual_sha256 => {
            return Err(DownloadError::ChecksumMismatch {
                url,
                expected,
                actual: actual_sha256,
            });
        }
        Some(_) => {}
        None => tracing::debug!(
            target: "assistd::hf::download",
            %url,
            "no LFS SHA-256 reported; skipping integrity check"
        ),
    }
    persist_durably(part, dest).await?;
    tracing::info!(
        target: "assistd::hf::download",
        path = %dest.display(),
        "download complete"
    );
    Ok(())
}

fn download_builder(read_timeout: Duration) -> ClientBuilder {
    remote_client_builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(read_timeout)
}

/// The lowercase hex SHA-256 from the `x-linked-etag` header on the
/// unfollowed `HEAD` redirect for `url`; `None` for files outside LFS.
async fn fetch_expected_sha256(
    client: &Client,
    url: &str,
) -> Result<Option<String>, DownloadError> {
    let response = client
        .head(url)
        .send()
        .await
        .map_err(|source| DownloadError::Request {
            url: url.to_string(),
            source,
        })?;
    Ok(response
        .headers()
        .get(LINKED_ETAG)
        .and_then(|value| value.to_str().ok())
        .and_then(parse_sha256_etag))
}

fn parse_sha256_etag(etag: &str) -> Option<String> {
    let digest = etag.trim_start_matches("W/").trim_matches('"');
    (digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .then(|| digest.to_ascii_lowercase())
}

async fn create_part_file(parent: &Path) -> Result<NamedTempFile, DownloadError> {
    let dir = parent.to_path_buf();
    tokio::task::spawn_blocking(move || {
        tempfile::Builder::new()
            .suffix(".part")
            .tempfile_in(&dir)
            .map_err(|source| io_error(&dir, source))
    })
    .await
    .map_err(|join| io_error(parent, io::Error::other(join)))?
}

/// Rename `part` onto `dest` and fsync the directory so the rename survives
/// a crash.
async fn persist_durably(part: NamedTempFile, dest: &Path) -> Result<(), DownloadError> {
    let dest_path = dest.to_path_buf();
    tokio::task::spawn_blocking(move || {
        part.persist(&dest_path)
            .map_err(|err| io_error(&dest_path, err.error))?;
        let parent = dest_path.parent().expect("cache path always has a parent");
        std::fs::File::open(parent)
            .and_then(|dir| dir.sync_all())
            .map_err(|source| io_error(parent, source))
    })
    .await
    .map_err(|join| io_error(dest, io::Error::other(join)))?
}

async fn fetch(client: &Client, url: &str) -> Result<Response, DownloadError> {
    let response = client
        .get(url)
        .send()
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

/// Write the response body to `part` and sync it to disk, returning its
/// lowercase hex SHA-256. Logs progress every 5% when the length is known
/// and stops once the body passes [`MAX_DOWNLOAD_BYTES`].
async fn stream_to_file(
    response: Response,
    url: &str,
    part: &NamedTempFile,
) -> Result<String, DownloadError> {
    let too_large = || DownloadError::TooLarge {
        url: url.to_string(),
        limit_bytes: MAX_DOWNLOAD_BYTES,
    };
    let total = response.content_length();
    if total.is_some_and(|total| total > MAX_DOWNLOAD_BYTES) {
        return Err(too_large());
    }
    let part_path = part.path();
    let mut out = part
        .as_file()
        .try_clone()
        .map(tokio::fs::File::from_std)
        .map_err(|source| io_error(part_path, source))?;

    let mut stream = response.bytes_stream();
    let mut hasher = Sha256::new();
    let mut downloaded: u64 = 0;
    let mut next_progress_log: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|source| DownloadError::Request {
            url: url.to_string(),
            source,
        })?;
        out.write_all(&chunk)
            .await
            .map_err(|source| io_error(part_path, source))?;
        hasher.update(&chunk);
        downloaded = downloaded.saturating_add(chunk.len() as u64);
        if downloaded > MAX_DOWNLOAD_BYTES {
            return Err(too_large());
        }
        if let Some(total) = total
            && total > 0
            && downloaded >= next_progress_log
        {
            tracing::info!(
                target: "assistd::hf::download",
                pct = downloaded * 100 / total,
                downloaded_mib = downloaded / (1024 * 1024),
                total_mib = total / (1024 * 1024),
                "download progress"
            );
            next_progress_log = downloaded + total / 20;
        }
    }
    out.flush()
        .await
        .map_err(|source| io_error(part_path, source))?;
    out.sync_all()
        .await
        .map_err(|source| io_error(part_path, source))?;
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
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
    fn linked_etag_parses_only_quoted_sha256() {
        let digest = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3";
        assert_eq!(
            parse_sha256_etag(&format!("\"{}\"", digest.to_ascii_uppercase())).as_deref(),
            Some(digest)
        );
        assert_eq!(
            parse_sha256_etag("\"d41d8cd98f00b204e9800998ecf8427e\""),
            None
        );
        assert_eq!(
            parse_sha256_etag(&format!("\"{}\"", digest.replace('a', "z"))),
            None
        );
    }

    #[tokio::test]
    async fn stalled_download_fails_instead_of_hanging() {
        let server = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/model.bin", server.local_addr().unwrap());
        let stalled_server = tokio::spawn(async move {
            let (mut conn, _) = server.accept().await.unwrap();
            let mut request = [0u8; 1024];
            let _ = tokio::io::AsyncReadExt::read(&mut conn, &mut request).await;
            conn.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\npartial")
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_secs(30)).await;
        });

        let part = NamedTempFile::new().unwrap();
        let client = download_builder(Duration::from_millis(200))
            .build()
            .unwrap();
        let download = async {
            let response = fetch(&client, &url).await?;
            stream_to_file(response, &url, &part).await
        };
        let result = tokio::time::timeout(Duration::from_secs(10), download)
            .await
            .expect("download hung past its read timeout");
        assert!(
            matches!(result, Err(DownloadError::Request { .. })),
            "{result:?}"
        );
        stalled_server.abort();
    }
}
