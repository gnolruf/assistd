//! Hugging Face file downloads into an on-disk cache.

use std::ffi::OsString;
use std::fs::{File, OpenOptions, TryLockError};
use std::io::{self, SeekFrom};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use futures_util::StreamExt;
use reqwest::header::RANGE;
use reqwest::{Client, ClientBuilder, Response, StatusCode, redirect};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

use super::{HfFileId, InvalidHfId};
use crate::http::remote_client_builder;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
/// Longest wait for the next chunk before a download counts as stalled.
const READ_TIMEOUT: Duration = Duration::from_secs(60);
/// Largest file a download may write; well above any model assistd fetches.
const MAX_DOWNLOAD_BYTES: u64 = 8 * 1024 * 1024 * 1024;
/// Header on a `resolve` redirect carrying the LFS file's SHA-256.
const LINKED_ETAG: &str = "x-linked-etag";
const HASH_BUFFER_BYTES: usize = 1024 * 1024;

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

/// Download `id` to `dest` unless it exists, via `<dest>.part` held under an exclusive
/// advisory lock. A part left by an interrupted run is resumed when Hugging Face reports
/// the file's SHA-256 and truncated otherwise; a part failing the checksum is deleted.
/// Fails when connecting or a read stalls past its timeout or the file exceeds 8 GiB.
pub async fn ensure_file(id: &HfFileId, dest: &Path) -> Result<(), DownloadError> {
    if is_cached(dest).await? {
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
    let part_path = part_path_for(dest);
    let mut part = lock_part_file(&part_path).await?;
    if is_cached(dest).await? {
        return remove_part(&part_path).await;
    }

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
    download_to_part(&url, &mut part, &part_path).await?;
    persist_durably(&part_path, dest).await?;
    tracing::info!(
        target: "assistd::hf::download",
        path = %dest.display(),
        "download complete"
    );
    Ok(())
}

/// Bytes already in a part file and the running hash over them.
#[derive(Default)]
struct Progress {
    offset: u64,
    hasher: Sha256,
}

impl Progress {
    fn hex_digest(&self) -> String {
        self.hasher
            .clone()
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
}

/// Fill `part` with the body of `url` and verify it against Hugging Face's SHA-256,
/// deleting the part on a mismatch so the next attempt starts over.
async fn download_to_part(
    url: &str,
    part: &mut tokio::fs::File,
    part_path: &Path,
) -> Result<(), DownloadError> {
    let request_error = |source| DownloadError::Request {
        url: url.to_string(),
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
    let expected_sha256 = fetch_expected_sha256(&probe_client, url).await?;
    let progress = match &expected_sha256 {
        Some(_) => hash_existing_part(part, part_path).await?,
        None => Progress::default(),
    };
    let actual_sha256 = if expected_sha256.as_ref() == Some(&progress.hex_digest()) {
        progress.hex_digest()
    } else {
        let (response, progress) = fetch_resuming(&client, url, progress).await?;
        stream_to_file(response, url, part, part_path, progress).await?
    };
    match expected_sha256 {
        Some(expected) if expected != actual_sha256 => {
            remove_part(part_path).await?;
            Err(DownloadError::ChecksumMismatch {
                url: url.to_string(),
                expected,
                actual: actual_sha256,
            })
        }
        Some(_) => Ok(()),
        None => {
            tracing::debug!(
                target: "assistd::hf::download",
                %url,
                "no LFS SHA-256 reported; skipping integrity check"
            );
            Ok(())
        }
    }
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

async fn is_cached(dest: &Path) -> Result<bool, DownloadError> {
    tokio::fs::try_exists(dest)
        .await
        .map_err(|source| io_error(dest, source))
}

fn part_path_for(dest: &Path) -> PathBuf {
    let mut name = OsString::from(dest.as_os_str());
    name.push(".part");
    PathBuf::from(name)
}

/// Open `path` and take an exclusive advisory lock on it, waiting for any other
/// holder; the lock lasts until the returned file is dropped.
async fn lock_part_file(path: &Path) -> Result<tokio::fs::File, DownloadError> {
    let owned = path.to_path_buf();
    tokio::task::spawn_blocking(move || open_locked(&owned))
        .await
        .map_err(|join| io_error(path, io::Error::other(join)))?
        .map(tokio::fs::File::from_std)
        .map_err(|source| io_error(path, source))
}

/// Retries until the locked file is still the one linked at `path`, since a previous
/// holder may have renamed or deleted it while this one waited.
fn open_locked(path: &Path) -> io::Result<File> {
    loop {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        match file.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                tracing::info!(
                    target: "assistd::hf::download",
                    path = %path.display(),
                    "waiting for another download of this file"
                );
                file.lock()?;
            }
            Err(TryLockError::Error(source)) => return Err(source),
        }
        if is_linked_at(&file, path)? {
            return Ok(file);
        }
    }
}

fn is_linked_at(file: &File, path: &Path) -> io::Result<bool> {
    let opened = file.metadata()?;
    match std::fs::metadata(path) {
        Ok(linked) => Ok(linked.dev() == opened.dev() && linked.ino() == opened.ino()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(err),
    }
}

async fn remove_part(part_path: &Path) -> Result<(), DownloadError> {
    tokio::fs::remove_file(part_path)
        .await
        .map_err(|source| io_error(part_path, source))
}

/// Hash every byte already in `part`, leaving its cursor at the end.
async fn hash_existing_part(
    part: &mut tokio::fs::File,
    part_path: &Path,
) -> Result<Progress, DownloadError> {
    let mut progress = Progress::default();
    let mut buffer = vec![0u8; HASH_BUFFER_BYTES];
    loop {
        let read = part
            .read(&mut buffer)
            .await
            .map_err(|source| io_error(part_path, source))?;
        if read == 0 {
            return Ok(progress);
        }
        progress.hasher.update(&buffer[..read]);
        progress.offset += read as u64;
    }
}

/// Request `url` from `progress.offset`, falling back to the whole file when the
/// server ignores or rejects the range.
async fn fetch_resuming(
    client: &Client,
    url: &str,
    progress: Progress,
) -> Result<(Response, Progress), DownloadError> {
    if progress.offset == 0 {
        return Ok((fetch(client, url).await?, progress));
    }
    let response = send_get(client, url, progress.offset).await?;
    match response.status() {
        StatusCode::PARTIAL_CONTENT => {
            tracing::info!(
                target: "assistd::hf::download",
                %url,
                resumed_mib = progress.offset / (1024 * 1024),
                "resuming download"
            );
            Ok((response, progress))
        }
        StatusCode::RANGE_NOT_SATISFIABLE => Ok((fetch(client, url).await?, Progress::default())),
        status => Ok((require_success(response, url, status)?, Progress::default())),
    }
}

/// Rename `part_path` onto `dest` and fsync the directory so the rename
/// survives a crash.
async fn persist_durably(part_path: &Path, dest: &Path) -> Result<(), DownloadError> {
    tokio::fs::rename(part_path, dest)
        .await
        .map_err(|source| io_error(dest, source))?;
    let parent = dest
        .parent()
        .expect("cache path always has a parent")
        .to_path_buf();
    tokio::task::spawn_blocking(move || {
        File::open(&parent)
            .and_then(|dir| dir.sync_all())
            .map_err(|source| io_error(&parent, source))
    })
    .await
    .map_err(|join| io_error(dest, io::Error::other(join)))?
}

async fn fetch(client: &Client, url: &str) -> Result<Response, DownloadError> {
    let response = send_get(client, url, 0).await?;
    let status = response.status();
    require_success(response, url, status)
}

async fn send_get(client: &Client, url: &str, offset: u64) -> Result<Response, DownloadError> {
    let request = client.get(url);
    let request = match offset {
        0 => request,
        offset => request.header(RANGE, format!("bytes={offset}-")),
    };
    request
        .send()
        .await
        .map_err(|source| DownloadError::Request {
            url: url.to_string(),
            source,
        })
}

fn require_success(
    response: Response,
    url: &str,
    status: StatusCode,
) -> Result<Response, DownloadError> {
    if !status.is_success() {
        return Err(DownloadError::Http {
            url: url.to_string(),
            status: status.as_u16(),
        });
    }
    Ok(response)
}

/// Write the response body to `out` from `progress.offset`, truncating anything past it,
/// and sync it to disk, returning the whole file's lowercase hex SHA-256. Logs progress
/// every 5% when the length is known and stops once the file passes [`MAX_DOWNLOAD_BYTES`].
async fn stream_to_file(
    response: Response,
    url: &str,
    out: &mut tokio::fs::File,
    part_path: &Path,
    progress: Progress,
) -> Result<String, DownloadError> {
    let too_large = || DownloadError::TooLarge {
        url: url.to_string(),
        limit_bytes: MAX_DOWNLOAD_BYTES,
    };
    let Progress { offset, mut hasher } = progress;
    let total = response
        .content_length()
        .map(|remaining| remaining.saturating_add(offset));
    if total.is_some_and(|total| total > MAX_DOWNLOAD_BYTES) {
        return Err(too_large());
    }
    let io_err = |source| io_error(part_path, source);
    out.set_len(offset).await.map_err(io_err)?;
    out.seek(SeekFrom::Start(offset)).await.map_err(io_err)?;

    let mut stream = response.bytes_stream();
    let mut downloaded = offset;
    let mut next_progress_log: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|source| DownloadError::Request {
            url: url.to_string(),
            source,
        })?;
        out.write_all(&chunk).await.map_err(io_err)?;
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
    out.flush().await.map_err(io_err)?;
    out.sync_all().await.map_err(io_err)?;
    Ok(Progress {
        offset: downloaded,
        hasher,
    }
    .hex_digest())
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
            let _ = conn.read(&mut request).await;
            conn.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\npartial")
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_secs(30)).await;
        });

        let part_path = tempfile::NamedTempFile::new().unwrap().into_temp_path();
        let mut part = tokio::fs::File::create(&part_path).await.unwrap();
        let client = download_builder(Duration::from_millis(200))
            .build()
            .unwrap();
        let download = async {
            let response = fetch(&client, &url).await?;
            stream_to_file(response, &url, &mut part, &part_path, Progress::default()).await
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

    #[tokio::test]
    async fn resumed_download_appends_range_and_hashes_whole_file() {
        let server = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/model.bin", server.local_addr().unwrap());
        let range_server = tokio::spawn(async move {
            let (mut conn, _) = server.accept().await.unwrap();
            let mut request = [0u8; 1024];
            let read = conn.read(&mut request).await.unwrap();
            let request = String::from_utf8_lossy(&request[..read]).to_ascii_lowercase();
            let response: &[u8] = if request.contains("range: bytes=6-") {
                b"HTTP/1.1 206 Partial Content\r\nContent-Length: 5\r\n\r\nworld"
            } else {
                b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n"
            };
            conn.write_all(response).await.unwrap();
        });

        let dir = tempfile::tempdir().unwrap();
        let part_path = dir.path().join("model.bin.part");
        std::fs::write(&part_path, b"hello ").unwrap();
        let mut part = lock_part_file(&part_path).await.unwrap();
        let progress = hash_existing_part(&mut part, &part_path).await.unwrap();
        let client = download_builder(READ_TIMEOUT).build().unwrap();
        let (response, progress) = fetch_resuming(&client, &url, progress).await.unwrap();
        let digest = stream_to_file(response, &url, &mut part, &part_path, progress)
            .await
            .unwrap();

        assert_eq!(std::fs::read(&part_path).unwrap(), b"hello world");
        assert_eq!(
            digest,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
        range_server.await.unwrap();
    }
}
