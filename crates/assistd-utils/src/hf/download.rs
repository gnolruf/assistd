//! Hugging Face file downloads into an on-disk cache.

use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use futures_util::StreamExt;
use reqwest::{Client, Response};
use tempfile::NamedTempFile;
use tokio::io::AsyncWriteExt;

use super::{HfFileId, InvalidHfId};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
/// Longest wait for the next chunk before a download counts as stalled.
const READ_TIMEOUT: Duration = Duration::from_secs(60);
/// Largest file a download may write; well above any model assistd fetches.
const MAX_DOWNLOAD_BYTES: u64 = 8 * 1024 * 1024 * 1024;

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
    let id = parse_id(hf_id)?;
    let dest = cached_path(cache_dir, &id);
    ensure_file(&id, &dest).await?;
    Ok(dest)
}

/// Download `id` to `dest` unless it already exists. The download lands in
/// a `.part` temp file beside `dest`, deleted unless the download completes
/// (including when this future is dropped) and renamed on success. Fails when
/// connecting or any read stalls past its timeout, or the file exceeds 8 GiB.
pub async fn ensure_file(id: &HfFileId, dest: &Path) -> Result<(), DownloadError> {
    if dest.exists() {
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

    let client = download_client(READ_TIMEOUT).map_err(|source| DownloadError::Request {
        url: url.clone(),
        source,
    })?;
    let part = tempfile::Builder::new()
        .suffix(".part")
        .tempfile_in(parent)
        .map_err(|source| io_error(parent, source))?;
    let response = fetch(&client, &url).await?;
    stream_to_file(response, &url, &part).await?;
    part.persist(dest)
        .map_err(|err| io_error(dest, err.error))?;
    tracing::info!(
        target: "assistd::hf::download",
        path = %dest.display(),
        "download complete"
    );
    Ok(())
}

fn download_client(read_timeout: Duration) -> reqwest::Result<Client> {
    Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(read_timeout)
        .build()
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

/// Write the response body to `part`, logging progress every 5% when the
/// length is known. Stops once the body passes [`MAX_DOWNLOAD_BYTES`].
async fn stream_to_file(
    response: Response,
    url: &str,
    part: &NamedTempFile,
) -> Result<(), DownloadError> {
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
        .map_err(|source| io_error(part_path, source))
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
        let client = download_client(Duration::from_millis(200)).unwrap();
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
