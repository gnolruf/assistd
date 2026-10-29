//! The private directory llama-server saves and restores slot state in, and
//! the guard that keeps the main conversation's slot across sub-agent runs.

use std::fs::{self, DirBuilder};
use std::future::Future;
use std::io::{self, ErrorKind};
use std::net::SocketAddr;
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use assistd_utils::child_server::ApiKey;
use assistd_utils::http::{local_client_builder, read_capped};
use assistd_utils::text::human_size;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use super::error::LlamaServerError;
use crate::LlmHealthProbe;

const OWNER_ONLY: u32 = 0o700;
/// The only slot with `-np 1`.
const MAIN_SLOT_PATH: &str = "/slots/0";
/// Fixed so at most one main-slot file exists at a time.
const MAIN_SLOT_FILE: &str = "main.bin";
const SLOT_ACTION_TIMEOUT: Duration = Duration::from_secs(60);
const SLOT_RESPONSE_MAX: usize = 64 * 1024;

/// A mode-0700 directory passed as `--slot-save-path`, emptied before every
/// server spawn since a slot file is only valid for the model and settings
/// that wrote it.
#[derive(Debug, Clone)]
pub struct SlotDir {
    path: Arc<Path>,
}

impl SlotDir {
    /// Use `path` as the slot directory; nothing is touched until [`Self::reset`].
    pub fn new(path: PathBuf) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Replace whatever is at the path with an empty owner-only directory.
    pub fn reset(&self) -> io::Result<()> {
        if let Err(e) = fs::remove_dir_all(&self.path)
            && e.kind() != ErrorKind::NotFound
        {
            return Err(e);
        }
        DirBuilder::new()
            .recursive(true)
            .mode(OWNER_ONLY)
            .create(&self.path)
    }
}

/// Saves the chat server's main slot before a group of sub-agent requests
/// and restores it after, so the next main turn skips re-processing its
/// context. Every failure is logged and the group runs regardless.
#[derive(Debug)]
pub struct MainSlot {
    client: reqwest::Client,
    url: String,
    model: String,
    slot_dir: SlotDir,
    health: Option<Arc<dyn LlmHealthProbe>>,
}

impl MainSlot {
    /// A guard for `model`'s slot on the router at `addr`, whose files land
    /// in `slot_dir`. With `health`, a restore is skipped once the serving
    /// child has changed since the save.
    pub fn new(
        addr: SocketAddr,
        api_key: Option<&ApiKey>,
        model: String,
        slot_dir: SlotDir,
        health: Option<Arc<dyn LlmHealthProbe>>,
    ) -> Result<Self, LlamaServerError> {
        let client = local_client_builder()
            .default_headers(
                api_key
                    .map(ApiKey::authorization_headers)
                    .unwrap_or_default(),
            )
            .timeout(SLOT_ACTION_TIMEOUT)
            .build()?;
        Ok(Self {
            client,
            url: format!("http://{addr}{MAIN_SLOT_PATH}"),
            model,
            slot_dir,
            health,
        })
    }

    /// Run `group` between a save and a restore of the main slot. The restore
    /// runs however `group` ends, but not if this future is dropped. Hold the
    /// agent turn lock across the call so no main turn lands in between.
    pub async fn preserve_across<F: Future>(&self, group: F) -> F::Output {
        let saved = self.save().await;
        let output = group.await;
        if let Some(saved) = saved {
            self.restore(saved).await;
            self.remove_file().await;
        }
        output
    }

    async fn save(&self) -> Option<SavedSlot> {
        let Ok(pid) = self.serving_pid().await else {
            warn!(target: "assistd::slots", "llama-server is not serving; main slot not saved");
            return None;
        };
        let started = Instant::now();
        match self.post_action("save").await {
            Ok(response) => {
                debug!(
                    target: "assistd::slots",
                    tokens = response.n_saved.unwrap_or_default(),
                    size = %human_size(response.n_written.unwrap_or_default()),
                    elapsed = ?started.elapsed(),
                    "saved main slot"
                );
                Some(SavedSlot { pid })
            }
            Err(e) => {
                warn!(target: "assistd::slots", "saving the main slot failed: {e}");
                None
            }
        }
    }

    async fn restore(&self, saved: SavedSlot) {
        if self.serving_pid().await.ok() != Some(saved.pid) {
            warn!(
                target: "assistd::slots",
                "llama-server restarted since the main slot was saved; restore skipped"
            );
            return;
        }
        let started = Instant::now();
        match self.post_action("restore").await {
            Ok(response) => debug!(
                target: "assistd::slots",
                tokens = response.n_restored.unwrap_or_default(),
                size = %human_size(response.n_read.unwrap_or_default()),
                elapsed = ?started.elapsed(),
                "restored main slot"
            ),
            Err(e) => warn!(target: "assistd::slots", "restoring the main slot failed: {e}"),
        }
    }

    async fn remove_file(&self) {
        let path = self.slot_dir.path().join(MAIN_SLOT_FILE);
        if let Err(e) = tokio::fs::remove_file(&path).await
            && e.kind() != ErrorKind::NotFound
        {
            warn!(
                target: "assistd::slots",
                "removing the saved main slot {} failed: {e}",
                path.display()
            );
        }
    }

    /// The serving child's pid, `None` without a probe; `Err` when the
    /// probed child is not serving.
    async fn serving_pid(&self) -> Result<Option<u32>, LlamaServerError> {
        let Some(probe) = &self.health else {
            return Ok(None);
        };
        match probe.snapshot().await {
            Some(snapshot) if snapshot.is_serving() => Ok(snapshot.pid),
            _ => Err(LlamaServerError::NotReady),
        }
    }

    async fn post_action(&self, action: &str) -> Result<SlotActionResponse, LlamaServerError> {
        let body = SlotActionRequest {
            model: &self.model,
            filename: MAIN_SLOT_FILE,
        };
        let resp = self
            .client
            .post(&self.url)
            .query(&[("action", action)])
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            return Err(LlamaServerError::ControlHttp {
                method: "POST",
                path: MAIN_SLOT_PATH,
                status: status.as_u16(),
            });
        }
        let body = read_capped(resp, SLOT_RESPONSE_MAX).await?;
        Ok(serde_json::from_slice(&body)?)
    }
}

/// A main slot saved while the child with `pid` was serving.
struct SavedSlot {
    pid: Option<u32>,
}

/// The router picks the model child from `model`.
#[derive(Serialize)]
struct SlotActionRequest<'a> {
    model: &'a str,
    filename: &'a str,
}

#[derive(Deserialize)]
struct SlotActionResponse {
    #[serde(default)]
    n_saved: Option<u64>,
    #[serde(default)]
    n_written: Option<u64>,
    #[serde(default)]
    n_restored: Option<u64>,
    #[serde(default)]
    n_read: Option<u64>,
}
