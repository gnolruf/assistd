use std::io;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use rustix::process::Pid;
use tokio::sync::watch;
use tracing::{debug, warn};

use super::error::LlamaServerError;
use super::listener::is_listening_in_group;

const POLL_INTERVAL: Duration = Duration::from_millis(250);
const PROBE_TIMEOUT: Duration = Duration::from_secs(1);

/// Polls `/health` until llama-server reports ready or a timeout elapses.
/// A 200 only counts when the listener belongs to the child's process group.
pub struct HealthChecker {
    client: reqwest::Client,
    addr: SocketAddr,
    owner: Pid,
    url: String,
    poll_interval: Duration,
    probe_timeout: Duration,
    ready_timeout: Duration,
}

impl HealthChecker {
    /// Build a checker for `http://{addr}/health`, served by process group
    /// `owner`, with the given overall `ready_timeout`.
    pub fn new(
        addr: SocketAddr,
        owner: Pid,
        ready_timeout: Duration,
    ) -> Result<Self, LlamaServerError> {
        let client = reqwest::Client::builder()
            .no_proxy()
            .connect_timeout(PROBE_TIMEOUT)
            .build()?;
        Ok(Self {
            client,
            addr,
            owner,
            url: format!("http://{addr}/health"),
            poll_interval: POLL_INTERVAL,
            probe_timeout: PROBE_TIMEOUT,
            ready_timeout,
        })
    }

    /// Poll `/health` until the child's listener returns 200 OK, the
    /// deadline elapses, or shutdown is requested. Non-200 responses,
    /// foreign listeners, and transport errors keep polling.
    pub async fn wait_ready(
        &self,
        shutdown_rx: &mut watch::Receiver<bool>,
    ) -> Result<(), LlamaServerError> {
        if *shutdown_rx.borrow() {
            return Err(LlamaServerError::ShutdownDuringHealth);
        }

        let deadline = Instant::now() + self.ready_timeout;
        loop {
            if Instant::now() >= deadline {
                return Err(LlamaServerError::HealthTimeout {
                    timeout: self.ready_timeout,
                });
            }

            tokio::select! {
                biased;
                _ = shutdown_rx.changed() => {
                    return Err(LlamaServerError::ShutdownDuringHealth);
                }
                res = self.probe() => {
                    match res {
                        Ok(true) if self.served_by_child().await => return Ok(()),
                        Ok(true) => warn!(
                            target: "assistd::llama_server",
                            "health: ignoring 200 from {}, which llama-server does not own",
                            self.addr
                        ),
                        Ok(false) => debug!(target: "assistd::llama_server", "health: non-200 response"),
                        Err(e) => debug!(target: "assistd::llama_server", "health: {e}"),
                    }
                }
            }

            tokio::select! {
                biased;
                _ = shutdown_rx.changed() => {
                    return Err(LlamaServerError::ShutdownDuringHealth);
                }
                _ = tokio::time::sleep(self.poll_interval) => {}
            }
        }
    }

    async fn served_by_child(&self) -> bool {
        let (addr, owner) = (self.addr, self.owner);
        tokio::task::spawn_blocking(move || is_listening_in_group(addr, owner))
            .await
            .map_err(io::Error::other)
            .and_then(|owned| owned)
            .unwrap_or_else(|e| {
                debug!(target: "assistd::llama_server", "health: listener ownership: {e}");
                false
            })
    }

    async fn probe(&self) -> Result<bool, reqwest::Error> {
        let resp = self
            .client
            .get(&self.url)
            .timeout(self.probe_timeout)
            .send()
            .await?;
        Ok(resp.status() == reqwest::StatusCode::OK)
    }
}
