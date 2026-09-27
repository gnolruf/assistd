use std::io;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use rustix::process::Pid;
use tokio::sync::watch;
use tracing::{debug, warn};

use super::error::ChildServerError;
use crate::procfs::is_listening_in_group;

const POLL_INTERVAL: Duration = Duration::from_millis(250);
const PROBE_TIMEOUT: Duration = Duration::from_secs(1);

/// Polls `/health` until the child reports ready or a timeout elapses.
/// A 200 only counts when the listener belongs to the child's process group.
pub(super) struct HealthChecker {
    server: &'static str,
    client: reqwest::Client,
    addr: SocketAddr,
    owner: Pid,
    url: String,
    ready_timeout: Duration,
}

impl HealthChecker {
    /// Checker for `http://{addr}/health` served by process group `owner`;
    /// `ready_timeout` bounds each [`Self::wait_ready`].
    pub(super) fn new(
        server: &'static str,
        addr: SocketAddr,
        owner: Pid,
        ready_timeout: Duration,
    ) -> Result<Self, ChildServerError> {
        let client = reqwest::Client::builder()
            .no_proxy()
            .connect_timeout(PROBE_TIMEOUT)
            .build()?;
        Ok(Self {
            server,
            client,
            addr,
            owner,
            url: format!("http://{addr}/health"),
            ready_timeout,
        })
    }

    /// Poll until the child's listener returns 200 OK, the deadline elapses,
    /// or shutdown is requested. Non-200 responses, foreign listeners, and
    /// transport errors keep polling.
    pub(super) async fn wait_ready(
        &self,
        shutdown_rx: &mut watch::Receiver<bool>,
    ) -> Result<(), ChildServerError> {
        if *shutdown_rx.borrow() {
            return Err(ChildServerError::ShutdownDuringHealth);
        }

        let deadline = Instant::now() + self.ready_timeout;
        loop {
            if Instant::now() >= deadline {
                return Err(ChildServerError::HealthTimeout {
                    server: self.server,
                    timeout: self.ready_timeout,
                });
            }

            tokio::select! {
                biased;
                _ = shutdown_rx.changed() => {
                    return Err(ChildServerError::ShutdownDuringHealth);
                }
                probe = self.probe() => {
                    match probe {
                        Ok(true) if self.served_by_child().await => return Ok(()),
                        Ok(true) => warn!(
                            target: "assistd::child_server",
                            server = self.server,
                            "health: ignoring 200 from {}, which {} does not own",
                            self.addr,
                            self.server,
                        ),
                        Ok(false) => debug!(
                            target: "assistd::child_server",
                            server = self.server,
                            "health: non-200 response"
                        ),
                        Err(e) => debug!(
                            target: "assistd::child_server",
                            server = self.server,
                            "health: {e}"
                        ),
                    }
                }
            }

            tokio::select! {
                biased;
                _ = shutdown_rx.changed() => {
                    return Err(ChildServerError::ShutdownDuringHealth);
                }
                _ = tokio::time::sleep(POLL_INTERVAL) => {}
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
                debug!(
                    target: "assistd::child_server",
                    server = self.server,
                    "health: listener ownership: {e}"
                );
                false
            })
    }

    async fn probe(&self) -> Result<bool, reqwest::Error> {
        let response = self
            .client
            .get(&self.url)
            .timeout(PROBE_TIMEOUT)
            .send()
            .await?;
        Ok(response.status() == reqwest::StatusCode::OK)
    }
}
