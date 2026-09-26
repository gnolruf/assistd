use std::io;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use rustix::process::Pid;
use tokio::sync::watch;
use tracing::{debug, warn};

use super::error::EmbedServerError;
use super::listener::is_listening_in_group;

const POLL_INTERVAL: Duration = Duration::from_millis(250);
const PROBE_TIMEOUT: Duration = Duration::from_secs(1);

/// Polls `/health` until it returns 200 or the deadline elapses. A 200
/// only counts when the listener belongs to the child's process group.
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
    /// Checker for `http://{addr}/health` served by process group `owner`;
    /// `ready_timeout` bounds each [`Self::wait_ready`].
    pub fn new(
        addr: SocketAddr,
        owner: Pid,
        ready_timeout: Duration,
    ) -> Result<Self, EmbedServerError> {
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

    /// Poll until the child's listener returns 200; errors with `HealthTimeout` at the deadline or
    /// `ShutdownDuringHealth` when the watch fires.
    pub async fn wait_ready(
        &self,
        shutdown_rx: &mut watch::Receiver<bool>,
    ) -> Result<(), EmbedServerError> {
        if *shutdown_rx.borrow() {
            return Err(EmbedServerError::ShutdownDuringHealth);
        }

        let deadline = Instant::now() + self.ready_timeout;
        loop {
            if Instant::now() >= deadline {
                return Err(EmbedServerError::HealthTimeout {
                    timeout: self.ready_timeout,
                });
            }

            tokio::select! {
                biased;
                _ = shutdown_rx.changed() => {
                    return Err(EmbedServerError::ShutdownDuringHealth);
                }
                probe = self.probe() => {
                    match probe {
                        Ok(true) if self.served_by_child().await => return Ok(()),
                        Ok(true) => warn!(
                            target: "assistd::embed_server",
                            "health: ignoring 200 from {}, which embed-server does not own",
                            self.addr
                        ),
                        Ok(false) => debug!(target: "assistd::embed_server", "health: non-200 response"),
                        Err(e) => debug!(target: "assistd::embed_server", "health: {e}"),
                    }
                }
            }

            tokio::select! {
                biased;
                _ = shutdown_rx.changed() => {
                    return Err(EmbedServerError::ShutdownDuringHealth);
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
                debug!(target: "assistd::embed_server", "health: listener ownership: {e}");
                false
            })
    }

    async fn probe(&self) -> Result<bool, reqwest::Error> {
        let response = self
            .client
            .get(&self.url)
            .timeout(self.probe_timeout)
            .send()
            .await?;
        Ok(response.status() == reqwest::StatusCode::OK)
    }
}
