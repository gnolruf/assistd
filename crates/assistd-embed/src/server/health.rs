use super::error::EmbedServerError;
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tracing::debug;

const POLL_INTERVAL: Duration = Duration::from_millis(250);
const PROBE_TIMEOUT: Duration = Duration::from_secs(1);

/// Polls the embed server's `/health` endpoint until it returns 200 or
/// the deadline elapses. Mirrors `assistd_llm::llama_server::HealthChecker`
/// but with an embed-targeted error type.
pub struct HealthChecker {
    client: reqwest::Client,
    url: String,
    poll_interval: Duration,
    probe_timeout: Duration,
    ready_timeout: Duration,
}

impl HealthChecker {
    pub fn new(host: &str, port: u16, ready_timeout: Duration) -> Result<Self, EmbedServerError> {
        let client = reqwest::Client::builder()
            .no_proxy()
            .connect_timeout(PROBE_TIMEOUT)
            .build()?;
        Ok(Self {
            client,
            url: format!("http://{host}:{port}/health"),
            poll_interval: POLL_INTERVAL,
            probe_timeout: PROBE_TIMEOUT,
            ready_timeout,
        })
    }

    pub fn ready_timeout(&self) -> Duration {
        self.ready_timeout
    }

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
                res = self.probe() => {
                    match res {
                        Ok(true) => return Ok(()),
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
