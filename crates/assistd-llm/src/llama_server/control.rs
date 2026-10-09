//! HTTP control plane for llama.cpp's router-mode server: load and unload
//! model weights without restarting the process, and query what is loaded.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::debug;

use super::error::LlamaServerError;
use crate::LlmHealthProbe;

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);
const PROPS_TIMEOUT: Duration = Duration::from_secs(2);

/// HTTP client for the model-management endpoints on llama-server.
/// Stateless apart from the connection pool, so it survives server restarts.
#[derive(Debug)]
pub struct LlamaServerControl {
    client: reqwest::Client,
    addr: SocketAddr,
    base_url: String,
    health: Option<Arc<dyn LlmHealthProbe>>,
}

impl LlamaServerControl {
    /// Build a control client for `http://{addr}`. With `health`, a call made
    /// while its child is not serving fails with [`LlamaServerError::NotReady`]
    /// without being sent.
    pub fn new(
        addr: SocketAddr,
        health: Option<Arc<dyn LlmHealthProbe>>,
    ) -> Result<Self, LlamaServerError> {
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(DEFAULT_TIMEOUT)
            .build()?;
        Ok(Self {
            client,
            addr,
            base_url: format!("http://{addr}"),
            health,
        })
    }

    /// `GET /props` on the server, under a short probe timeout.
    pub async fn props(&self) -> Result<Value, LlamaServerError> {
        self.fetch_props(&self.base_url).await
    }

    /// `GET /props` on the router child listening on `port` of the same
    /// host, under the same timeout as [`Self::props`].
    pub async fn child_props(&self, port: u16) -> Result<Value, LlamaServerError> {
        let child_addr = SocketAddr::new(self.addr.ip(), port);
        self.fetch_props(&format!("http://{child_addr}")).await
    }

    /// Ask the server to load `model`. Returns once the request is
    /// acknowledged, which is before the weights are live; see
    /// [`Self::wait_for_loaded`].
    pub async fn load_model(&self, model: &str) -> Result<(), LlamaServerError> {
        self.post_model_action("/models/load", model).await
    }

    /// Ask the server to unload `model`, freeing its VRAM while keeping
    /// the process alive.
    pub async fn unload_model(&self, model: &str) -> Result<(), LlamaServerError> {
        self.post_model_action("/models/unload", model).await
    }

    /// Whether `GET /models` reports `model` as loaded.
    pub async fn model_is_loaded(&self, model: &str) -> Result<bool, LlamaServerError> {
        Ok(self.fetch_models().await?.contains_loaded(model))
    }

    /// Port of the router child that hosts `model`, or `None` when the
    /// model is missing, unloaded, or the server is not in router mode.
    pub async fn find_loaded_child_port(
        &self,
        model: &str,
    ) -> Result<Option<u16>, LlamaServerError> {
        Ok(self.fetch_models().await?.find_loaded_child_port(model))
    }

    /// Poll `/models` until `model` reports loaded, erroring with
    /// [`LlamaServerError::LoadTimeout`] once `deadline` elapses.
    pub async fn wait_for_loaded(
        &self,
        model: &str,
        deadline: Duration,
        poll_interval: Duration,
    ) -> Result<(), LlamaServerError> {
        let start = tokio::time::Instant::now();
        loop {
            if let Ok(true) = self.model_is_loaded(model).await {
                return Ok(());
            }
            if start.elapsed() >= deadline {
                return Err(LlamaServerError::LoadTimeout {
                    model: model.to_string(),
                    timeout: deadline,
                });
            }
            tokio::time::sleep(poll_interval).await;
        }
    }

    async fn require_serving(&self) -> Result<(), LlamaServerError> {
        match &self.health {
            Some(probe) if !probe.is_serving().await => Err(LlamaServerError::NotReady),
            _ => Ok(()),
        }
    }

    async fn fetch_models(&self) -> Result<ModelsResponse, LlamaServerError> {
        self.require_serving().await?;
        let url = format!("{}/models", self.base_url);
        let resp = self.client.get(&url).send().await?;
        let status = resp.status();
        if !status.is_success() {
            return Err(control_http_error("GET", "/models", status));
        }
        Ok(resp.json().await?)
    }

    async fn fetch_props(&self, base_url: &str) -> Result<Value, LlamaServerError> {
        self.require_serving().await?;
        let resp = self
            .client
            .get(format!("{base_url}/props"))
            .timeout(PROPS_TIMEOUT)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            return Err(control_http_error("GET", "/props", status));
        }
        Ok(resp.json().await?)
    }

    async fn post_model_action(
        &self,
        path: &'static str,
        model: &str,
    ) -> Result<(), LlamaServerError> {
        self.require_serving().await?;
        let url = format!("{}{}", self.base_url, path);
        let body = ModelActionRequest { model };
        debug!(target: "assistd::llama_server", "POST {url} model={model}");
        let resp = self.client.post(&url).json(&body).send().await?;
        let status = resp.status();
        if !status.is_success() {
            return Err(control_http_error("POST", path, status));
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct ModelActionRequest<'a> {
    model: &'a str,
}

#[derive(Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    data: Vec<ModelEntry>,
}

impl ModelsResponse {
    fn contains_loaded(&self, model: &str) -> bool {
        self.data.iter().any(|e| e.matches(model) && e.is_loaded())
    }

    fn find_loaded_child_port(&self, model: &str) -> Option<u16> {
        self.data
            .iter()
            .find(|e| e.matches(model) && e.is_loaded())
            .and_then(ModelEntry::child_port)
    }
}

#[derive(Deserialize)]
struct ModelEntry {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    status: Option<ModelStatus>,
}

impl ModelEntry {
    fn matches(&self, model: &str) -> bool {
        self.id.as_deref() == Some(model)
    }

    fn is_loaded(&self) -> bool {
        self.status.as_ref().is_some_and(ModelStatus::is_loaded)
    }

    fn child_port(&self) -> Option<u16> {
        self.status.as_ref().and_then(ModelStatus::child_port)
    }
}

#[derive(Deserialize)]
struct ModelStatus {
    value: String,
    #[serde(default)]
    args: Vec<String>,
}

impl ModelStatus {
    fn is_loaded(&self) -> bool {
        self.value == "loaded"
    }

    fn child_port(&self) -> Option<u16> {
        let mut args = self.args.iter();
        while let Some(arg) = args.next() {
            if arg == "--port" {
                return args.next().and_then(|s| s.parse::<u16>().ok());
            }
        }
        None
    }
}

fn control_http_error(
    method: &'static str,
    path: &'static str,
    status: reqwest::StatusCode,
) -> LlamaServerError {
    LlamaServerError::ControlHttp {
        method,
        path,
        status: status.as_u16(),
    }
}

#[cfg(test)]
mod tests {
    use std::net::Ipv4Addr;

    use async_trait::async_trait;
    use tokio::net::TcpListener;

    use super::*;
    use crate::{HealthSnapshot, HealthWaitError, ReadyState};

    #[derive(Debug)]
    struct RestartingProbe;

    #[async_trait]
    impl LlmHealthProbe for RestartingProbe {
        async fn snapshot(&self) -> Option<HealthSnapshot> {
            Some(HealthSnapshot {
                state: ReadyState::BackingOff { attempt: 1 },
                pid: None,
            })
        }

        async fn wait_for_ready(&self, _timeout: Duration) -> Result<(), HealthWaitError> {
            Err(HealthWaitError::Timeout)
        }
    }

    fn parse(body: &str) -> ModelsResponse {
        serde_json::from_str(body).unwrap()
    }

    #[test]
    fn find_loaded_child_port_reads_the_port_spawn_arg() {
        let cases = [
            (
                "loaded, port among other args",
                r#"{"id":"foo/bar:Q4","status":{"value":"loaded","args":["--host","127.0.0.1","--port","48881","--alias","foo/bar:Q4"]}}"#,
                "foo/bar:Q4",
                Some(48881),
            ),
            (
                "unloaded model",
                r#"{"id":"foo/bar:Q4","status":{"value":"unloaded","args":["--port","48881"]}}"#,
                "foo/bar:Q4",
                None,
            ),
            (
                "unknown model",
                r#"{"id":"foo/bar:Q4","status":{"value":"loaded","args":["--port","48881"]}}"#,
                "other/model:Q4",
                None,
            ),
            (
                "non-numeric port",
                r#"{"id":"foo/bar:Q4","status":{"value":"loaded","args":["--port","auto"]}}"#,
                "foo/bar:Q4",
                None,
            ),
        ];
        for (label, entry, model, expected) in cases {
            let parsed = parse(&format!(r#"{{"data":[{entry}]}}"#));
            assert_eq!(parsed.find_loaded_child_port(model), expected, "{label}");
        }
    }

    #[tokio::test]
    async fn calls_are_not_sent_while_the_server_is_not_serving() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let control = LlamaServerControl::new(
            listener.local_addr().unwrap(),
            Some(Arc::new(RestartingProbe)),
        )
        .unwrap();

        let results = [
            control.load_model("m").await,
            control.unload_model("m").await,
            control.model_is_loaded("m").await.map(drop),
            control.props().await.map(drop),
        ];
        for result in results {
            assert!(
                matches!(result, Err(LlamaServerError::NotReady)),
                "{result:?}"
            );
        }
        assert!(
            tokio::time::timeout(Duration::from_millis(100), listener.accept())
                .await
                .is_err(),
            "a control request reached the listener"
        );
    }
}
