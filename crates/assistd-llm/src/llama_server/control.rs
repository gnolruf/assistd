//! HTTP control plane for the managed llama-server.
//!
//! llama.cpp's router-mode server exposes `POST /models/load` and
//! `POST /models/unload` endpoints that attach/detach model weights without
//! restarting the process. `LlamaServerControl` wraps those calls so the
//! presence-state machine can free VRAM on `drowse` and rehydrate on `wake`
//! without paying the cost of spawning a new child.
//!
//! The server is assumed to be listening already; callers that also need to
//! supervise the child process should use [`super::LlamaService`] for that.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use tracing::debug;

use super::error::LlamaServerError;

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);

/// Small HTTP client for the model-management endpoints on llama-server.
///
/// Stateless modulo the underlying `reqwest::Client`'s connection pool — safe
/// to reuse across many calls and across process restarts of the server
/// (the base URL is fixed).
pub struct LlamaServerControl {
    client: reqwest::Client,
    base_url: String,
}

impl LlamaServerControl {
    pub fn new(host: &str, port: u16) -> Result<Self, LlamaServerError> {
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(DEFAULT_TIMEOUT)
            .build()?;
        Ok(Self {
            client,
            base_url: format!("http://{host}:{port}"),
        })
    }

    /// Asks the server to load `model` into memory. Returns once the server
    /// acknowledges the request; the server itself is responsible for
    /// blocking the response until the weights are resident.
    pub async fn load_model(&self, model: &str) -> Result<(), LlamaServerError> {
        self.post_model_action("/models/load", model).await
    }

    /// Asks the server to unload `model`, freeing its VRAM while keeping
    /// the process alive.
    pub async fn unload_model(&self, model: &str) -> Result<(), LlamaServerError> {
        self.post_model_action("/models/unload", model).await
    }

    /// Best-effort check for whether `model` is currently loaded.
    ///
    /// Queries `GET /models` and looks for an entry matching `model` with
    /// a loaded status marker. Resilient to different response shapes —
    /// tolerates either a top-level `data`/`models` array and either a
    /// `status: "loaded"` field or a boolean `loaded` field.
    pub async fn model_is_loaded(&self, model: &str) -> Result<bool, LlamaServerError> {
        let url = format!("{}/models", self.base_url);
        let resp = self.client.get(&url).send().await?;
        let status = resp.status();
        if !status.is_success() {
            return Err(control_http_error("GET", "/models", status));
        }
        let body: ModelsResponse = resp.json().await?;
        Ok(body.contains_loaded(model))
    }

    async fn post_model_action(
        &self,
        path: &'static str,
        model: &str,
    ) -> Result<(), LlamaServerError> {
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

#[derive(Serialize)]
struct ModelActionRequest<'a> {
    model: &'a str,
}

#[derive(Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    data: Vec<ModelEntry>,
    #[serde(default)]
    models: Vec<ModelEntry>,
}

impl ModelsResponse {
    fn contains_loaded(&self, model: &str) -> bool {
        self.data
            .iter()
            .chain(self.models.iter())
            .any(|e| e.matches(model) && e.is_loaded())
    }
}

#[derive(Deserialize)]
struct ModelEntry {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    loaded: Option<bool>,
}

impl ModelEntry {
    fn matches(&self, model: &str) -> bool {
        self.id.as_deref() == Some(model) || self.name.as_deref() == Some(model)
    }

    fn is_loaded(&self) -> bool {
        if let Some(true) = self.loaded {
            return true;
        }
        matches!(self.status.as_deref(), Some("loaded"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_response_recognises_loaded_status_field() {
        let body = r#"{"data":[{"id":"a","status":"loaded"},{"id":"b","status":"unloaded"}]}"#;
        let parsed: ModelsResponse = serde_json::from_str(body).unwrap();
        assert!(parsed.contains_loaded("a"));
        assert!(!parsed.contains_loaded("b"));
        assert!(!parsed.contains_loaded("c"));
    }

    #[test]
    fn models_response_recognises_loaded_bool_field() {
        let body = r#"{"models":[{"name":"x","loaded":true}]}"#;
        let parsed: ModelsResponse = serde_json::from_str(body).unwrap();
        assert!(parsed.contains_loaded("x"));
    }

    #[test]
    fn models_response_empty_is_not_loaded() {
        let body = r#"{"data":[]}"#;
        let parsed: ModelsResponse = serde_json::from_str(body).unwrap();
        assert!(!parsed.contains_loaded("anything"));
    }
}
