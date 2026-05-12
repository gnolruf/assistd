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
/// Stateless modulo the underlying `reqwest::Client`'s connection pool, safe
/// to reuse across many calls and across process restarts of the server
/// (the base URL is fixed).
pub struct LlamaServerControl {
    client: reqwest::Client,
    base_url: String,
}

impl LlamaServerControl {
    /// Creates a control client pointing at `http://{host}:{port}`.
    ///
    /// # Errors
    /// Returns [`LlamaServerError`] if the underlying HTTP client cannot be built.
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
    /// a loaded status marker. Resilient to different response shapes:
    /// tolerates either a top-level `data`/`models` array and either a
    /// structured `status: {value: "loaded", args: [...]}` object, a flat
    /// `status: "loaded"` string, or a boolean `loaded` field.
    pub async fn model_is_loaded(&self, model: &str) -> Result<bool, LlamaServerError> {
        Ok(self.fetch_models().await?.contains_loaded(model))
    }

    /// Returns the child server's listening port for `model` if it is
    /// currently loaded, or `None` otherwise.
    ///
    /// In router mode llama-server spawns a child per loaded model on a
    /// transient port encoded in the per-model `status.args` array (the
    /// value following `--port`). Callers need this port to reach the
    /// model directly — `/props`, `/health`, etc. on the router itself
    /// describe the router, not the loaded model.
    ///
    /// Returns `None` when the model is missing, marked unloaded, or
    /// when the response shape doesn't carry the spawn args (e.g.
    /// non-router-mode servers).
    pub async fn find_loaded_child_port(
        &self,
        model: &str,
    ) -> Result<Option<u16>, LlamaServerError> {
        Ok(self.fetch_models().await?.find_loaded_child_port(model))
    }

    /// Polls `/models` until `model` reports `status.value == "loaded"`
    /// or `deadline` elapses.
    ///
    /// `POST /models/load` on the router returns `200` the moment the
    /// router accepts the spawn request — it does not block until the
    /// child has actually finished loading weights + mmproj. Anything
    /// that depends on the model being live (vision probe, first
    /// chat request) needs to wait for this transition explicitly.
    pub async fn wait_for_loaded(
        &self,
        model: &str,
        deadline: Duration,
        poll_interval: Duration,
    ) -> Result<(), LlamaServerError> {
        let start = tokio::time::Instant::now();
        loop {
            // Suppress transient errors during the poll — the router can
            // briefly 5xx while spawning a child. Only the deadline is
            // fatal.
            if let Ok(true) = self.model_is_loaded(model).await {
                return Ok(());
            }
            if start.elapsed() >= deadline {
                return Err(LlamaServerError::HealthTimeout { timeout: deadline });
            }
            tokio::time::sleep(poll_interval).await;
        }
    }

    async fn fetch_models(&self) -> Result<ModelsResponse, LlamaServerError> {
        let url = format!("{}/models", self.base_url);
        let resp = self.client.get(&url).send().await?;
        let status = resp.status();
        if !status.is_success() {
            return Err(control_http_error("GET", "/models", status));
        }
        Ok(resp.json().await?)
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
    fn entries(&self) -> impl Iterator<Item = &ModelEntry> {
        self.data.iter().chain(self.models.iter())
    }

    fn contains_loaded(&self, model: &str) -> bool {
        self.entries().any(|e| e.matches(model) && e.is_loaded())
    }

    fn find_loaded_child_port(&self, model: &str) -> Option<u16> {
        self.entries()
            .find(|e| e.matches(model) && e.is_loaded())
            .and_then(ModelEntry::child_port)
    }
}

#[derive(Deserialize)]
struct ModelEntry {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    status: Option<ModelStatus>,
    #[serde(default)]
    loaded: Option<bool>,
}

/// `/models` historically returned `status` as a flat string
/// (`"loaded"` / `"unloaded"`); current router-mode builds return a
/// structured object with the child's spawn args. We accept both via
/// an untagged enum so an upgrade in either direction keeps parsing.
#[derive(Deserialize)]
#[serde(untagged)]
enum ModelStatus {
    Structured {
        value: String,
        #[serde(default)]
        args: Vec<String>,
    },
    Legacy(String),
}

impl ModelStatus {
    fn is_loaded(&self) -> bool {
        match self {
            ModelStatus::Structured { value, .. } | ModelStatus::Legacy(value) => value == "loaded",
        }
    }

    /// Extract the value following `--port` in the spawn args of a
    /// structured status. Only present when the router actually spawned
    /// a child for this model.
    fn child_port(&self) -> Option<u16> {
        let ModelStatus::Structured { args, .. } = self else {
            return None;
        };
        let mut iter = args.iter();
        while let Some(arg) = iter.next() {
            if arg == "--port" {
                return iter.next().and_then(|s| s.parse::<u16>().ok());
            }
        }
        None
    }
}

impl ModelEntry {
    fn matches(&self, model: &str) -> bool {
        self.id.as_deref() == Some(model) || self.name.as_deref() == Some(model)
    }

    fn is_loaded(&self) -> bool {
        if let Some(true) = self.loaded {
            return true;
        }
        self.status.as_ref().is_some_and(ModelStatus::is_loaded)
    }

    fn child_port(&self) -> Option<u16> {
        self.status.as_ref().and_then(ModelStatus::child_port)
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

    #[test]
    fn models_response_recognises_structured_status_loaded() {
        // Shape the router currently returns: status is an object with
        // `value` and `args` (the child's spawn command line).
        let body = r#"{"data":[
            {"id":"foo/bar:Q4","status":{"value":"loaded","args":["--host","127.0.0.1","--port","48881"]}},
            {"id":"baz/qux:Q4","status":{"value":"unloaded","args":["--host","127.0.0.1","--port","0"]}}
        ]}"#;
        let parsed: ModelsResponse = serde_json::from_str(body).unwrap();
        assert!(parsed.contains_loaded("foo/bar:Q4"));
        assert!(!parsed.contains_loaded("baz/qux:Q4"));
    }

    #[test]
    fn find_loaded_child_port_extracts_port_from_spawn_args() {
        let body = r#"{"data":[
            {"id":"foo/bar:Q4","status":{"value":"loaded","args":["--host","127.0.0.1","--port","48881","--alias","foo/bar:Q4"]}}
        ]}"#;
        let parsed: ModelsResponse = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.find_loaded_child_port("foo/bar:Q4"), Some(48881));
    }

    #[test]
    fn find_loaded_child_port_returns_none_for_unloaded_model() {
        // Even with a `--port` in args, an unloaded entry has no live
        // child — surface None rather than a stale port.
        let body = r#"{"data":[
            {"id":"foo/bar:Q4","status":{"value":"unloaded","args":["--port","48881"]}}
        ]}"#;
        let parsed: ModelsResponse = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.find_loaded_child_port("foo/bar:Q4"), None);
    }

    #[test]
    fn find_loaded_child_port_returns_none_when_args_missing() {
        // Legacy/flat status carries no spawn args; vision detection
        // falls back to direct probing in that case.
        let body = r#"{"data":[{"id":"foo/bar:Q4","status":"loaded"}]}"#;
        let parsed: ModelsResponse = serde_json::from_str(body).unwrap();
        assert!(parsed.contains_loaded("foo/bar:Q4"));
        assert_eq!(parsed.find_loaded_child_port("foo/bar:Q4"), None);
    }

    #[test]
    fn find_loaded_child_port_returns_none_for_unknown_model() {
        let body = r#"{"data":[
            {"id":"foo/bar:Q4","status":{"value":"loaded","args":["--port","48881"]}}
        ]}"#;
        let parsed: ModelsResponse = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.find_loaded_child_port("other/model:Q4"), None);
    }

    #[test]
    fn find_loaded_child_port_ignores_non_numeric_port() {
        // Future-proof: if `--port` is followed by a non-numeric token,
        // refuse to invent one rather than panicking.
        let body = r#"{"data":[
            {"id":"foo/bar:Q4","status":{"value":"loaded","args":["--port","auto"]}}
        ]}"#;
        let parsed: ModelsResponse = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.find_loaded_child_port("foo/bar:Q4"), None);
    }

    #[test]
    fn find_loaded_child_port_handles_port_zero_as_unbound() {
        // The router stores `--port 0` for entries it has never spawned;
        // we parse it as 0 which callers should treat as "no live child."
        // A real spawn replaces 0 with the actual bound port.
        let body = r#"{"data":[
            {"id":"foo/bar:Q4","status":{"value":"loaded","args":["--port","0"]}}
        ]}"#;
        let parsed: ModelsResponse = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.find_loaded_child_port("foo/bar:Q4"), Some(0));
    }
}
