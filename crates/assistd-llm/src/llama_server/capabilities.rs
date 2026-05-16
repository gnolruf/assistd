//! Capability probe for the running llama-server.
//!
//! After [`super::LlamaService::start`] reports the child as ready
//! (`/health` = 200), call [`probe_capabilities_routed`] (or
//! [`detect_vision_support`] for the boolean-only variant) to decide
//! whether the loaded model has a multimodal projector (mmproj)
//! bundled. llama.cpp auto-loads the projector when the HF repo
//! carries one alongside the text weights; there is no separate
//! config flag to opt in. The only way to know is to ask the server.
//!
//! Router-mode awareness: when llama-server runs as a router, the
//! daemon-managed port hosts only the router process, and `/props`
//! there reports `role: "router"` with no model info — the real model
//! (and the real `/props`, including `modalities`) lives in a child
//! server spawned on a transient port. [`probe_capabilities_routed`]
//! detects this by checking the `role` field, then consults
//! [`super::LlamaServerControl::find_loaded_child_port`] to discover
//! the child's port and re-probes `/props` there. Non-router setups
//! see no detour: the first probe is the answer.
//!
//! The probe intentionally fails-closed: any HTTP error, parse
//! failure, or absent capability field collapses to
//! `vision_enabled = false`. The image-producing tools (`see`,
//! `screenshot`, `/attach`) refuse with a navigation-compliant
//! `[error] …: vision not available …` line in that case, which is a
//! safer default than silently sending image bytes the model will drop.
//!
//! We read the capability from llama.cpp's canonical structured
//! `modalities` object (`{"modalities": {"vision": true}}`).

use std::time::Duration;

use serde_json::Value;
use tracing::{debug, warn};

use super::control::LlamaServerControl;

const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// Snapshot of a single `/props` probe. `model_id` is whatever
/// llama-server reports as the loaded model, used by callers to
/// detect a model swap between probes and trigger a re-probe rather
/// than trusting a stale `vision_supported` value.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VisionState {
    pub model_id: Option<String>,
    pub vision_supported: bool,
}

/// Probe llama-server for both the model identifier and vision
/// capability. Returns a default (no model id, no vision) on any
/// failure, with the same fail-closed semantics as the historical
/// [`detect_vision_support`] entrypoint.
pub async fn probe_capabilities(host: &str, port: u16) -> VisionState {
    let body = match fetch_props(host, port).await {
        Some(v) => v,
        None => return VisionState::default(),
    };
    let model_id = parse_model_id(&body);
    let vision_supported = parse_vision_supported(&body);
    debug!(
        target: "assistd::llama_server",
        "/props parsed; model_id = {model_id:?}, vision_supported = {vision_supported}"
    );
    VisionState {
        model_id,
        vision_supported,
    }
}

/// Returns `true` iff the running llama-server reports a loaded vision
/// encoder. Returns `false` on any error (HTTP failure, non-200,
/// malformed JSON) and logs at `warn` so the daemon's startup log
/// captures the reason. The caller should treat a `false` result as
/// "vision unavailable" and fall back accordingly.
///
/// Thin wrapper around [`probe_capabilities`] preserved for callers
/// that don't need the model id.
pub async fn detect_vision_support(host: &str, port: u16) -> bool {
    probe_capabilities(host, port).await.vision_supported
}

/// Probe the *effective* model's capabilities, transparently following
/// router indirection when needed.
///
/// `host`/`port` identify the daemon-managed llama-server. `model` is
/// the configured model id (e.g. `unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_XL`)
/// — used to look up the right child when the server is in router
/// mode. `control` is the same HTTP client the presence layer uses, so
/// router-mode discovery reuses the configured base URL and connection
/// pool.
///
/// Behaviour:
/// - Direct mode: `/props` carries the model info → return it.
/// - Router mode (`role == "router"` in `/props`): consult `/models`
///   for the loaded child's port, then probe `/props` on that child.
/// - Anything that fails along the way collapses to the default
///   (vision off, no model id) so the gate fails closed.
pub async fn probe_capabilities_routed(
    host: &str,
    port: u16,
    model: &str,
    control: &LlamaServerControl,
) -> VisionState {
    let body = match fetch_props(host, port).await {
        Some(v) => v,
        None => return VisionState::default(),
    };

    if !is_router_props(&body) {
        return VisionState {
            model_id: parse_model_id(&body),
            vision_supported: parse_vision_supported(&body),
        };
    }

    let child_port = match control.find_loaded_child_port(model).await {
        Ok(Some(p)) if p != 0 => p,
        Ok(Some(_)) => {
            debug!(
                target: "assistd::llama_server",
                "router /models reports `--port 0` for {model}; child not yet bound — \
                 reporting vision unsupported until the next probe"
            );
            return VisionState::default();
        }
        Ok(None) => {
            debug!(
                target: "assistd::llama_server",
                "router /models has no loaded entry for {model}; reporting vision \
                 unsupported until the next probe"
            );
            return VisionState::default();
        }
        Err(e) => {
            warn!(
                target: "assistd::llama_server",
                "router /models lookup for {model} failed: {e}"
            );
            return VisionState::default();
        }
    };

    let child_body = match fetch_props(host, child_port).await {
        Some(v) => v,
        None => return VisionState::default(),
    };
    let vision_supported = parse_vision_supported(&child_body);

    let model_id = parse_model_id(&child_body).or_else(|| Some(model.to_string()));
    debug!(
        target: "assistd::llama_server",
        "router child probe: model={model_id:?}, vision_supported={vision_supported}, \
         child_port={child_port}"
    );
    VisionState {
        model_id,
        vision_supported,
    }
}

fn is_router_props(body: &Value) -> bool {
    body.get("role").and_then(Value::as_str) == Some("router")
}

async fn fetch_props(host: &str, port: u16) -> Option<Value> {
    let url = format!("http://{host}:{port}/props");
    let client = match reqwest::Client::builder()
        .no_proxy()
        .timeout(PROBE_TIMEOUT)
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            warn!(
                target: "assistd::llama_server",
                "/props probe: failed to build HTTP client: {e}"
            );
            return None;
        }
    };
    let resp = match client.get(&url).send().await {
        Ok(r) => r,
        Err(e) => {
            warn!(
                target: "assistd::llama_server",
                "/props probe: GET failed: {e}"
            );
            return None;
        }
    };
    if !resp.status().is_success() {
        warn!(
            target: "assistd::llama_server",
            "/props probe: GET returned {}",
            resp.status()
        );
        return None;
    }
    match resp.json::<Value>().await {
        Ok(v) => Some(v),
        Err(e) => {
            warn!(
                target: "assistd::llama_server",
                "/props body was not JSON: {e}"
            );
            None
        }
    }
}

fn parse_model_id(body: &Value) -> Option<String> {
    const TOP_LEVEL_KEYS: &[&str] = &["model", "model_path", "model_name"];
    for key in TOP_LEVEL_KEYS {
        if let Some(s) = body.get(*key).and_then(Value::as_str) {
            return Some(s.to_string());
        }
    }
    if let Some(settings) = body.get("default_generation_settings")
        && let Some(s) = settings.get("model").and_then(Value::as_str)
    {
        return Some(s.to_string());
    }
    None
}

fn parse_vision_supported(body: &Value) -> bool {
    body.get("modalities")
        .and_then(|m| m.get("vision"))
        .and_then(Value::as_bool)
        == Some(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn detects_modalities_vision_true() {
        let body = json!({"modalities": {"vision": true}, "total_slots": 1});
        assert!(parse_vision_supported(&body));
    }

    #[test]
    fn modalities_vision_false_does_not_promote_to_true() {
        let body = json!({"modalities": {"vision": false}});
        assert!(!parse_vision_supported(&body));
    }

    #[test]
    fn ignores_non_bool_modalities_vision() {
        let body = json!({"modalities": {"vision": "true"}});
        assert!(!parse_vision_supported(&body));
    }

    #[test]
    fn defaults_false_when_modalities_absent() {
        let body = json!({"total_slots": 1, "model_path": "/some/model.gguf"});
        assert!(!parse_vision_supported(&body));
    }

    #[test]
    fn defaults_false_when_modalities_lacks_vision_key() {
        let body = json!({"modalities": {"audio": true}});
        assert!(!parse_vision_supported(&body));
    }

    #[test]
    fn empty_object_is_not_vision_supported() {
        let body = json!({});
        assert!(!parse_vision_supported(&body));
    }

    #[test]
    fn parses_top_level_model_field() {
        let body = json!({"model": "bartowski/Qwen3-14B-GGUF:Q4_K_M"});
        assert_eq!(
            parse_model_id(&body).as_deref(),
            Some("bartowski/Qwen3-14B-GGUF:Q4_K_M")
        );
    }

    #[test]
    fn parses_top_level_model_path_when_model_absent() {
        let body = json!({"model_path": "/var/cache/llm/qwen-vl.gguf"});
        assert_eq!(
            parse_model_id(&body).as_deref(),
            Some("/var/cache/llm/qwen-vl.gguf")
        );
    }

    #[test]
    fn parses_nested_default_generation_settings_model() {
        let body = json!({
            "default_generation_settings": {"model": "nested/model"},
        });
        assert_eq!(parse_model_id(&body).as_deref(), Some("nested/model"));
    }

    #[test]
    fn returns_none_when_no_model_field_present() {
        let body = json!({"total_slots": 1});
        assert_eq!(parse_model_id(&body), None);
    }

    #[test]
    fn ignores_non_string_model_value() {
        let body = json!({"model": 42});
        assert_eq!(parse_model_id(&body), None);
    }

    #[test]
    fn detects_router_props_by_role_field() {
        let body = json!({"role": "router", "max_instances": 4, "model_path": "none"});
        assert!(is_router_props(&body));
    }

    #[test]
    fn non_router_props_lack_router_role() {
        let body = json!({"model": null, "model_path": "/some/model.gguf", "modalities": {"vision": true}});
        assert!(!is_router_props(&body));
    }

    #[test]
    fn non_router_props_with_unrelated_role_value() {
        let body = json!({"role": "worker", "model_path": "/x.gguf"});
        assert!(!is_router_props(&body));
    }
}
