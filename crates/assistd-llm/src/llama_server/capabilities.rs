//! Capability probe for the running llama-server.
//!
//! After [`super::LlamaService::start`] reports the child as ready
//! (`/health` = 200), call [`detect_vision_support`] to decide whether
//! the loaded model has a multimodal projector (mmproj) bundled.
//! llama.cpp auto-loads the projector when the HF repo carries one
//! alongside the text weights — there is no separate config flag to
//! opt in. The only way to know is to ask the server.
//!
//! The probe intentionally fails-closed: any HTTP error, parse
//! failure, or absent capability field collapses to
//! `vision_enabled = false`. The image-producing tools (`see`,
//! `screenshot`, `/attach`) refuse with a navigation-compliant
//! `[error] …: vision not available …` line in that case, which is a
//! safer default than silently sending image bytes the model will drop.
//!
//! Field-name spread: across the multimodal-merge era llama.cpp has
//! exposed the capability under several different keys. We tolerate
//! the spread by checking a small union: any of `has_multimodal`,
//! `has_vision`, `multimodal`, or
//! `default_generation_settings.multimodal` set to `true` enables
//! vision. New names can be added here without touching call sites.

use std::time::Duration;

use serde_json::Value;
use tracing::{debug, warn};

const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// Snapshot of a single `/props` probe. `model_id` is whatever
/// llama-server reports as the loaded model — used by callers to
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

/// Pull a model identifier from a `/props` body. llama.cpp has reported
/// this under a few different keys (`model`, `model_path`,
/// `default_generation_settings.model`), so try them in order. Returns
/// `None` if no recognizable field is present.
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

/// Inspect a parsed `/props` body for any field that signals
/// multimodal/vision support. Tolerates the historical spread of
/// field names in llama.cpp.
fn parse_vision_supported(body: &Value) -> bool {
    const TOP_LEVEL_KEYS: &[&str] = &["has_multimodal", "has_vision", "multimodal"];
    for key in TOP_LEVEL_KEYS {
        if body.get(*key).and_then(Value::as_bool) == Some(true) {
            return true;
        }
    }
    if let Some(settings) = body.get("default_generation_settings")
        && settings.get("multimodal").and_then(Value::as_bool) == Some(true)
    {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn detects_top_level_has_multimodal_true() {
        let body = json!({"has_multimodal": true, "total_slots": 1});
        assert!(parse_vision_supported(&body));
    }

    #[test]
    fn detects_top_level_has_vision_true() {
        let body = json!({"has_vision": true});
        assert!(parse_vision_supported(&body));
    }

    #[test]
    fn detects_top_level_multimodal_true() {
        let body = json!({"multimodal": true});
        assert!(parse_vision_supported(&body));
    }

    #[test]
    fn detects_nested_multimodal_under_default_generation_settings() {
        let body = json!({
            "default_generation_settings": {"multimodal": true, "temperature": 0.7},
        });
        assert!(parse_vision_supported(&body));
    }

    #[test]
    fn ignores_field_when_explicitly_false() {
        let body = json!({"has_multimodal": false, "has_vision": false});
        assert!(!parse_vision_supported(&body));
    }

    #[test]
    fn defaults_false_when_no_field_present() {
        let body = json!({"total_slots": 1, "model_path": "/some/model.gguf"});
        assert!(!parse_vision_supported(&body));
    }

    #[test]
    fn nested_false_does_not_promote_to_true() {
        let body = json!({
            "default_generation_settings": {"multimodal": false},
        });
        assert!(!parse_vision_supported(&body));
    }

    #[test]
    fn ignores_non_bool_field_value() {
        // Future-proofing: a string "true" must NOT count. We require
        // a JSON boolean to avoid accidentally enabling vision based on
        // an unrelated string field that happens to share a name.
        let body = json!({"has_multimodal": "true"});
        assert!(!parse_vision_supported(&body));
    }

    #[test]
    fn empty_object_is_not_vision_supported() {
        let body = json!({});
        assert!(!parse_vision_supported(&body));
    }

    #[test]
    fn any_truthy_field_wins_even_if_others_false() {
        let body = json!({
            "has_multimodal": false,
            "has_vision": true,
            "multimodal": false,
        });
        assert!(parse_vision_supported(&body));
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
}
