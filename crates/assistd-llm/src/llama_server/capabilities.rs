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

/// Returns `true` iff the running llama-server reports a loaded vision
/// encoder. Returns `false` on any error (HTTP failure, non-200,
/// malformed JSON) and logs at `warn` so the daemon's startup log
/// captures the reason. The caller should treat a `false` result as
/// "vision unavailable" and fall back accordingly.
pub async fn detect_vision_support(host: &str, port: u16) -> bool {
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
                "vision capability probe: failed to build HTTP client: {e}; assuming no vision support"
            );
            return false;
        }
    };

    let resp = match client.get(&url).send().await {
        Ok(r) => r,
        Err(e) => {
            warn!(
                target: "assistd::llama_server",
                "vision capability probe: GET /props failed: {e}; assuming no vision support"
            );
            return false;
        }
    };
    if !resp.status().is_success() {
        warn!(
            target: "assistd::llama_server",
            "vision capability probe: GET /props returned {}; assuming no vision support",
            resp.status()
        );
        return false;
    }

    let body: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            warn!(
                target: "assistd::llama_server",
                "vision capability probe: /props body was not JSON: {e}; assuming no vision support"
            );
            return false;
        }
    };

    let supported = parse_vision_supported(&body);
    debug!(
        target: "assistd::llama_server",
        "vision capability probe: /props parsed; vision_supported = {supported}"
    );
    supported
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
}
