//! Vision-capability probe: asks `/props` for the loaded model's
//! `modalities`, since llama.cpp loads a projector implicitly.

use serde_json::Value;
use tracing::{debug, warn};

use super::control::LlamaServerControl;
use super::error::LlamaServerError;

/// Snapshot of one `/props` probe. `model_id` is `None` when the probe
/// never reached the loaded model.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VisionState {
    pub model_id: Option<String>,
    pub vision_supported: bool,
}

/// Probe the loaded model's capabilities through `control`, following a
/// router's `/models` to the child hosting `model`. Fails closed: any error
/// or missing field yields the default (no vision).
pub async fn probe_capabilities_routed(control: &LlamaServerControl, model: &str) -> VisionState {
    let Some(body) = props_or_warn(control.props().await) else {
        return VisionState::default();
    };

    if !is_router_props(&body) {
        return VisionState {
            model_id: parse_model_id(&body),
            vision_supported: parse_vision_supported(&body),
        };
    }

    let Some(child_port) = bound_child_port(control, model).await else {
        return VisionState::default();
    };

    let Some(child_body) = props_or_warn(control.child_props(child_port).await) else {
        return VisionState::default();
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

async fn bound_child_port(control: &LlamaServerControl, model: &str) -> Option<u16> {
    match control.find_loaded_child_port(model).await {
        Ok(Some(port)) if port != 0 => Some(port),
        Ok(Some(_)) => {
            debug!(
                target: "assistd::llama_server",
                "router /models reports `--port 0` for {model}; child not yet bound — \
                 reporting vision unsupported until the next probe"
            );
            None
        }
        Ok(None) => {
            debug!(
                target: "assistd::llama_server",
                "router /models has no loaded entry for {model}; reporting vision \
                 unsupported until the next probe"
            );
            None
        }
        Err(e) => {
            warn!(
                target: "assistd::llama_server",
                "router /models lookup for {model} failed: {e}"
            );
            None
        }
    }
}

fn is_router_props(body: &Value) -> bool {
    body.get("role").and_then(Value::as_str) == Some("router")
}

fn props_or_warn(props: Result<Value, LlamaServerError>) -> Option<Value> {
    props
        .inspect_err(|e| warn!(target: "assistd::llama_server", "/props probe failed: {e}"))
        .ok()
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
    use serde_json::json;

    use super::*;

    #[test]
    fn vision_requires_modalities_vision_true() {
        let cases = [
            (
                json!({"modalities": {"vision": true}, "total_slots": 1}),
                true,
            ),
            (json!({"modalities": {"vision": false}}), false),
            (json!({"modalities": {"vision": "true"}}), false),
            (json!({"modalities": {"audio": true}}), false),
            (json!({"model_path": "/some/model.gguf"}), false),
            (json!({}), false),
        ];
        for (body, expected) in cases {
            assert_eq!(parse_vision_supported(&body), expected, "{body}");
        }
    }

    #[test]
    fn model_id_comes_from_the_first_string_field_present() {
        let cases = [
            (
                json!({"model": "bartowski/Qwen3-14B-GGUF:Q4_K_M"}),
                Some("bartowski/Qwen3-14B-GGUF:Q4_K_M"),
            ),
            (
                json!({"model_path": "/var/cache/llm/qwen-vl.gguf"}),
                Some("/var/cache/llm/qwen-vl.gguf"),
            ),
            (
                json!({"default_generation_settings": {"model": "nested/model"}}),
                Some("nested/model"),
            ),
            (json!({"total_slots": 1}), None),
            (json!({"model": 42}), None),
        ];
        for (body, expected) in cases {
            assert_eq!(parse_model_id(&body).as_deref(), expected, "{body}");
        }
    }

    #[test]
    fn router_props_are_identified_by_role() {
        let cases = [
            (
                json!({"role": "router", "max_instances": 4, "model_path": "none"}),
                true,
            ),
            (
                json!({"model": null, "model_path": "/some/model.gguf", "modalities": {"vision": true}}),
                false,
            ),
            (json!({"role": "worker", "model_path": "/x.gguf"}), false),
        ];
        for (body, expected) in cases {
            assert_eq!(is_router_props(&body), expected, "{body}");
        }
    }
}
