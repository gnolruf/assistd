//! Popup data model and per-turn coalescing.

use std::collections::{HashMap, HashSet};

use assistd_config::TrayPopupConfig;
use assistd_ipc::Event;
use serde_json::Value;

/// Snapshot pushed through the `watch` channel to the GUI thread.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PopupState {
    pub body: String,
    pub footer: Option<ToolCallLine>,
    pub activity: PopupActivity,
    pub visible: bool,
}

/// Coarse activity classification rendered as a one-line status above
/// the popup body.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PopupActivity {
    #[default]
    Idle,
    Streaming,
    Thinking,
    RunningTool {
        name: String,
    },
    Listening,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TurnActivity {
    Streaming,
    Thinking,
    RunningTool(String),
    Finished,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallLine {
    pub name: String,
    pub args_summary: String,
}

#[derive(Debug, Clone, Default)]
struct TurnState {
    body: String,
    footer: Option<ToolCallLine>,
    activity: Option<TurnActivity>,
}

/// Tracks every signal the popup cares about and produces a
/// [`PopupState`] snapshot on demand.
#[derive(Debug, Default)]
pub struct PopupTracker {
    turns: HashMap<String, TurnState>,
    in_flight: HashSet<String>,
    displayed: Option<String>,
    listening: bool,
    speaking: HashSet<String>,
}

impl PopupTracker {
    pub fn snapshot(&self, cfg: &TrayPopupConfig) -> PopupState {
        let displayed_id = self.displayed.as_deref();
        let displayed = displayed_id.and_then(|id| self.turns.get(id));
        let displayed_in_flight = displayed_id
            .map(|id| self.in_flight.contains(id))
            .unwrap_or(false);
        PopupState {
            body: displayed
                .map(|t| truncate_chars_from_end(&t.body, cfg.truncate_chars))
                .unwrap_or_default(),
            footer: displayed.and_then(|t| t.footer.clone()),
            activity: self.activity(displayed, displayed_in_flight),
            visible: false,
        }
    }

    pub fn is_busy(&self) -> bool {
        !self.in_flight.is_empty()
    }

    pub fn is_listening(&self) -> bool {
        self.listening
    }

    pub fn is_speaking(&self) -> bool {
        !self.speaking.is_empty()
    }

    pub fn set_disconnected(&mut self) {
        self.turns.clear();
        self.in_flight.clear();
        self.displayed = None;
        self.listening = false;
        self.speaking.clear();
    }

    fn activity(&self, displayed: Option<&TurnState>, in_flight: bool) -> PopupActivity {
        if let Some(turn) = displayed
            && in_flight
        {
            return match turn.activity.as_ref() {
                Some(TurnActivity::Streaming) => PopupActivity::Streaming,
                Some(TurnActivity::Thinking) => PopupActivity::Thinking,
                Some(TurnActivity::RunningTool(name)) => {
                    PopupActivity::RunningTool { name: name.clone() }
                }
                Some(TurnActivity::Finished) | None => PopupActivity::Thinking,
            };
        }
        if self.listening {
            return PopupActivity::Listening;
        }
        PopupActivity::Idle
    }

    pub fn ingest(&mut self, ev: &Event, cfg: &TrayPopupConfig) -> PopupState {
        match ev {
            Event::Delta { id, .. } => {
                // LastDelta carries the full coalesced body; appending
                // the raw Delta on top would double-count tokens.
                self.bring_turn_to_front(id);
                self.in_flight.insert(id.clone());
                self.turns.entry(id.clone()).or_default().activity = Some(TurnActivity::Streaming);
            }
            Event::LastDelta { id, text } => {
                self.bring_turn_to_front(id);
                self.in_flight.insert(id.clone());
                let turn = self.turns.entry(id.clone()).or_default();
                turn.body = text.clone();
                turn.activity = Some(TurnActivity::Streaming);
            }
            Event::ReasoningDelta { id, .. } => {
                self.bring_turn_to_front(id);
                self.in_flight.insert(id.clone());
                self.turns.entry(id.clone()).or_default().activity = Some(TurnActivity::Thinking);
            }
            Event::ToolCall { id, name, args } => {
                self.bring_turn_to_front(id);
                self.in_flight.insert(id.clone());
                let line = ToolCallLine {
                    name: name.clone(),
                    args_summary: summarize_args(args, FOOTER_ARGS_MAX_CHARS),
                };
                let turn = self.turns.entry(id.clone()).or_default();
                turn.footer = Some(line);
                turn.activity = Some(TurnActivity::RunningTool(name.clone()));
            }
            Event::ToolResult { id, .. } => {
                self.bring_turn_to_front(id);
                self.in_flight.insert(id.clone());
                self.turns.entry(id.clone()).or_default().activity = Some(TurnActivity::Thinking);
            }
            Event::Done { id } | Event::Error { id, .. } => {
                self.in_flight.remove(id);
                if self.displayed.as_deref() != Some(id.as_str()) {
                    self.turns.remove(id);
                } else if let Some(turn) = self.turns.get_mut(id) {
                    turn.activity = Some(TurnActivity::Finished);
                }
            }
            Event::ListenState { active, .. } => {
                self.listening = *active;
            }
            Event::SpeakingState { id, speaking } => {
                if *speaking {
                    self.speaking.insert(id.clone());
                } else {
                    self.speaking.remove(id);
                }
            }
            _ => {}
        }
        self.snapshot(cfg)
    }

    fn bring_turn_to_front(&mut self, id: &str) {
        if self.displayed.as_deref() != Some(id)
            && let Some(prev) = self.displayed.take()
            && !self.in_flight.contains(&prev)
        {
            self.turns.remove(&prev);
        }
        self.displayed = Some(id.to_string());
    }
}

const FOOTER_ARGS_MAX_CHARS: usize = 80;

/// Single-line, ≤`max_chars`-codepoint preview of a JSON value.
pub fn summarize_args(v: &Value, max_chars: usize) -> String {
    let raw = match v {
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        Value::Array(items) => {
            if items.is_empty() {
                "[]".to_string()
            } else {
                format!("[{} items]", items.len())
            }
        }
        Value::Object(map) => {
            if map.is_empty() {
                "{}".to_string()
            } else {
                map.iter()
                    .map(|(k, vv)| format!("{k}={}", value_inline(vv)))
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        }
    };
    truncate_chars_from_start(&flatten_whitespace(&raw), max_chars)
}

fn value_inline(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => {
            if s.chars().any(|c| c.is_whitespace() || c == '=') {
                format!("\"{s}\"")
            } else {
                s.clone()
            }
        }
        Value::Array(items) => format!("[{} items]", items.len()),
        Value::Object(map) => format!("{{{} keys}}", map.len()),
    }
}

fn flatten_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for ch in s.chars() {
        if ch.is_control() || ch.is_whitespace() {
            if !prev_space && !out.is_empty() {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    while out.ends_with(' ') {
        out.pop();
    }
    out
}

fn truncate_chars_from_end(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    let len = s.chars().count();
    if len <= max {
        return s.to_string();
    }
    let tail: String = s.chars().skip(len - max).collect();
    format!("…{tail}")
}

fn truncate_chars_from_start(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    if s.chars().count() <= max {
        return s.to_string();
    }
    let head: String = s.chars().take(max).collect();
    format!("{head}…")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cfg() -> TrayPopupConfig {
        TrayPopupConfig::default()
    }

    #[test]
    fn summarize_args_object_renders_key_equals_value() {
        let s = summarize_args(&json!({"a": 1, "b": "two"}), 100);
        assert!(s.contains("a=1"), "missing a=1: {s}");
        assert!(s.contains("b=two"), "missing b=two: {s}");
    }

    #[test]
    fn summarize_args_collapses_newlines_to_spaces() {
        let s = summarize_args(&json!({"command": "ls -la\n/tmp"}), 100);
        assert!(!s.contains('\n'));
        assert!(s.contains("ls -la /tmp"), "got: {s}");
    }

    #[test]
    fn summarize_args_quotes_strings_with_whitespace_or_equals() {
        let s = summarize_args(&json!({"q": "hello world"}), 100);
        assert!(s.contains("q=\"hello world\""), "got: {s}");
    }

    #[test]
    fn summarize_args_truncates_to_max_chars() {
        let long = "x".repeat(1000);
        let s = summarize_args(&json!({"a": long}), 30);
        assert_eq!(s.chars().count(), 31);
        assert!(s.ends_with('…'));
    }

    #[test]
    fn summarize_args_array_reports_count() {
        let s = summarize_args(&json!([1, 2, 3, 4, 5]), 100);
        assert_eq!(s, "[5 items]");
    }

    #[test]
    fn summarize_args_null_is_empty_string() {
        assert_eq!(summarize_args(&Value::Null, 100), "");
    }

    #[test]
    fn truncate_chars_from_end_keeps_last_n() {
        assert_eq!(truncate_chars_from_end("hello world", 5), "…world");
        assert_eq!(truncate_chars_from_end("short", 100), "short");
        assert_eq!(truncate_chars_from_end("", 5), "");
    }

    #[test]
    fn truncate_chars_from_end_preserves_unicode_codepoints() {
        let s = "🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀";
        let truncated = truncate_chars_from_end(s, 3);
        assert_eq!(truncated.chars().count(), 4);
    }

    fn delta(id: &str, text: &str) -> Event {
        Event::Delta {
            id: id.into(),
            text: text.into(),
        }
    }
    fn last_delta(id: &str, text: &str) -> Event {
        Event::LastDelta {
            id: id.into(),
            text: text.into(),
        }
    }
    fn tool_call(id: &str, name: &str, args: Value) -> Event {
        Event::ToolCall {
            id: id.into(),
            name: name.into(),
            args,
        }
    }
    fn done(id: &str) -> Event {
        Event::Done { id: id.into() }
    }

    #[test]
    fn tracker_body_comes_only_from_last_delta() {
        let mut t = PopupTracker::default();
        t.ingest(&delta("a", "hello "), &cfg());
        t.ingest(&delta("a", "world"), &cfg());
        assert_eq!(
            t.snapshot(&cfg()).body,
            "",
            "Delta must not populate the body"
        );
        t.ingest(&last_delta("a", "the coalesced reply"), &cfg());
        assert_eq!(t.snapshot(&cfg()).body, "the coalesced reply");
    }

    #[test]
    fn tracker_interleaved_last_delta_and_delta_does_not_double_count() {
        let mut t = PopupTracker::default();
        t.ingest(&last_delta("a", "A"), &cfg());
        t.ingest(&delta("a", "A"), &cfg());
        assert_eq!(t.snapshot(&cfg()).body, "A");
        t.ingest(&delta("a", "B"), &cfg());
        t.ingest(&delta("a", "C"), &cfg());
        assert_eq!(t.snapshot(&cfg()).body, "A", "raw deltas must not append");
        t.ingest(&last_delta("a", "ABCD"), &cfg());
        t.ingest(&delta("a", "D"), &cfg());
        assert_eq!(t.snapshot(&cfg()).body, "ABCD");
    }

    #[test]
    fn tracker_keeps_displayed_turn_after_done() {
        let mut t = PopupTracker::default();
        t.ingest(&last_delta("a", "the reply"), &cfg());
        t.ingest(&done("a"), &cfg());
        let s = t.snapshot(&cfg());
        assert_eq!(s.body, "the reply");
    }

    #[test]
    fn tracker_switches_to_new_turn_and_drops_old() {
        let mut t = PopupTracker::default();
        t.ingest(&last_delta("a", "turn a body"), &cfg());
        t.ingest(&done("a"), &cfg());
        t.ingest(&last_delta("b", "turn b body"), &cfg());
        let s = t.snapshot(&cfg());
        assert_eq!(s.body, "turn b body");
    }

    #[test]
    fn tracker_renders_tool_call_footer_with_args_summary() {
        let mut t = PopupTracker::default();
        t.ingest(
            &tool_call("a", "bash", json!({"command": "ls /tmp"})),
            &cfg(),
        );
        let s = t.snapshot(&cfg());
        let footer = s.footer.expect("footer present");
        assert_eq!(footer.name, "bash");
        assert!(footer.args_summary.contains("command="), "got: {footer:?}");
    }

    #[test]
    fn tracker_truncates_body_to_configured_n() {
        let mut t = PopupTracker::default();
        let long = "x".repeat(1000);
        t.ingest(&last_delta("a", &long), &cfg());
        let mut narrow = cfg();
        narrow.truncate_chars = 50;
        let s = t.snapshot(&narrow);
        assert_eq!(s.body.chars().count(), 51);
        assert!(s.body.starts_with('…'));
    }

    #[test]
    fn tracker_ignores_unrelated_event_kinds() {
        let mut t = PopupTracker::default();
        let before = t.snapshot(&cfg());
        t.ingest(
            &Event::Capabilities {
                id: "a".into(),
                vision: false,
                model_name: "test".into(),
            },
            &cfg(),
        );
        let after = t.snapshot(&cfg());
        assert_eq!(before, after);
    }

    #[test]
    fn tool_result_marks_displayed_turn_as_thinking() {
        let mut t = PopupTracker::default();
        t.ingest(
            &tool_call("a", "bash", json!({"command": "sleep 30"})),
            &cfg(),
        );
        assert!(matches!(
            t.snapshot(&cfg()).activity,
            PopupActivity::RunningTool { .. }
        ));
        t.ingest(
            &Event::ToolResult {
                id: "a".into(),
                name: "bash".into(),
                result: json!({"ok": true}),
            },
            &cfg(),
        );
        assert_eq!(t.snapshot(&cfg()).activity, PopupActivity::Thinking);
    }

    #[test]
    fn tracker_marks_busy_while_turn_in_flight() {
        let mut t = PopupTracker::default();
        assert!(!t.is_busy());
        t.ingest(&delta("a", "hi"), &cfg());
        assert!(t.is_busy());
        t.ingest(&done("a"), &cfg());
        assert!(!t.is_busy());
    }

    #[test]
    fn tracker_tracks_listen_state_and_surfaces_listening_activity() {
        let mut t = PopupTracker::default();
        assert!(!t.is_listening());
        assert_eq!(t.snapshot(&cfg()).activity, PopupActivity::Idle);

        t.ingest(
            &Event::ListenState {
                id: "x".into(),
                active: true,
            },
            &cfg(),
        );
        assert!(t.is_listening());
        assert_eq!(t.snapshot(&cfg()).activity, PopupActivity::Listening);

        t.ingest(&delta("a", "hi"), &cfg());
        assert_eq!(t.snapshot(&cfg()).activity, PopupActivity::Streaming);
        t.ingest(&done("a"), &cfg());
        assert_eq!(t.snapshot(&cfg()).activity, PopupActivity::Listening);

        t.ingest(
            &Event::ListenState {
                id: "x".into(),
                active: false,
            },
            &cfg(),
        );
        assert!(!t.is_listening());
        assert_eq!(t.snapshot(&cfg()).activity, PopupActivity::Idle);
    }

    #[test]
    fn tracker_tracks_speaking_state_per_turn() {
        let mut t = PopupTracker::default();
        assert!(!t.is_speaking());

        t.ingest(
            &Event::SpeakingState {
                id: "a".into(),
                speaking: true,
            },
            &cfg(),
        );
        assert!(t.is_speaking());

        t.ingest(
            &Event::SpeakingState {
                id: "b".into(),
                speaking: true,
            },
            &cfg(),
        );
        assert!(t.is_speaking());

        t.ingest(
            &Event::SpeakingState {
                id: "a".into(),
                speaking: false,
            },
            &cfg(),
        );
        assert!(t.is_speaking());

        t.ingest(
            &Event::SpeakingState {
                id: "b".into(),
                speaking: false,
            },
            &cfg(),
        );
        assert!(!t.is_speaking());
    }

    #[test]
    fn tracker_disconnect_clears_speaking_state() {
        let mut t = PopupTracker::default();
        t.ingest(
            &Event::SpeakingState {
                id: "a".into(),
                speaking: true,
            },
            &cfg(),
        );
        assert!(t.is_speaking());
        t.set_disconnected();
        assert!(!t.is_speaking());
    }

    #[test]
    fn tracker_disconnect_clears_turn_state() {
        let mut t = PopupTracker::default();
        t.ingest(&last_delta("a", "x"), &cfg());
        t.set_disconnected();
        let s = t.snapshot(&cfg());
        assert_eq!(s.body, "");
        assert!(s.footer.is_none());
    }
}
