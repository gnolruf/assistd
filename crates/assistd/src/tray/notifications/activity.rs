//! Notification data model and per-turn coalescing.

use std::collections::{HashMap, HashSet, VecDeque};

use assistd_ipc::Event;
use serde_json::Value;

const TOOL_ARGS_MAX_CHARS: usize = 80;
/// Tool calls kept per turn, oldest dropped first.
const MAX_TOOL_CALLS: usize = 3;
/// Only the last `BODY_CHARS` codepoints of the reply are shown;
/// notification daemons cut longer text in their own ways.
const BODY_CHARS: usize = 300;

/// Everything a notification shows, as one snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct ActivityView {
    pub body: String,
    /// Oldest first.
    pub tool_calls: Vec<ToolCallLine>,
    pub error: Option<String>,
    /// The latest session title the daemon broadcast.
    pub title: Option<String>,
    pub activity: Activity,
}

/// What the displayed turn, or the daemon, is doing.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) enum Activity {
    #[default]
    Idle,
    Streaming,
    Thinking,
    RunningTool {
        name: String,
    },
    Listening,
    Done,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TurnActivity {
    Streaming,
    Thinking,
    RunningTool(String),
    Finished,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ToolCallLine {
    pub name: String,
    pub args_summary: String,
}

#[derive(Debug, Clone, Default)]
struct TurnState {
    body: String,
    tool_calls: VecDeque<ToolCallLine>,
    error: Option<String>,
    activity: Option<TurnActivity>,
}

impl TurnState {
    /// Reply text after tool calls replaces them and the narration before them.
    fn start_streaming(&mut self) {
        if !self.tool_calls.is_empty() && self.activity != Some(TurnActivity::Streaming) {
            self.tool_calls.clear();
            self.body.clear();
        }
        self.activity = Some(TurnActivity::Streaming);
    }
}

/// Tracks every signal the notifications care about and produces an
/// [`ActivityView`] snapshot on demand.
#[derive(Debug, Default)]
pub(super) struct ActivityTracker {
    turns: HashMap<String, TurnState>,
    in_flight: HashSet<String>,
    displayed: Option<String>,
    session_title: Option<String>,
    listening: bool,
    speaking: HashSet<String>,
}

impl ActivityTracker {
    pub(super) fn snapshot(&self) -> ActivityView {
        let displayed_id = self.displayed.as_deref();
        let displayed = displayed_id.and_then(|id| self.turns.get(id));
        let displayed_in_flight = displayed_id.is_some_and(|id| self.in_flight.contains(id));
        ActivityView {
            body: displayed
                .map(|t| truncate_chars_from_end(&t.body, BODY_CHARS))
                .unwrap_or_default(),
            tool_calls: displayed
                .map(|t| t.tool_calls.iter().cloned().collect())
                .unwrap_or_default(),
            error: displayed.and_then(|t| t.error.clone()),
            title: self.session_title.clone(),
            activity: self.activity(displayed, displayed_in_flight),
        }
    }

    /// The turn the snapshot shows.
    pub(super) fn current_turn(&self) -> Option<&str> {
        self.displayed.as_deref()
    }

    pub(super) fn is_busy(&self) -> bool {
        !self.in_flight.is_empty()
    }

    pub(super) fn is_listening(&self) -> bool {
        self.listening
    }

    pub(super) fn is_speaking(&self) -> bool {
        !self.speaking.is_empty()
    }

    pub(super) fn set_disconnected(&mut self) {
        self.turns.clear();
        self.in_flight.clear();
        self.displayed = None;
        self.session_title = None;
        self.listening = false;
        self.speaking.clear();
    }

    fn activity(&self, displayed: Option<&TurnState>, in_flight: bool) -> Activity {
        let turn_activity = displayed.and_then(|turn| turn.activity.as_ref());
        if in_flight {
            return match turn_activity {
                Some(TurnActivity::Streaming) => Activity::Streaming,
                Some(TurnActivity::RunningTool(name)) => {
                    Activity::RunningTool { name: name.clone() }
                }
                Some(TurnActivity::Thinking | TurnActivity::Finished | TurnActivity::Failed)
                | None => Activity::Thinking,
            };
        }
        if self.listening {
            return Activity::Listening;
        }
        match turn_activity {
            Some(TurnActivity::Failed) => Activity::Failed,
            Some(_) => Activity::Done,
            None => Activity::Idle,
        }
    }

    /// Apply `ev`. The body comes only from `LastDelta`, which carries
    /// the whole reply; `Delta` never appends.
    pub(super) fn ingest(&mut self, ev: &Event) {
        match ev {
            Event::Delta { id, .. } => self.activate_turn(id).start_streaming(),
            Event::LastDelta { id, text } => {
                let turn = self.activate_turn(id);
                turn.start_streaming();
                turn.body.clone_from(text);
            }
            Event::ReasoningDelta { id, .. } => {
                self.activate_turn(id).activity = Some(TurnActivity::Thinking);
            }
            Event::ToolCall { id, name, args } => {
                let turn = self.activate_turn(id);
                if turn.tool_calls.len() == MAX_TOOL_CALLS {
                    turn.tool_calls.pop_front();
                }
                turn.tool_calls.push_back(ToolCallLine {
                    name: name.clone(),
                    args_summary: summarize_args(args, TOOL_ARGS_MAX_CHARS),
                });
                turn.activity = Some(TurnActivity::RunningTool(name.clone()));
            }
            Event::ToolResult { id, .. } => {
                self.activate_turn(id).activity = Some(TurnActivity::Thinking);
            }
            Event::Done { id } => self.finish_turn(id, TurnActivity::Finished, None),
            Event::Error { id, message } => {
                self.finish_turn(id, TurnActivity::Failed, Some(message.clone()));
            }
            Event::SessionTitle { title, .. } => {
                self.session_title = Some(title.clone());
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
    }

    fn finish_turn(&mut self, id: &str, outcome: TurnActivity, error: Option<String>) {
        self.in_flight.remove(id);
        if self.displayed.as_deref() != Some(id) {
            self.turns.remove(id);
        } else if let Some(turn) = self.turns.get_mut(id) {
            turn.activity = Some(outcome);
            turn.error = error;
        }
    }

    fn activate_turn(&mut self, id: &str) -> &mut TurnState {
        self.bring_turn_to_front(id);
        self.in_flight.insert(id.to_string());
        self.turns.entry(id.to_string()).or_default()
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

/// Single-line, ≤`max_chars`-codepoint preview of a JSON value.
pub(super) fn summarize_args(v: &Value, max_chars: usize) -> String {
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
                    .map(|(key, value)| format!("{key}={}", value_inline(value)))
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
    use serde_json::json;

    use super::*;

    #[test]
    fn summarize_args_renders_a_single_line_preview() {
        for (args, expected) in [
            (json!({"a": 1, "b": "two"}), "a=1 b=two"),
            (
                json!({"command": "ls -la\n/tmp"}),
                "command=\"ls -la /tmp\"",
            ),
            (json!({"q": "hello world"}), "q=\"hello world\""),
            (json!({"k": "a=b"}), "k=\"a=b\""),
            (json!([1, 2, 3, 4, 5]), "[5 items]"),
            (Value::Null, ""),
        ] {
            assert_eq!(summarize_args(&args, 100), expected, "{args}");
        }
    }

    #[test]
    fn summarize_args_truncates_to_max_chars() {
        let s = summarize_args(&json!({"a": "x".repeat(1000)}), 30);
        assert_eq!(s, format!("a={}…", "x".repeat(28)));
    }

    #[test]
    fn truncate_chars_from_end_keeps_last_n_codepoints() {
        for (input, max, expected) in [
            ("hello world", 5, "…world"),
            ("short", 100, "short"),
            ("", 5, ""),
            ("🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀", 3, "…🦀🦀🦀"),
        ] {
            assert_eq!(truncate_chars_from_end(input, max), expected, "{input:?}");
        }
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
        let mut t = ActivityTracker::default();
        t.ingest(&delta("a", "hello"));
        assert_eq!(t.snapshot().body, "", "Delta must not populate the body");
        t.ingest(&last_delta("a", "A"));
        t.ingest(&delta("a", "A"));
        assert_eq!(t.snapshot().body, "A");
        t.ingest(&delta("a", "B"));
        t.ingest(&delta("a", "C"));
        assert_eq!(t.snapshot().body, "A", "raw deltas must not append");
        t.ingest(&last_delta("a", "ABCD"));
        t.ingest(&delta("a", "D"));
        assert_eq!(t.snapshot().body, "ABCD");
    }

    #[test]
    fn tracker_keeps_displayed_turn_after_done() {
        let mut t = ActivityTracker::default();
        t.ingest(&last_delta("a", "the reply"));
        t.ingest(&done("a"));
        let s = t.snapshot();
        assert_eq!(s.body, "the reply");
    }

    #[test]
    fn tracker_switches_to_new_turn_and_drops_old() {
        let mut t = ActivityTracker::default();
        t.ingest(&last_delta("a", "turn a body"));
        t.ingest(&done("a"));
        t.ingest(&last_delta("b", "turn b body"));
        assert_eq!(t.snapshot().body, "turn b body");
        assert!(!t.turns.contains_key("a"));
    }

    #[test]
    fn tracker_keeps_the_last_tool_calls_with_args_summaries() {
        let mut t = ActivityTracker::default();
        for n in 0..4 {
            t.ingest(&tool_call(
                "a",
                "bash",
                json!({"command": format!("ls /tmp/{n}")}),
            ));
        }
        let s = t.snapshot();
        let summaries: Vec<_> = s
            .tool_calls
            .iter()
            .map(|c| c.args_summary.as_str())
            .collect();
        assert_eq!(
            summaries,
            [
                "command=\"ls /tmp/1\"",
                "command=\"ls /tmp/2\"",
                "command=\"ls /tmp/3\""
            ]
        );
        assert_eq!(
            s.activity,
            Activity::RunningTool {
                name: "bash".into()
            }
        );
    }

    #[test]
    fn reply_text_after_tool_calls_replaces_them() {
        let mut t = ActivityTracker::default();
        t.ingest(&last_delta("a", "let me look"));
        t.ingest(&tool_call("a", "bash", json!({"command": "ls"})));
        t.ingest(&Event::ToolResult {
            id: "a".into(),
            name: "bash".into(),
            result: json!({}),
        });
        let before = t.snapshot();
        assert_eq!(before.body, "let me look");
        assert_eq!(before.tool_calls.len(), 1);

        t.ingest(&delta("a", "Found"));
        let s = t.snapshot();
        assert!(s.tool_calls.is_empty());
        assert_eq!(s.body, "");
        t.ingest(&last_delta("a", "Found it"));
        t.ingest(&delta("a", " it"));
        assert_eq!(t.snapshot().body, "Found it");
    }

    #[test]
    fn session_title_outlives_the_turn_it_arrived_with() {
        let mut t = ActivityTracker::default();
        t.ingest(&last_delta("a", "reply"));
        assert_eq!(t.snapshot().title, None);
        t.ingest(&done("a"));
        t.ingest(&Event::SessionTitle {
            id: "a".into(),
            session_id: "s".into(),
            title: "Cats And Dogs".into(),
        });
        t.ingest(&delta("b", "x"));
        t.ingest(&tool_call("b", "bash", json!({})));
        assert_eq!(t.snapshot().title.as_deref(), Some("Cats And Dogs"));
        t.set_disconnected();
        assert_eq!(t.snapshot().title, None);
    }

    #[test]
    fn finished_turn_reports_done_or_failed_with_its_error() {
        let mut t = ActivityTracker::default();
        t.ingest(&last_delta("a", "reply"));
        t.ingest(&done("a"));
        assert_eq!(t.snapshot().activity, Activity::Done);

        t.ingest(&delta("b", "x"));
        t.ingest(&Event::Error {
            id: "b".into(),
            message: "boom".into(),
        });
        let s = t.snapshot();
        assert_eq!(s.activity, Activity::Failed);
        assert_eq!(s.error.as_deref(), Some("boom"));
        assert_eq!(t.current_turn(), Some("b"));
    }

    #[test]
    fn tracker_truncates_body_to_the_last_body_chars() {
        let mut t = ActivityTracker::default();
        let long = "a".repeat(BODY_CHARS) + &"b".repeat(BODY_CHARS);
        t.ingest(&last_delta("a", &long));
        assert_eq!(t.snapshot().body, format!("…{}", "b".repeat(BODY_CHARS)));
    }

    #[test]
    fn tracker_ignores_unrelated_event_kinds() {
        let mut t = ActivityTracker::default();
        let before = t.snapshot();
        t.ingest(&Event::Capabilities {
            id: "a".into(),
            vision: false,
            model_name: "test".into(),
        });
        let after = t.snapshot();
        assert_eq!(before, after);
    }

    #[test]
    fn tool_result_marks_displayed_turn_as_thinking() {
        let mut t = ActivityTracker::default();
        t.ingest(&tool_call("a", "bash", json!({"command": "sleep 30"})));
        t.ingest(&Event::ToolResult {
            id: "a".into(),
            name: "bash".into(),
            result: json!({"ok": true}),
        });
        assert_eq!(t.snapshot().activity, Activity::Thinking);
    }

    #[test]
    fn tracker_marks_busy_while_turn_in_flight() {
        let mut t = ActivityTracker::default();
        assert!(!t.is_busy());
        t.ingest(&delta("a", "hi"));
        assert!(t.is_busy());
        t.ingest(&done("a"));
        assert!(!t.is_busy());
    }

    #[test]
    fn tracker_tracks_listen_state_and_surfaces_listening_activity() {
        let mut t = ActivityTracker::default();
        assert!(!t.is_listening());
        assert_eq!(t.snapshot().activity, Activity::Idle);

        t.ingest(&Event::ListenState {
            id: "x".into(),
            active: true,
        });
        assert!(t.is_listening());
        assert_eq!(t.snapshot().activity, Activity::Listening);

        t.ingest(&delta("a", "hi"));
        assert_eq!(t.snapshot().activity, Activity::Streaming);
        t.ingest(&done("a"));
        assert_eq!(t.snapshot().activity, Activity::Listening);

        t.ingest(&Event::ListenState {
            id: "x".into(),
            active: false,
        });
        assert!(!t.is_listening());
        assert_eq!(t.snapshot().activity, Activity::Done);
    }

    #[test]
    fn tracker_tracks_speaking_state_per_turn() {
        let mut t = ActivityTracker::default();
        assert!(!t.is_speaking());

        t.ingest(&Event::SpeakingState {
            id: "a".into(),
            speaking: true,
        });
        assert!(t.is_speaking());

        t.ingest(&Event::SpeakingState {
            id: "b".into(),
            speaking: true,
        });
        assert!(t.is_speaking());

        t.ingest(&Event::SpeakingState {
            id: "a".into(),
            speaking: false,
        });
        assert!(t.is_speaking());

        t.ingest(&Event::SpeakingState {
            id: "b".into(),
            speaking: false,
        });
        assert!(!t.is_speaking());
    }

    #[test]
    fn tracker_disconnect_clears_all_state() {
        let mut t = ActivityTracker::default();
        t.ingest(&last_delta("a", "x"));
        t.ingest(&tool_call("a", "bash", json!({"command": "ls"})));
        t.ingest(&Event::ListenState {
            id: "x".into(),
            active: true,
        });
        t.ingest(&Event::SpeakingState {
            id: "a".into(),
            speaking: true,
        });
        t.set_disconnected();
        assert_eq!(t.snapshot(), ActivityView::default());
        assert!(!t.is_busy());
        assert!(!t.is_listening());
        assert!(!t.is_speaking());
    }
}
