//! Popup data model: rendered fields, per-turn coalescing, and the
//! small argument-summarizer that turns a `ToolCall`'s JSON args into a
//! one-line footer.
//!
//! The tracker is deliberately ignorant of the GUI layer; the daemon
//! event stream lands here, the rendered string fields land in the
//! shared [`PopupState`] snapshot, and the eframe app just reads the
//! latest snapshot on each repaint.

use std::collections::{HashMap, HashSet};

use assistd_config::TrayPopupConfig;
use assistd_ipc::Event;
use serde_json::Value;

/// Snapshot pushed through the `watch` channel to the GUI thread. The
/// GUI side reads this each repaint; tracker mutations on the tokio
/// side push a new snapshot only when something visible actually
/// changed.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PopupState {
    /// Last N codepoints of the in-flight (or last-completed) turn's
    /// reply text, after coalescing. May be empty when no turn has
    /// produced any output yet.
    pub body: String,
    /// Most recent tool invocation for the active turn, rendered as
    /// `<fully-qualified-tool-name>` + a one-line arg summary.
    pub footer: Option<ToolCallLine>,
    /// What the agent / daemon is currently doing, rendered as a small
    /// status indicator above the body. Driven by the most recent event
    /// for the displayed turn plus the daemon's listening flag.
    pub activity: PopupActivity,
    /// Whether the GUI should show the window right now. The visibility
    /// state machine owns this flag; the tracker leaves it untouched
    /// when ingesting events.
    pub visible: bool,
}

/// Coarse activity classification rendered as a one-line status above
/// the popup body. Distinct from raw event kind so the GUI can show a
/// stable label across an event burst (e.g. `Streaming` covers both
/// `Delta` and `LastDelta`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PopupActivity {
    /// No turn in flight and listener not active — popup is just
    /// surfacing the last completed turn.
    #[default]
    Idle,
    /// Agent is producing reply text right now.
    Streaming,
    /// Agent is between text chunks: reasoning, planning the next tool
    /// call, or waiting for the LLM to start the next round after a
    /// tool result landed.
    Thinking,
    /// Agent has dispatched a tool and is waiting for it to return.
    /// `name` is the fully-qualified tool name.
    RunningTool { name: String },
    /// Daemon's continuous listener is on and no turn is in flight; the
    /// user can speak without pressing a key.
    Listening,
}

/// Per-turn activity kind we last observed, used by the snapshot to
/// derive [`PopupActivity`] for the displayed turn.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TurnActivity {
    Streaming,
    Thinking,
    RunningTool(String),
    Finished,
}

/// Footer row rendered under the body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallLine {
    /// Fully-qualified tool name, exactly as it appeared on the wire
    /// (e.g. `mcp__filesystem__read_text_file` or a native name like
    /// `bash`). The IPC payload already prefixes MCP tools, so we
    /// render it verbatim.
    pub name: String,
    /// Compact one-line summary of the tool's arguments. Newlines and
    /// control characters are flattened to spaces and the result is
    /// truncated to a configured maximum.
    pub args_summary: String,
}

/// Per-turn working state. The popup may display content from a
/// single turn even after it terminates, so we retain `body`, `footer`
/// and `activity` until the *next* turn's first event lands.
#[derive(Debug, Clone, Default)]
struct TurnState {
    body: String,
    footer: Option<ToolCallLine>,
    activity: Option<TurnActivity>,
}

/// Tracks every signal the popup cares about and produces a
/// [`PopupState`] snapshot on demand. Owned by the popup driver task
/// (one task, single-threaded ingestion — no internal locking needed).
#[derive(Debug, Default)]
pub struct PopupTracker {
    /// Turns we've seen at least one event for. Keyed by IPC request
    /// id. Pruned on `Done` / `Error` only if it isn't the displayed
    /// turn — see [`PopupTracker::ingest`].
    turns: HashMap<String, TurnState>,
    /// Turn ids that haven't yet received a `Done` / `Error`. Used so
    /// `set_disconnected` can drop in-flight turns whose ids may no
    /// longer map to live turns on the daemon after a restart.
    in_flight: HashSet<String>,
    /// IPC id of the turn whose body/footer the popup is currently
    /// showing. Switches on the first event of a new turn so concurrent
    /// queries don't trample each other.
    displayed: Option<String>,
    /// Whether the daemon's continuous listener is on. Tracked from
    /// `ListenState` events so the visibility driver can extend the
    /// auto-hide window when the user can verbally reply without
    /// pressing a key.
    listening: bool,
}

impl PopupTracker {
    /// Build a `PopupState` from the current tracker. The truncation
    /// cap is applied here so the GUI never sees more body text than
    /// the user configured.
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

    /// Whether any turn is still in flight on the daemon side. The
    /// visibility driver consults this to keep the popup pinned open
    /// while the agent is mid-response — including the gap between a
    /// `ToolCall` and the corresponding `ToolResult`, where no events
    /// arrive but the agent is still busy.
    pub fn is_busy(&self) -> bool {
        !self.in_flight.is_empty()
    }

    /// Whether the daemon's continuous listener is currently active.
    /// Used by the driver to swap in the longer hands-free auto-hide
    /// window.
    pub fn is_listening(&self) -> bool {
        self.listening
    }

    /// Forget per-turn state — any in-flight turn id we remembered no
    /// longer maps to a live turn on the daemon side after a restart.
    pub fn set_disconnected(&mut self) {
        self.turns.clear();
        self.in_flight.clear();
        self.displayed = None;
        self.listening = false;
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
                // No per-turn signal yet but the turn is still live — the
                // model has accepted the request and is presumably
                // composing its first response.
                Some(TurnActivity::Finished) | None => PopupActivity::Thinking,
            };
        }
        if self.listening {
            return PopupActivity::Listening;
        }
        PopupActivity::Idle
    }

    /// Apply one daemon event. Returns the latest snapshot so the
    /// caller can decide whether to push it onto the watch.
    pub fn ingest(&mut self, ev: &Event, cfg: &TrayPopupConfig) -> PopupState {
        match ev {
            Event::Delta { id, text } => {
                self.bring_turn_to_front(id);
                self.in_flight.insert(id.clone());
                let turn = self.turns.entry(id.clone()).or_default();
                turn.body.push_str(text);
                turn.activity = Some(TurnActivity::Streaming);
            }
            Event::LastDelta { id, text } => {
                // Server-coalesced cumulative snapshot — overwrite, don't append.
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
                // Tool came back; the agent is now planning the next
                // step. Surface that as "thinking" until the next
                // Delta / ToolCall arrives.
                self.bring_turn_to_front(id);
                self.in_flight.insert(id.clone());
                self.turns.entry(id.clone()).or_default().activity = Some(TurnActivity::Thinking);
            }
            Event::Done { id } | Event::Error { id, .. } => {
                self.in_flight.remove(id);
                // Keep the displayed turn's content around so the popup
                // doesn't blank out when a turn finishes. Non-displayed
                // turns are dropped.
                if self.displayed.as_deref() != Some(id.as_str()) {
                    self.turns.remove(id);
                } else if let Some(turn) = self.turns.get_mut(id) {
                    turn.activity = Some(TurnActivity::Finished);
                }
            }
            Event::ListenState { active, .. } => {
                self.listening = *active;
            }
            _ => {}
        }
        self.snapshot(cfg)
    }

    /// Switch the displayed turn to `id` and drop any older
    /// already-terminated turn we were holding onto. If the previous
    /// displayed turn is still in flight, keep it in `turns` — it may
    /// matter again later (rare, but possible if the daemon
    /// interleaves two long-running turns).
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

/// Cap on the footer argument summary independently of the body
/// truncation. ~80 chars fits one wrapped line at the default popup
/// width without intruding on the body area.
const FOOTER_ARGS_MAX_CHARS: usize = 80;

/// Render a `serde_json::Value` as a single-line, ≤`max_chars`-codepoint
/// preview. Newlines and control characters become single spaces;
/// objects render as flat `key=value` pairs; arrays render as their
/// length only when long.
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
            // Keep simple strings unquoted; quote when they'd otherwise
            // ambiguate with the surrounding `k=v k=v` rendering.
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

/// Replace every whitespace / control character with a single space
/// and collapse repeated spaces. Used so a JSON `command` arg
/// containing `\n` doesn't break the popup's one-line footer.
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

/// Truncate to the last `max` codepoints, prepending an ellipsis when
/// the original was longer. Operates on chars (not bytes) so we never
/// split a multibyte codepoint.
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

/// Truncate to the first `max` codepoints, appending an ellipsis when
/// the original was longer. Used for argument summaries — the keys are
/// usually meaningful and the value tail rarely is.
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
        // serde_json preserves insertion order for objects, which
        // matches what the daemon sends.
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
        assert_eq!(s.chars().count(), 31); // 30 chars + 1 ellipsis
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
        // Each emoji is multiple bytes; chars-based truncation must
        // not split them.
        let s = "🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀";
        let truncated = truncate_chars_from_end(s, 3);
        // 3 emoji + 1 ellipsis
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
    fn tracker_accumulates_delta_text_until_last_delta_overwrites() {
        let mut t = PopupTracker::default();
        t.ingest(&delta("a", "hello "), &cfg());
        t.ingest(&delta("a", "world"), &cfg());
        let s = t.snapshot(&cfg());
        assert_eq!(s.body, "hello world");
        t.ingest(&last_delta("a", "completely fresh body"), &cfg());
        assert_eq!(t.snapshot(&cfg()).body, "completely fresh body");
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
        assert_eq!(s.body.chars().count(), 51); // 50 chars + 1 ellipsis
        assert!(s.body.starts_with('…'));
    }

    #[test]
    fn tracker_ignores_unrelated_event_kinds() {
        let mut t = PopupTracker::default();
        let before = t.snapshot(&cfg());
        // Pick an event kind the popup deliberately has no rendering
        // for (Capabilities reply); it should leave both content and
        // activity untouched.
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

        // While a turn is in flight, the per-turn activity takes
        // precedence over the listening flag.
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
    fn tracker_disconnect_clears_turn_state() {
        let mut t = PopupTracker::default();
        t.ingest(&last_delta("a", "x"), &cfg());
        t.set_disconnected();
        let s = t.snapshot(&cfg());
        assert_eq!(s.body, "");
        assert!(s.footer.is_none());
    }
}
