//! Turns an [`ActivityView`] into the text, actions and hints of one
//! freedesktop notification.

use super::activity::{Activity, ActivityView};

/// The action key notification daemons run on a click of the body.
pub(super) const HIDE_ACTION: &str = "default";
const HIDE_LABEL: &str = "Hide";

/// What the notification service supports, from `GetCapabilities`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct Capabilities {
    /// `body-markup`: `<b>` and friends render instead of showing.
    pub markup: bool,
    pub actions: bool,
}

/// A notification to show or update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Notification {
    pub view: ActivityView,
    /// Offer the Hide action; set while a turn is in flight.
    pub hideable: bool,
    /// Milliseconds before the service closes it; `0` never does.
    pub expire_ms: u64,
}

/// [`Notification`] in the shape `Notify` takes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Rendered {
    pub icon: &'static str,
    pub summary: String,
    pub body: String,
    /// Alternating action keys and labels.
    pub actions: Vec<&'static str>,
    /// freedesktop urgency: 1 normal, 2 critical.
    pub urgency: u8,
    pub expire_timeout: i32,
}

pub(super) fn render(notification: &Notification, caps: Capabilities) -> Rendered {
    let view = &notification.view;
    Rendered {
        icon: icon_for(&view.activity),
        summary: summary_for(view),
        body: body_for(view, caps.markup),
        actions: if notification.hideable && caps.actions {
            vec![HIDE_ACTION, HIDE_LABEL]
        } else {
            Vec::new()
        },
        urgency: if view.activity == Activity::Failed {
            2
        } else {
            1
        },
        expire_timeout: i32::try_from(notification.expire_ms).unwrap_or(i32::MAX),
    }
}

fn icon_for(activity: &Activity) -> &'static str {
    match activity {
        Activity::Failed => "dialog-error",
        Activity::Listening => "audio-input-microphone",
        _ => "system-run",
    }
}

/// The session title once one arrives, `assistd` until then.
fn summary_for(view: &ActivityView) -> String {
    view.title.clone().unwrap_or_else(|| "assistd".into())
}

/// The turn's tool calls, reply and error, else a line naming the activity.
fn body_for(view: &ActivityView, markup: bool) -> String {
    let text = |raw: &str| {
        if markup {
            escape_markup(raw)
        } else {
            raw.to_string()
        }
    };
    let tool_lines = view.tool_calls.iter().map(|call| {
        let name = if markup {
            format!("<b>{}</b>", escape_markup(&call.name))
        } else {
            call.name.clone()
        };
        format!("{name} {}", text(&call.args_summary))
    });
    let reply = (!view.body.is_empty()).then(|| text(&view.body));
    let error = view.error.as_deref().map(text);
    let lines: Vec<_> = tool_lines.chain(reply).chain(error).collect();
    if lines.is_empty() {
        status_line_for(&view.activity).into()
    } else {
        lines.join("\n")
    }
}

fn status_line_for(activity: &Activity) -> &'static str {
    match activity {
        Activity::Thinking | Activity::Streaming | Activity::RunningTool { .. } => "Thinking…",
        Activity::Listening => "Listening…",
        Activity::Idle | Activity::Done | Activity::Failed => "",
    }
}

/// Escape the three characters the notification markup subset parses.
fn escape_markup(raw: &str) -> String {
    raw.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::super::activity::ToolCallLine;
    use super::*;

    const MARKUP: Capabilities = Capabilities {
        markup: true,
        actions: true,
    };
    const PLAIN: Capabilities = Capabilities {
        markup: false,
        actions: false,
    };

    fn busy(view: ActivityView) -> Notification {
        Notification {
            view,
            hideable: true,
            expire_ms: 0,
        }
    }

    fn tool(name: &str, args: &str) -> ToolCallLine {
        ToolCallLine {
            name: name.into(),
            args_summary: args.into(),
        }
    }

    #[test]
    fn summary_is_the_session_title_whatever_the_activity() {
        for activity in [
            Activity::Idle,
            Activity::Streaming,
            Activity::Thinking,
            Activity::RunningTool {
                name: "bash".into(),
            },
            Activity::Listening,
            Activity::Done,
            Activity::Failed,
        ] {
            let untitled = ActivityView {
                activity: activity.clone(),
                ..ActivityView::default()
            };
            assert_eq!(render(&busy(untitled), PLAIN).summary, "assistd");
            let titled = ActivityView {
                title: Some("Cats And Dogs".into()),
                activity,
                ..ActivityView::default()
            };
            assert_eq!(render(&busy(titled), PLAIN).summary, "Cats And Dogs");
        }
    }

    #[test]
    fn empty_body_names_the_activity() {
        for (activity, expected) in [
            (Activity::Thinking, "Thinking…"),
            (Activity::Streaming, "Thinking…"),
            (Activity::Listening, "Listening…"),
            (Activity::Done, ""),
        ] {
            let view = ActivityView {
                activity,
                ..ActivityView::default()
            };
            assert_eq!(render(&busy(view), PLAIN).body, expected);
        }
        let replying = ActivityView {
            body: "hi".into(),
            activity: Activity::Streaming,
            ..ActivityView::default()
        };
        assert_eq!(render(&busy(replying), PLAIN).body, "hi");
    }

    #[test]
    fn markup_body_bolds_tool_names_and_escapes_model_text() {
        let view = ActivityView {
            body: "a < b && c > d".into(),
            tool_calls: vec![tool("bash", "command=\"ls <dir>\"")],
            ..ActivityView::default()
        };
        assert_eq!(
            render(&busy(view), MARKUP).body,
            "<b>bash</b> command=\"ls &lt;dir&gt;\"\na &lt; b &amp;&amp; c &gt; d"
        );
    }

    #[test]
    fn plain_body_sends_text_unescaped() {
        let view = ActivityView {
            body: "a < b".into(),
            tool_calls: vec![tool("bash", "ls")],
            error: Some("boom".into()),
            activity: Activity::Failed,
            ..ActivityView::default()
        };
        let rendered = render(&busy(view), PLAIN);
        assert_eq!(rendered.body, "bash ls\na < b\nboom");
        assert_eq!(rendered.urgency, 2);
    }

    #[test]
    fn hide_action_needs_a_busy_turn_and_service_support() {
        for (hideable, caps, expected) in [
            (true, MARKUP, vec![HIDE_ACTION, HIDE_LABEL]),
            (false, MARKUP, vec![]),
            (true, PLAIN, vec![]),
        ] {
            let notification = Notification {
                hideable,
                ..busy(ActivityView::default())
            };
            assert_eq!(
                render(&notification, caps).actions,
                expected,
                "{hideable} {caps:?}"
            );
        }
    }

    #[test]
    fn expire_timeout_saturates() {
        let notification = Notification {
            expire_ms: u64::MAX,
            ..busy(ActivityView::default())
        };
        assert_eq!(render(&notification, PLAIN).expire_timeout, i32::MAX);
    }
}
