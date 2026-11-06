//! The status line's task segments: runs waiting on the user's answer, and runs running.

use assistd_ipc::{QueueItem, TaskQueueSnapshot, TaskRunState};

use crate::client::terminal_text::escape_controls_single_line;

/// One task segment's text, and whether it asks the user to answer a paused run.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct QueueSegment {
    pub text: String,
    pub needs_input: bool,
}

/// The needs-input segment, then the running segment, each left out when no run is in that
/// state.
pub(super) fn queue_segments(snapshot: &TaskQueueSnapshot) -> Vec<QueueSegment> {
    let needs_input = needs_input_segment(runs_in(snapshot, &TaskRunState::NeedsInput).count());
    let running: Vec<String> = runs_in(snapshot, &TaskRunState::Running)
        .map(|name| escape_controls_single_line(name).into_owned())
        .collect();
    let running = (!running.is_empty()).then(|| QueueSegment {
        text: match running.as_slice() {
            [name] => format!("task: {name}"),
            names => format!("tasks: {}", names.join(", ")),
        },
        needs_input: false,
    });
    needs_input.into_iter().chain(running).collect()
}

/// The task names of the runs in `state`.
fn runs_in<'a>(
    snapshot: &'a TaskQueueSnapshot,
    state: &'a TaskRunState,
) -> impl Iterator<Item = &'a str> {
    snapshot.items.iter().filter_map(move |item| match item {
        QueueItem::Run {
            task_name,
            state: run_state,
            ..
        } if run_state == state => Some(task_name.as_str()),
        _ => None,
    })
}

fn needs_input_segment(count: usize) -> Option<QueueSegment> {
    let text = match count {
        0 => return None,
        1 => "1 task needs input · ctrl+t".to_string(),
        _ => format!("{count} tasks need input · ctrl+t"),
    };
    Some(QueueSegment {
        text,
        needs_input: true,
    })
}
