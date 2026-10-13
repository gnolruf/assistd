//! Compact record of tool steps folded out of the current turn: each call
//! and the exit status its result reported, without the output.

use super::{Conversation, Message, ToolCallRecord, truncate_utf8};

/// Most folded calls a turn keeps; older entries fall off the front.
pub(super) const MAX_LOGGED_CALLS: usize = 24;
const LOGGED_CALL_BYTES: usize = 200;
const STEP_LOG_HEADER: &str =
    "[Tool calls already made for this request; outputs removed to save context]";
const RECORD_OVERHEAD_BYTES: usize = 24;

/// One tool call folded out of its turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepRecord {
    /// Tool name and arguments, cut to a bounded length.
    pub call: String,
    /// `None` when the result carried no `[exit:N | …]` footer.
    pub exit_code: Option<i32>,
}

/// Records for every tool call in `steps`, paired with its result.
pub(super) fn records(steps: &[Message]) -> Vec<StepRecord> {
    let mut out = Vec::new();
    let mut rest = steps;
    while let Some((step, tail)) = rest.split_first() {
        let result_count = tail
            .iter()
            .take_while(|m| Conversation::is_tool_result(m))
            .count();
        let (results, after) = tail.split_at(result_count);
        out.extend(
            step.tool_calls
                .iter()
                .enumerate()
                .map(|(position, call)| record(call, result_for(call, position, results))),
        );
        rest = after;
    }
    out
}

/// The log as rendered after the question's text.
pub(super) fn render(records: &[StepRecord]) -> String {
    let mut out = String::from(STEP_LOG_HEADER);
    for record in records {
        out.push_str("\n- ");
        out.push_str(&record.call);
        match record.exit_code {
            Some(0) => out.push_str(" → ok"),
            Some(code) => out.push_str(&format!(" → exit {code}")),
            None => out.push_str(" → no exit status"),
        }
    }
    out
}

/// Rendered size in bytes, without rendering.
pub(super) fn rendered_len(records: &[StepRecord]) -> usize {
    if records.is_empty() {
        return 0;
    }
    STEP_LOG_HEADER.len()
        + records
            .iter()
            .map(|r| r.call.len() + RECORD_OVERHEAD_BYTES)
            .sum::<usize>()
}

fn record(call: &ToolCallRecord, result: Option<&Message>) -> StepRecord {
    StepRecord {
        call: truncate_utf8(
            &format!("{} {}", call.name, call.arguments),
            LOGGED_CALL_BYTES,
        ),
        exit_code: result.and_then(|r| exit_code(&r.content)),
    }
}

fn result_for<'a>(
    call: &ToolCallRecord,
    position: usize,
    results: &'a [Message],
) -> Option<&'a Message> {
    results
        .iter()
        .find(|r| r.tool_call_id.as_deref() == Some(call.id.as_str()))
        .or_else(|| results.get(position))
}

fn exit_code(output: &str) -> Option<i32> {
    let footer = output.trim_end().lines().last()?;
    let (_, after) = footer.rsplit_once("[exit:")?;
    let (code, _) = after.split_once(" |")?;
    code.parse().ok()
}
