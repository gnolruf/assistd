//! Compact record of tool steps folded out of the current turn: each call,
//! the exit status its result reported, and the first error line of a
//! failure, without the rest of the output.

use super::{Conversation, Message, ToolCallRecord, truncate_utf8};

/// Most folded calls a turn keeps; older entries fall off the front.
const MAX_LOGGED_CALLS: usize = 24;
const LOGGED_CALL_BYTES: usize = 200;
const LOGGED_FAILURE_BYTES: usize = 160;
const STDERR_MARKER: &str = "[stderr] ";
const ERROR_MARKER: &str = "[error]";
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
    /// First error line of a failed result, cut to a bounded length.
    pub failure: Option<String>,
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

/// Add `records` to the end of `log`, dropping its oldest entries past
/// [`MAX_LOGGED_CALLS`].
pub(super) fn append(log: &mut Vec<StepRecord>, records: Vec<StepRecord>) {
    log.extend(records);
    let excess = log.len().saturating_sub(MAX_LOGGED_CALLS);
    log.drain(..excess);
}

/// The log as rendered after the question's text.
pub(super) fn render(records: &[StepRecord]) -> String {
    let mut out = String::from(STEP_LOG_HEADER);
    for record in records {
        out.push_str("\n- ");
        out.push_str(&record.call);
        match (record.exit_code, &record.failure) {
            (Some(code), Some(failure)) if code != 0 => {
                out.push_str(&format!(" → exit {code}: {failure}"));
            }
            (_, Some(failure)) => out.push_str(&format!(" → error: {failure}")),
            (Some(0), None) => out.push_str(" → ok"),
            (Some(code), None) => out.push_str(&format!(" → exit {code}")),
            (None, None) => out.push_str(" → no exit status"),
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
            .map(|r| {
                r.call.len() + r.failure.as_ref().map_or(0, String::len) + RECORD_OVERHEAD_BYTES
            })
            .sum::<usize>()
}

fn record(call: &ToolCallRecord, result: Option<&Message>) -> StepRecord {
    let exit_code = result.and_then(|r| exit_code(&r.content));
    StepRecord {
        call: truncate_utf8(
            &format!("{} {}", call.name, call.arguments),
            LOGGED_CALL_BYTES,
        ),
        exit_code,
        failure: result.and_then(|r| failure_line(&r.content, exit_code)),
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

/// The first `[error]` line of `output`, else, when `exit_code` reports a
/// failure, its first stderr line or first line of output.
fn failure_line(output: &str, exit_code: Option<i32>) -> Option<String> {
    let line = match output.lines().find(|l| l.contains(ERROR_MARKER)) {
        Some(error) => error,
        None if exit_code.is_some_and(|code| code != 0) => first_failure_detail(output)?,
        None => return None,
    };
    let cleaned = line.trim_start_matches(STDERR_MARKER).replace('\t', " ");
    Some(truncate_utf8(cleaned.trim(), LOGGED_FAILURE_BYTES))
}

fn first_failure_detail(output: &str) -> Option<&str> {
    output
        .lines()
        .find(|l| l.starts_with(STDERR_MARKER))
        .or_else(|| {
            output
                .lines()
                .find(|l| !l.trim().is_empty() && !l.starts_with("[exit:"))
        })
}

fn exit_code(output: &str) -> Option<i32> {
    let footer = output.trim_end().lines().last()?;
    let (_, after) = footer.rsplit_once("[exit:")?;
    let (code, _) = after.split_once(" |")?;
    code.parse().ok()
}
