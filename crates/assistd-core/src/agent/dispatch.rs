//! Tool dispatch: each call a step requests run under its deadline, with a call that
//! hangs, panics, names no tool or is never run answered as a failed result.

use std::any::Any;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::pin;
use std::time::{Duration, Instant};

use futures_util::FutureExt;
use serde_json::Value;
use tracing::{debug, info, warn};

use assistd_llm::{LlmError, LlmEvent, ToolCall, ToolResultPayload};
use assistd_tools::{ToolError, ToolOutput, ToolRegistry, TurnContext};
use assistd_utils::text::panic_message;

use super::Agent;
use super::run::Turn;

/// Leads what the user said when declining a prompt, in the declined call's result.
const DECLINED_WITH_MESSAGE: &str = "The user declined and said: ";

impl Agent {
    /// Announce the step's calls and dispatch each in order, answering any left unrun after
    /// the client leaves or cancellation with a cancelled payload so the batch stays
    /// complete. Returns `true` when a call hit the duplicate-call limit.
    pub(super) async fn dispatch_tool_calls(
        &self,
        turn: &mut Turn,
        calls: Vec<ToolCall>,
    ) -> (Vec<ToolResultPayload>, bool) {
        let mut results = Vec::with_capacity(calls.len());
        let mut stuck = false;
        turn.emit(LlmEvent::ToolCallsRequested {
            calls: calls.clone(),
        })
        .await;
        let mut pending = calls.into_iter();
        for call in pending.by_ref() {
            stuck |= turn.streak.record(&call) >= self.definition.limits.duplicate_calls;
            if turn.stop_requested() {
                debug!(
                    target: "assistd::agent",
                    iteration = turn.iteration,
                    tool = %call.name,
                    cancelled = turn.cancel.is_cancelled(),
                    "stopping mid-call (client gone or explicit cancel) without dispatch"
                );
                results.push(skip_tool_call(turn, &call).await);
                break;
            }

            let (payload, cancelled_mid_dispatch) = self.run_tool_call(turn, &call).await;
            results.push(payload);
            if cancelled_mid_dispatch {
                break;
            }
        }
        for call in pending {
            results.push(skip_tool_call(turn, &call).await);
        }
        (results, stuck)
    }

    /// Announce and run one call, or abandon it when the turn is cancelled
    /// meanwhile. The flag is `true` when cancellation fired.
    async fn run_tool_call(&self, turn: &Turn, call: &ToolCall) -> (ToolResultPayload, bool) {
        turn.emit(LlmEvent::ToolCall {
            id: call.id.clone(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
        })
        .await;

        let dispatched = tokio::select! {
            biased;
            () = turn.cancel.cancelled() => None,
            r = dispatch_tool_call(turn, call, self.host.tool_deadline) => Some(r),
        };
        let (result, read_host) = dispatched.unwrap_or_else(|| {
            warn!(
                target: "assistd::agent",
                iteration = turn.iteration,
                tool = %call.name,
                "cancellation fired during tool dispatch; abandoning tool"
            );
            (
                cancelled_tool_result(call, "cancelled during dispatch"),
                false,
            )
        });
        let result = quoting_decline_messages(result, turn.context.take_decline_messages());
        let cancelled_mid_dispatch = turn.cancel.is_cancelled();
        (
            send_tool_result(turn, call, result, read_host).await,
            cancelled_mid_dispatch,
        )
    }

    pub(super) async fn refuse_withdrawn_calls(
        &self,
        turn: &Turn,
        calls: &[ToolCall],
    ) -> Result<(), LlmError> {
        warn!(
            target: "assistd::agent",
            iteration = turn.iteration,
            "model requested tools after they were withdrawn; ending turn"
        );
        turn.emit(LlmEvent::ToolCallsRequested {
            calls: calls.to_vec(),
        })
        .await;
        let mut results = Vec::with_capacity(calls.len());
        for call in calls {
            let result = cancelled_tool_result(call, "ended; tools are unavailable");
            results.push(send_tool_result(turn, call, result, false).await);
        }
        self.backend.push_tool_results(results).await?;
        turn.emit(LlmEvent::Done).await;
        Ok(())
    }
}

/// Answer a call that will never run, reporting the result like any other
/// so the client and the transcript see the batch completed.
async fn skip_tool_call(turn: &Turn, call: &ToolCall) -> ToolResultPayload {
    send_tool_result(
        turn,
        call,
        cancelled_tool_result(call, "cancelled before dispatch"),
        false,
    )
    .await
}

/// Report `result`, and whether its call read this machine, to the client and return what
/// the model is shown.
async fn send_tool_result(
    turn: &Turn,
    call: &ToolCall,
    result: ToolOutput,
    read_host: bool,
) -> ToolResultPayload {
    let payload = ToolResultPayload {
        call_id: call.id.clone(),
        name: call.name.clone(),
        content: result.output.clone(),
        attachments: result.attachments.clone(),
    };
    turn.emit(LlmEvent::ToolResult {
        id: call.id.clone(),
        name: call.name.clone(),
        result: Box::new(result.report()),
        read_host,
        accesses: result.accesses,
    })
    .await;
    payload
}

/// `result` followed by what the user said when declining the call's prompts.
fn quoting_decline_messages(mut result: ToolOutput, messages: Vec<String>) -> ToolOutput {
    for message in messages {
        if !result.output.ends_with('\n') {
            result.output.push('\n');
        }
        result.output.push_str(DECLINED_WITH_MESSAGE);
        result.output.push_str(&message);
    }
    result
}

fn cancelled_tool_result(call: &ToolCall, reason: &str) -> ToolOutput {
    error_tool_result(
        &format!("[error] {}: agent turn {reason}.", call.name),
        Duration::ZERO,
    )
}

fn error_tool_result(message: &str, duration: Duration) -> ToolOutput {
    let duration_ms = duration.as_millis();
    ToolOutput::failed(format!("{message}\n[exit:-1 | {duration_ms}ms]"), duration)
}

fn panicked_tool_result(
    call: &ToolCall,
    iteration: u32,
    duration: Duration,
    panic: &(dyn Any + Send),
) -> ToolOutput {
    let message = panic_message(panic);
    warn!(
        target: "assistd::agent",
        iteration,
        tool = %call.name,
        duration_ms = duration.as_millis(),
        panic = %message,
        "tool invocation panicked"
    );
    error_tool_result(
        &format!(
            "[error] {}: the tool crashed ({message}). Try: a different command or source.",
            call.name
        ),
        duration,
    )
}

/// `fut`'s output, or `None` once it has run `deadline` without one; the clock stops while
/// `context`'s turn has a prompt parked.
async fn within_deadline<F: Future>(
    context: &TurnContext,
    deadline: Duration,
    fut: F,
) -> Option<F::Output> {
    let Some(mut parked) = context.parked() else {
        return tokio::time::timeout(deadline, fut).await.ok();
    };
    let mut fut = pin!(fut);
    let mut left = deadline;
    loop {
        let paused = *parked.borrow_and_update();
        let started = Instant::now();
        let clock = async {
            if paused {
                std::future::pending::<()>().await;
            } else {
                tokio::time::sleep(left).await;
            }
        };
        tokio::select! {
            output = &mut fut => return Some(output),
            () = clock => return None,
            changed = parked.changed() => {
                if !paused {
                    left = left.saturating_sub(started.elapsed());
                }
                if changed.is_err() {
                    return tokio::time::timeout(left, fut).await.ok();
                }
            }
        }
    }
}

/// Run `call`, returning its result and whether it read this machine.
async fn dispatch_tool_call(
    turn: &Turn,
    call: &ToolCall,
    deadline: Duration,
) -> (ToolOutput, bool) {
    let start = Instant::now();
    let (tools, iteration) = (turn.tools.as_ref(), turn.iteration);

    let Some(tool) = tools.get(&call.name) else {
        return (
            unknown_tool_result(tools, call, iteration, start.elapsed()),
            false,
        );
    };

    let context = turn.call_context(call);
    let invocation = AssertUnwindSafe(tool.invoke(&context, call.arguments.clone())).catch_unwind();
    let finished = if tool.bounded_by_tool_deadline() {
        within_deadline(&turn.context, deadline, invocation).await
    } else {
        Some(invocation.await)
    };
    let read_host = tool.reads_host(&call.arguments);
    if read_host {
        turn.context.record_host_read();
    }
    let result = finished_tool_result(call, iteration, finished, deadline, start);
    (result, read_host)
}

/// The result of a `finished` invocation of `call`, `None` being one its `deadline` cut off.
fn finished_tool_result(
    call: &ToolCall,
    iteration: u32,
    finished: Option<Result<Result<ToolOutput, ToolError>, Box<dyn Any + Send>>>,
    deadline: Duration,
    start: Instant,
) -> ToolOutput {
    let Some(result) = finished else {
        let duration = start.elapsed();
        warn!(
            target: "assistd::agent",
            iteration,
            tool = %call.name,
            duration_ms = duration.as_millis(),
            "tool call exceeded its deadline; abandoning it"
        );
        return error_tool_result(
            &format!(
                "[error] {}: no result after {}s; call abandoned. Try: a faster or narrower command.",
                call.name,
                deadline.as_secs()
            ),
            duration,
        );
    };
    let duration = start.elapsed();

    match result {
        Ok(Ok(output)) => {
            log_completed_call(call, &output, iteration, duration);
            output
        }
        Err(panic) => panicked_tool_result(call, iteration, duration, panic.as_ref()),
        Ok(Err(e)) => {
            warn!(
                target: "assistd::agent",
                iteration,
                tool = %call.name,
                duration_ms = duration.as_millis(),
                error = %e,
                "tool invocation errored"
            );
            error_tool_result(
                &format!(
                    "[error] {}: tool invocation failed. Check: {e}. Try: a different command.",
                    call.name
                ),
                duration,
            )
        }
    }
}

fn unknown_tool_result(
    tools: &ToolRegistry,
    call: &ToolCall,
    iteration: u32,
    duration: Duration,
) -> ToolOutput {
    warn!(
        target: "assistd::agent",
        iteration,
        tool = %call.name,
        duration_ms = duration.as_millis(),
        "unknown tool"
    );
    let available = tools.names().collect::<Vec<_>>().join(", ");
    error_tool_result(
        &format!(
            "[error] agent: unknown tool '{}'. Available: {available}.",
            call.name
        ),
        duration,
    )
}

fn log_completed_call(call: &ToolCall, output: &ToolOutput, iteration: u32, duration: Duration) {
    let command = call
        .arguments
        .get("command")
        .and_then(Value::as_str)
        .unwrap_or("");
    info!(
        target: "assistd::agent",
        iteration,
        tool = %call.name,
        command = %command,
        exit_code = output.exit_code,
        output_size = output.output.len(),
        duration_ms = duration.as_millis(),
        "tool call complete"
    );
}
