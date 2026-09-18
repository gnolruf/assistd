//! Per-turn agent loop: step the LLM, dispatch the tool calls it
//! requests, feed the results back, repeat until it answers.
//!
//! Invariant: the caller serialises turns. Two concurrent `run_turn`
//! calls would interleave in the backend's conversation state.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::recovery::{Component, RecoverySeverity};
use crate::recovery_event;
use anyhow::Result;
use assistd_llm::{
    HealthWaitError, LlmBackend, LlmError, LlmEvent, LlmHealthProbe, StepOutcome, ToolCall,
    ToolResultPayload,
};
use assistd_tools::{Attachment, ToolRegistry};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, instrument, warn};

/// Wall-clock budget for waiting on an LLM restart before falling
/// through to a terminal error. The supervisor's worst-case backoff
/// sum is 1+2+4+8+16+32 = 63s; 75s gives headroom for spawn + health
/// probe latency. Longer than this and we treat the restart as
/// hopeless and surface to the user.
const REPLAY_WAIT_BUDGET: Duration = Duration::from_secs(75);

/// Consecutive identical tool calls after which the model is treated
/// as stuck and its tools are withdrawn.
const DUPLICATE_CALL_LIMIT: usize = 3;

/// Ceiling on tool-calling steps per turn. Only bounds a turn that keeps
/// making *different* calls without answering; repeats are caught by
/// [`DUPLICATE_CALL_LIMIT`] long before this.
const MAX_TOOL_STEPS: u32 = 200;

/// Long-lived agent dependencies, reused across turns.
///
/// With a `health` probe the loop recovers from one llama-server crash
/// per step: it waits for the supervisor to report ready and replays the
/// step. Without one, a crash ends the turn.
pub struct Agent {
    backend: Arc<dyn LlmBackend>,
    tools: Arc<ToolRegistry>,
    health: Option<Arc<dyn LlmHealthProbe>>,
}

impl Agent {
    pub fn new(
        backend: Arc<dyn LlmBackend>,
        tools: Arc<ToolRegistry>,
        health: Option<Arc<dyn LlmHealthProbe>>,
    ) -> Self {
        Self {
            backend,
            tools,
            health,
        }
    }

    /// Run one agent turn: stream events on `tx` until the model answers,
    /// the turn is cancelled, or the backend fails.
    ///
    /// The turn stops without error when `tx` closes or `cancel` fires;
    /// both are checked between iterations and raced against the LLM
    /// step and each tool dispatch, so a slow tool is abandoned promptly.
    /// A never-cancelled token is fine for callers that only rely on the
    /// `tx` path.
    #[instrument(skip_all, name = "agent_turn")]
    pub async fn run_turn(
        &self,
        user_text: String,
        user_attachments: Vec<Attachment>,
        tx: mpsc::Sender<LlmEvent>,
        cancel: CancellationToken,
    ) -> Result<()> {
        self.backend
            .push_user(user_text, user_attachments)
            .await
            .map_err(anyhow::Error::new)?;
        let mut turn = Turn {
            tx,
            cancel,
            schemas: self.tools.openai_schemas(),
            tools_withdrawn: false,
            streak: CallStreak::default(),
            iteration: 0,
        };

        loop {
            if turn.stop_requested() {
                debug!(
                    target: "assistd::agent",
                    iteration = turn.iteration,
                    cancelled = turn.cancel.is_cancelled(),
                    "stopping between iterations (client gone or explicit cancel)"
                );
                return Ok(());
            }

            let outcome = match self.step_with_replay(&turn).await? {
                Some(outcome) => outcome,
                None => return Ok(()),
            };

            match outcome {
                StepOutcome::Final => {
                    let _ = turn.tx.send(LlmEvent::Done).await;
                    return Ok(());
                }
                StepOutcome::ToolCalls(calls) if turn.tools_withdrawn => {
                    warn!(
                        target: "assistd::agent",
                        iteration = turn.iteration,
                        "model requested tools after they were withdrawn; ending turn"
                    );
                    let results = calls
                        .iter()
                        .map(|call| cancelled_tool_result(call, "ended; tools are unavailable").0)
                        .collect();
                    self.backend
                        .push_tool_results(results)
                        .await
                        .map_err(anyhow::Error::new)?;
                    let _ = turn.tx.send(LlmEvent::Done).await;
                    return Ok(());
                }
                StepOutcome::ToolCalls(calls) => {
                    let (results, stuck) = self.dispatch_tool_calls(&mut turn, calls).await;
                    self.backend
                        .push_tool_results(results)
                        .await
                        .map_err(anyhow::Error::new)?;
                    turn.iteration += 1;
                    let exhausted = if stuck {
                        Some(ToolBudgetExhausted::Repeating)
                    } else if turn.iteration >= MAX_TOOL_STEPS {
                        Some(ToolBudgetExhausted::StepCeiling)
                    } else {
                        None
                    };
                    if let Some(why) = exhausted {
                        self.withdraw_tools(&mut turn, why).await;
                    }
                }
            }
        }
    }

    /// One LLM step. `Ok(None)` means the turn was cancelled while
    /// waiting. A crash mid-step is retried once after the supervisor
    /// reports the server ready again; any other failure, or a second
    /// crash, ends the turn with an error after telling the client.
    async fn step_with_replay(&self, turn: &Turn) -> Result<Option<StepOutcome>> {
        let mut restart_attempted = false;
        loop {
            let step_result = tokio::select! {
                biased;
                () = turn.cancel.cancelled() => {
                    debug!(
                        target: "assistd::agent",
                        iteration = turn.iteration,
                        "cancellation fired during LLM step; stopping"
                    );
                    return Ok(None);
                }
                r = self.backend.step(turn.schemas.clone(), turn.tx.clone()) => r,
            };
            match step_result {
                Ok(outcome) => return Ok(Some(outcome)),
                Err(LlmError::ServerRestarting(reason))
                    if !restart_attempted && self.health.is_some() =>
                {
                    restart_attempted = true;
                    let probe = self.health.as_ref().expect("health Some by guard");
                    match await_restart(probe, turn, &reason).await {
                        Replay::Ready => continue,
                        Replay::Cancelled => return Ok(None),
                        Replay::Abandoned(final_msg) => {
                            fail_turn(&turn.tx, &final_msg).await;
                            return Err(anyhow::Error::new(LlmError::ServerRestarting(final_msg)));
                        }
                    }
                }
                Err(e) => {
                    let label = match &e {
                        LlmError::Chat(_) => "chat backend",
                        LlmError::ToolCallParse(_) => "tool-call parse",
                        LlmError::Unavailable(_) => "backend unavailable",
                        LlmError::ServerRestarting(_) => "llm restarting",
                    };
                    fail_turn(&turn.tx, &format!("{label}: {e}")).await;
                    return Err(anyhow::Error::new(e));
                }
            }
        }
    }

    /// Dispatch every call in order, emitting `ToolCall` and `ToolResult`
    /// events. Stops early, with cancelled payloads for the rest, when the
    /// client goes away or the turn is cancelled. The flag is `true` when
    /// a call repeated [`DUPLICATE_CALL_LIMIT`] times in a row.
    async fn dispatch_tool_calls(
        &self,
        turn: &mut Turn,
        calls: Vec<ToolCall>,
    ) -> (Vec<ToolResultPayload>, bool) {
        let mut results = Vec::with_capacity(calls.len());
        let mut stuck = false;
        for call in calls {
            stuck |= turn.streak.record(&call) >= DUPLICATE_CALL_LIMIT;
            if turn.stop_requested() {
                debug!(
                    target: "assistd::agent",
                    iteration = turn.iteration,
                    tool = %call.name,
                    cancelled = turn.cancel.is_cancelled(),
                    "stopping mid-call (client gone or explicit cancel) without dispatch"
                );
                results.push(cancelled_tool_result(&call, "cancelled before dispatch").0);
                break;
            }

            let _ = turn
                .tx
                .send(LlmEvent::ToolCall {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                })
                .await;

            let dispatched = tokio::select! {
                biased;
                () = turn.cancel.cancelled() => None,
                r = dispatch_tool_call(&self.tools, &call, turn.iteration) => Some(r),
            };
            let (payload, raw_result) = dispatched.unwrap_or_else(|| {
                warn!(
                    target: "assistd::agent",
                    iteration = turn.iteration,
                    tool = %call.name,
                    "cancellation fired during tool dispatch; abandoning tool"
                );
                cancelled_tool_result(&call, "cancelled during dispatch")
            });
            let cancelled_mid_dispatch = turn.cancel.is_cancelled();

            let _ = turn
                .tx
                .send(LlmEvent::ToolResult {
                    id: payload.call_id.clone(),
                    name: payload.name.clone(),
                    result: raw_result,
                })
                .await;
            results.push(payload);

            if cancelled_mid_dispatch {
                break;
            }
        }
        (results, stuck)
    }

    async fn withdraw_tools(&self, turn: &mut Turn, why: ToolBudgetExhausted) {
        recovery_event!(
            RecoverySeverity::Warning,
            Component::Agent,
            "tools_withdrawn",
            iteration = turn.iteration,
            reason = %why.reason(),
            "withdrawing tools; asking model to answer from what it has"
        );
        let _ = turn
            .tx
            .send(status_event(
                RecoverySeverity::Warning,
                Component::Agent,
                "tools_withdrawn",
                format!(
                    "Agent stopped using tools ({}); answering with what it has",
                    why.reason()
                ),
            ))
            .await;
        if let Err(e) = self.backend.set_transient_context(why.model_note()).await {
            warn!(
                target: "assistd::agent",
                error = %e,
                "set_transient_context failed; answering without the note"
            );
        }
        turn.schemas = Vec::new();
        turn.tools_withdrawn = true;
    }
}

/// Per-turn state threaded through the loop.
struct Turn {
    tx: mpsc::Sender<LlmEvent>,
    cancel: CancellationToken,
    schemas: Vec<Value>,
    tools_withdrawn: bool,
    streak: CallStreak,
    iteration: u32,
}

impl Turn {
    fn stop_requested(&self) -> bool {
        self.tx.is_closed() || self.cancel.is_cancelled()
    }
}

enum Replay {
    Ready,
    Cancelled,
    Abandoned(String),
}

/// Tell the client the server crashed, wait for the supervisor to bring
/// it back, and report whether the step can be replayed.
async fn await_restart(probe: &Arc<dyn LlmHealthProbe>, turn: &Turn, reason: &str) -> Replay {
    recovery_event!(
        RecoverySeverity::Warning,
        Component::Llm,
        "crash_detected",
        iteration = turn.iteration,
        reason = %reason,
        "llama-server died mid-step; waiting for supervisor restart and replaying"
    );
    let _ = turn
        .tx
        .send(status_event(
            RecoverySeverity::Warning,
            Component::Llm,
            "restarting",
            "LLM crashed: restarting and replaying your query".to_string(),
        ))
        .await;

    let wait_result = tokio::select! {
        biased;
        () = turn.cancel.cancelled() => {
            debug!(
                target: "assistd::agent",
                iteration = turn.iteration,
                "cancellation fired while waiting for LLM restart"
            );
            return Replay::Cancelled;
        }
        res = probe.wait_for_ready(REPLAY_WAIT_BUDGET) => res,
    };

    match wait_result {
        Ok(()) => {
            recovery_event!(
                RecoverySeverity::Info,
                Component::Llm,
                "replay_ready",
                iteration = turn.iteration,
                "supervisor reported Ready; replaying user query"
            );
            let _ = turn
                .tx
                .send(status_event(
                    RecoverySeverity::Info,
                    Component::Llm,
                    "replaying",
                    "LLM restored: replaying your query".to_string(),
                ))
                .await;
            Replay::Ready
        }
        Err(wait_err) => {
            recovery_event!(
                RecoverySeverity::Error,
                Component::Llm,
                "replay_abandoned",
                iteration = turn.iteration,
                wait_error = %wait_err,
                "supervisor did not return to Ready; abandoning replay"
            );
            let final_msg = match wait_err {
                HealthWaitError::Timeout => "LLM did not recover before timeout",
                HealthWaitError::Degraded => {
                    "LLM supervisor entered degraded state; restart abandoned"
                }
                HealthWaitError::NoService => "LLM service is not currently attached",
            };
            let _ = turn
                .tx
                .send(status_event(
                    RecoverySeverity::Error,
                    Component::Llm,
                    "degraded",
                    final_msg.to_string(),
                ))
                .await;
            Replay::Abandoned(final_msg.to_string())
        }
    }
}

/// Surface a fatal error in the stream and close it with `Done`.
async fn fail_turn(tx: &mpsc::Sender<LlmEvent>, message: &str) {
    let _ = tx
        .send(LlmEvent::Delta {
            text: format!("\n[agent error: {message}]\n"),
        })
        .await;
    let _ = tx.send(LlmEvent::Done).await;
}

fn status_event(
    severity: RecoverySeverity,
    component: Component,
    event: &str,
    message: String,
) -> LlmEvent {
    LlmEvent::Status {
        severity: severity.as_str().to_string(),
        component: component.as_str().to_string(),
        event: event.to_string(),
        message,
    }
}

/// Why the loop stopped offering tools to the model for the rest of
/// the turn.
#[derive(Debug, Clone, Copy)]
enum ToolBudgetExhausted {
    Repeating,
    StepCeiling,
}

impl ToolBudgetExhausted {
    fn reason(self) -> String {
        match self {
            Self::Repeating => {
                format!("the same tool call was repeated {DUPLICATE_CALL_LIMIT} times in a row")
            }
            Self::StepCeiling => {
                format!("the turn reached its ceiling of {MAX_TOOL_STEPS} tool steps")
            }
        }
    }

    fn model_note(self) -> String {
        format!(
            "Tool access for this turn has been withdrawn because {}. Do not request \
             any more tool calls. Answer the user now using what you have already \
             learned, and say plainly which parts you could not finish.",
            self.reason()
        )
    }
}

/// Tracks how many times in a row the model has issued the same tool
/// call, so a stuck loop is caught by content rather than by count.
#[derive(Default)]
struct CallStreak {
    last: Option<(String, Value)>,
    count: usize,
}

impl CallStreak {
    fn record(&mut self, call: &ToolCall) -> usize {
        let same = self
            .last
            .as_ref()
            .is_some_and(|(name, args)| *name == call.name && *args == call.arguments);
        if same {
            self.count += 1;
        } else {
            self.last = Some((call.name.clone(), call.arguments.clone()));
            self.count = 1;
        }
        self.count
    }
}

fn cancelled_tool_result(call: &ToolCall, reason: &str) -> (ToolResultPayload, Value) {
    error_tool_result(
        call,
        format!("[error] {}: agent turn {reason}.", call.name),
        0,
    )
}

fn error_tool_result(
    call: &ToolCall,
    message: String,
    duration_ms: u128,
) -> (ToolResultPayload, Value) {
    let content = format!("{message}\n[exit:-1 | {duration_ms}ms]");
    let raw = serde_json::json!({
        "output": content,
        "exit_code": -1,
        "duration_ms": duration_ms,
        "truncated": false,
    });
    (
        ToolResultPayload {
            call_id: call.id.clone(),
            name: call.name.clone(),
            content,
            attachments: Vec::new(),
        },
        raw,
    )
}

async fn dispatch_tool_call(
    tools: &ToolRegistry,
    call: &ToolCall,
    iteration: u32,
) -> (ToolResultPayload, Value) {
    let start = Instant::now();

    let Some(tool) = tools.get(&call.name) else {
        let duration_ms = start.elapsed().as_millis();
        warn!(
            target: "assistd::agent",
            iteration,
            tool = %call.name,
            duration_ms = duration_ms,
            "unknown tool"
        );
        let available = tools.names().collect::<Vec<_>>().join(", ");
        return error_tool_result(
            call,
            format!(
                "[error] agent: unknown tool '{}'. Available: {available}.",
                call.name
            ),
            duration_ms,
        );
    };

    let result = tool.invoke(call.arguments.clone()).await;
    let duration_ms = start.elapsed().as_millis();

    let raw = match result {
        Ok(v) => v,
        Err(e) => {
            warn!(
                target: "assistd::agent",
                iteration,
                tool = %call.name,
                duration_ms = duration_ms,
                error = %e,
                "tool invocation errored"
            );
            return error_tool_result(
                call,
                format!(
                    "[error] {}: tool invocation failed. Check: {e}. Try: a different command.",
                    call.name
                ),
                duration_ms,
            );
        }
    };

    let content = raw
        .get("output")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let exit_code = raw.get("exit_code").and_then(|v| v.as_i64()).unwrap_or(0);
    let output_size = content.len();
    let command = call
        .arguments
        .get("command")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    info!(
        target: "assistd::agent",
        iteration,
        tool = %call.name,
        command = %command,
        exit_code = exit_code,
        output_size = output_size,
        duration_ms = duration_ms,
        "tool call complete"
    );

    let attachments = raw
        .get("attachments")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(parse_attachment).collect::<Vec<_>>())
        .unwrap_or_default();

    (
        ToolResultPayload {
            call_id: call.id.clone(),
            name: call.name.clone(),
            content,
            attachments,
        },
        raw,
    )
}

fn parse_attachment(v: &Value) -> Option<Attachment> {
    let kind = v.get("type")?.as_str()?;
    if kind != "image" {
        return None;
    }
    let mime = v.get("mime")?.as_str()?.to_string();
    let data_b64 = v.get("data")?.as_str()?;
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data_b64)
        .ok()?;
    Some(Attachment::Image { mime, bytes })
}

#[cfg(test)]
mod tests;
