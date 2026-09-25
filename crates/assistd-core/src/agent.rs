//! Per-turn agent loop: step the LLM, dispatch the tool calls it
//! requests, feed the results back, repeat until it answers.

use std::sync::Arc;
use std::time::{Duration, Instant};

use base64::Engine;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, instrument, warn};

use assistd_ipc::StatusKind;
use assistd_llm::{
    HealthWaitError, LlmBackend, LlmError, LlmEvent, LlmHealthProbe, StepOutcome, ToolCall,
    ToolResultPayload,
};
use assistd_tools::{Attachment, ToolRegistry};

use crate::recovery::{Component, StatusSeverity};
use crate::recovery_event;

/// How long a turn waits for an LLM restart: the supervisor's worst-case
/// backoff (63s) plus spawn and probe headroom.
const REPLAY_WAIT_BUDGET: Duration = Duration::from_secs(75);

/// Consecutive identical tool calls after which the model is treated as
/// stuck.
const DUPLICATE_CALL_LIMIT: usize = 3;

/// Ceiling on tool-calling steps per turn that keep making different calls.
const MAX_TOOL_STEPS: u32 = 200;

/// Long-lived agent dependencies, reused across turns.
///
/// With a `health` probe the loop replays a step once after a
/// llama-server crash; without one, a crash ends the turn. A tool still
/// running after `tool_deadline` is reported to the model as failed.
pub struct Agent {
    backend: Arc<dyn LlmBackend>,
    tools: Arc<ToolRegistry>,
    health: Option<Arc<dyn LlmHealthProbe>>,
    tool_deadline: Duration,
}

impl Agent {
    /// Build an agent over `backend` and `tools`; see [`Agent`] for how
    /// `health` and `tool_deadline` shape a turn.
    pub fn new(
        backend: Arc<dyn LlmBackend>,
        tools: Arc<ToolRegistry>,
        health: Option<Arc<dyn LlmHealthProbe>>,
        tool_deadline: Duration,
    ) -> Self {
        Self {
            backend,
            tools,
            health,
            tool_deadline,
        }
    }

    /// Run one agent turn, streaming events on `tx` until the model
    /// answers. Stops without error when `tx` closes or `cancel` fires.
    /// Turns must not run concurrently on one agent.
    ///
    /// Errors when the backend fails; an incomplete restart surfaces as
    /// [`LlmError::ServerRestarting`].
    #[instrument(skip_all, name = "agent_turn")]
    pub async fn run_turn(
        &self,
        user_text: String,
        user_attachments: Vec<Attachment>,
        tx: mpsc::Sender<LlmEvent>,
        cancel: CancellationToken,
    ) -> Result<(), LlmError> {
        self.backend.push_user(user_text, user_attachments).await?;
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

            let Some(outcome) = self.step_with_replay(&turn).await? else {
                return Ok(());
            };

            match outcome {
                StepOutcome::Final => {
                    let _ = turn.tx.send(LlmEvent::Done).await;
                    return Ok(());
                }
                StepOutcome::ToolCalls(calls) if turn.tools_withdrawn => {
                    return self.refuse_withdrawn_calls(&turn, &calls).await;
                }
                StepOutcome::ToolCalls(calls) => {
                    let (results, stuck) = self.dispatch_tool_calls(&mut turn, calls).await;
                    self.backend.push_tool_results(results).await?;
                    turn.iteration += 1;
                    if let Some(why) = ToolBudgetExhausted::check(stuck, turn.iteration) {
                        self.withdraw_tools(&mut turn, why).await;
                    }
                }
            }
        }
    }

    async fn refuse_withdrawn_calls(
        &self,
        turn: &Turn,
        calls: &[ToolCall],
    ) -> Result<(), LlmError> {
        warn!(
            target: "assistd::agent",
            iteration = turn.iteration,
            "model requested tools after they were withdrawn; ending turn"
        );
        let results = calls
            .iter()
            .map(|call| cancelled_tool_result(call, "ended; tools are unavailable").0)
            .collect();
        self.backend.push_tool_results(results).await?;
        let _ = turn.tx.send(LlmEvent::Done).await;
        Ok(())
    }

    /// One LLM step; `Ok(None)` means the turn was cancelled. A crash is
    /// replayed once after the server is ready again; any other failure
    /// ends the turn with an error after telling the client.
    async fn step_with_replay(&self, turn: &Turn) -> Result<Option<StepOutcome>, LlmError> {
        let mut restart_attempted = false;
        loop {
            let replay_probe = self.health.as_ref().filter(|_| !restart_attempted);
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
            match (step_result, replay_probe) {
                (Ok(outcome), _) => return Ok(Some(outcome)),
                (Err(LlmError::ServerRestarting(reason)), Some(probe)) => {
                    restart_attempted = true;
                    match await_restart(probe, turn, &reason).await {
                        Replay::Ready => continue,
                        Replay::Cancelled => return Ok(None),
                        Replay::Abandoned(final_msg) => {
                            fail_turn(&turn.tx, &final_msg).await;
                            return Err(LlmError::ServerRestarting(final_msg));
                        }
                    }
                }
                (Err(e), _) => {
                    fail_turn(&turn.tx, &format!("{}: {e}", llm_error_label(&e))).await;
                    return Err(e);
                }
            }
        }
    }

    /// Announce the step's calls, then dispatch each in order. Stops early
    /// with cancelled payloads when the client leaves or the turn is
    /// cancelled. The flag is `true` when a call hit [`DUPLICATE_CALL_LIMIT`].
    async fn dispatch_tool_calls(
        &self,
        turn: &mut Turn,
        calls: Vec<ToolCall>,
    ) -> (Vec<ToolResultPayload>, bool) {
        let mut results = Vec::with_capacity(calls.len());
        let mut stuck = false;
        let _ = turn
            .tx
            .send(LlmEvent::ToolCallsRequested {
                calls: calls.clone(),
            })
            .await;
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
                r = dispatch_tool_call(&self.tools, &call, turn.iteration, self.tool_deadline) => Some(r),
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
            StatusSeverity::Warning,
            Component::Agent,
            "tools_withdrawn",
            iteration = turn.iteration,
            reason = %why.reason(),
            "withdrawing tools; asking model to answer from what it has"
        );
        let _ = turn
            .tx
            .send(status_event(
                StatusSeverity::Warning,
                Component::Agent,
                StatusKind::ToolsWithdrawn,
                format!(
                    "Agent stopped using tools ({}); answering with what it has",
                    why.reason()
                ),
            ))
            .await;
        if let Err(e) = self.backend.set_transient_note(why.model_note()).await {
            warn!(
                target: "assistd::agent",
                error = %e,
                "set_transient_note failed; answering without the note"
            );
        }
        turn.tools_withdrawn = true;
    }
}

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

/// Why the loop stopped accepting tool calls for the rest of the turn.
///
/// The schemas stay in the request after withdrawal; without them the
/// server stops parsing tool-call markup and would stream it as text.
#[derive(Debug, Clone, Copy)]
enum ToolBudgetExhausted {
    Repeating,
    StepCeiling,
}

impl ToolBudgetExhausted {
    fn check(stuck: bool, iteration: u32) -> Option<Self> {
        if stuck {
            Some(Self::Repeating)
        } else if iteration >= MAX_TOOL_STEPS {
            Some(Self::StepCeiling)
        } else {
            None
        }
    }

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

/// How many times in a row the model has issued the same tool call.
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

fn llm_error_label(error: &LlmError) -> &'static str {
    match error {
        LlmError::Chat(_) => "chat backend",
        LlmError::ToolCallParse(_) => "tool-call parse",
        LlmError::Unavailable(_) => "backend unavailable",
        LlmError::ServerRestarting(_) => "llm restarting",
    }
}

/// Tell the client the server crashed, wait for the supervisor to bring
/// it back, and report whether the step can be replayed.
async fn await_restart(probe: &Arc<dyn LlmHealthProbe>, turn: &Turn, reason: &str) -> Replay {
    announce_crash(turn, reason).await;

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
            announce_replay(turn).await;
            Replay::Ready
        }
        Err(wait_err) => Replay::Abandoned(abandon_replay(turn, wait_err).await),
    }
}

async fn announce_crash(turn: &Turn, reason: &str) {
    recovery_event!(
        StatusSeverity::Warning,
        Component::Llm,
        "crash_detected",
        iteration = turn.iteration,
        reason = %reason,
        "llama-server died mid-step; waiting for supervisor restart and replaying"
    );
    let _ = turn
        .tx
        .send(status_event(
            StatusSeverity::Warning,
            Component::Llm,
            StatusKind::Restarting,
            "LLM crashed: restarting and replaying your query".to_string(),
        ))
        .await;
}

async fn announce_replay(turn: &Turn) {
    recovery_event!(
        StatusSeverity::Info,
        Component::Llm,
        "replay_ready",
        iteration = turn.iteration,
        "supervisor reported Ready; replaying user query"
    );
    let _ = turn
        .tx
        .send(status_event(
            StatusSeverity::Info,
            Component::Llm,
            StatusKind::Replaying,
            "LLM restored: replaying your query".to_string(),
        ))
        .await;
}

/// Report the failed restart to the client and return the turn's final
/// error message.
async fn abandon_replay(turn: &Turn, wait_err: HealthWaitError) -> String {
    recovery_event!(
        StatusSeverity::Error,
        Component::Llm,
        "replay_abandoned",
        iteration = turn.iteration,
        wait_error = %wait_err,
        "supervisor did not return to Ready; abandoning replay"
    );
    let final_msg = match wait_err {
        HealthWaitError::Timeout => "LLM did not recover before timeout",
        HealthWaitError::Degraded => "LLM supervisor entered degraded state; restart abandoned",
        HealthWaitError::NoService => "LLM service is not currently attached",
    };
    let _ = turn
        .tx
        .send(status_event(
            StatusSeverity::Error,
            Component::Llm,
            StatusKind::Degraded,
            final_msg.to_string(),
        ))
        .await;
    final_msg.to_string()
}

async fn fail_turn(tx: &mpsc::Sender<LlmEvent>, message: &str) {
    let _ = tx
        .send(LlmEvent::Delta {
            text: format!("\n[agent error: {message}]\n"),
        })
        .await;
    let _ = tx.send(LlmEvent::Done).await;
}

fn status_event(
    severity: StatusSeverity,
    component: Component,
    event: StatusKind,
    message: String,
) -> LlmEvent {
    LlmEvent::Status {
        severity,
        component,
        event,
        message,
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
    deadline: Duration,
) -> (ToolResultPayload, Value) {
    let start = Instant::now();

    let Some(tool) = tools.get(&call.name) else {
        return unknown_tool_result(tools, call, iteration, start.elapsed().as_millis());
    };

    let Ok(result) = tokio::time::timeout(deadline, tool.invoke(call.arguments.clone())).await
    else {
        let duration_ms = start.elapsed().as_millis();
        warn!(
            target: "assistd::agent",
            iteration,
            tool = %call.name,
            duration_ms,
            "tool call exceeded its deadline; abandoning it"
        );
        return error_tool_result(
            call,
            format!(
                "[error] {}: no result after {}s; call abandoned. Try: a faster or narrower command.",
                call.name,
                deadline.as_secs()
            ),
            duration_ms,
        );
    };
    let duration_ms = start.elapsed().as_millis();

    match result {
        Ok(raw) => completed_tool_result(call, raw, iteration, duration_ms),
        Err(e) => {
            warn!(
                target: "assistd::agent",
                iteration,
                tool = %call.name,
                duration_ms,
                error = %e,
                "tool invocation errored"
            );
            error_tool_result(
                call,
                format!(
                    "[error] {}: tool invocation failed. Check: {e}. Try: a different command.",
                    call.name
                ),
                duration_ms,
            )
        }
    }
}

fn unknown_tool_result(
    tools: &ToolRegistry,
    call: &ToolCall,
    iteration: u32,
    duration_ms: u128,
) -> (ToolResultPayload, Value) {
    warn!(
        target: "assistd::agent",
        iteration,
        tool = %call.name,
        duration_ms,
        "unknown tool"
    );
    let available = tools.names().collect::<Vec<_>>().join(", ");
    error_tool_result(
        call,
        format!(
            "[error] agent: unknown tool '{}'. Available: {available}.",
            call.name
        ),
        duration_ms,
    )
}

fn completed_tool_result(
    call: &ToolCall,
    raw: Value,
    iteration: u32,
    duration_ms: u128,
) -> (ToolResultPayload, Value) {
    let content = raw
        .get("output")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let exit_code = raw.get("exit_code").and_then(|v| v.as_i64()).unwrap_or(0);
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
        exit_code,
        output_size = content.len(),
        duration_ms,
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

fn parse_attachment(value: &Value) -> Option<Attachment> {
    let kind = value.get("type")?.as_str()?;
    if kind != "image" {
        return None;
    }
    let mime = value.get("mime")?.as_str()?.to_string();
    let data_b64 = value.get("data")?.as_str()?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data_b64)
        .ok()?;
    Some(Attachment::Image { mime, bytes })
}

#[cfg(test)]
mod tests;
