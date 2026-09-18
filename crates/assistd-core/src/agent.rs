//! Per-turn agent loop: step the LLM, dispatch the tool calls it
//! requests, feed the results back, repeat until it answers.
//!
//! Invariant: the caller serialises turns. Two concurrent `run_turn`
//! calls would interleave in the backend's conversation state.

use std::sync::Arc;
use std::time::{Duration, Instant};

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
        let backend = &self.backend;
        let tools = &self.tools;
        let health = self.health.as_ref();

        backend
            .push_user(user_text, user_attachments)
            .await
            .map_err(anyhow::Error::new)?;
        let mut schemas = tools.openai_schemas();
        let mut tools_withdrawn = false;
        let mut streak = CallStreak::default();
        let mut iteration: u32 = 0;

        loop {
            if tx.is_closed() || cancel.is_cancelled() {
                debug!(
                    target: "assistd::agent",
                    iteration,
                    cancelled = cancel.is_cancelled(),
                    "stopping between iterations (client gone or explicit cancel)"
                );
                return Ok(());
            }

            let mut restart_attempted = false;

            let outcome = loop {
                let step_result = tokio::select! {
                    biased;
                    () = cancel.cancelled() => {
                        debug!(
                            target: "assistd::agent",
                            iteration,
                            "cancellation fired during LLM step; stopping"
                        );
                        return Ok(());
                    }
                    r = backend.step(schemas.clone(), tx.clone()) => r,
                };

                match step_result {
                    Ok(o) => break o,
                    Err(LlmError::ServerRestarting(reason))
                        if !restart_attempted && health.is_some() =>
                    {
                        restart_attempted = true;
                        crate::recovery_event!(
                            crate::RecoverySeverity::Warning,
                            crate::Component::Llm,
                            "crash_detected",
                            iteration = iteration,
                            reason = %reason,
                            "llama-server died mid-step; waiting for supervisor restart and replaying"
                        );
                        let _ = tx
                            .send(LlmEvent::Status {
                                severity: crate::RecoverySeverity::Warning.as_str().to_string(),
                                component: crate::Component::Llm.as_str().to_string(),
                                event: "restarting".to_string(),
                                message: "LLM crashed: restarting and replaying your query"
                                    .to_string(),
                            })
                            .await;

                        let probe = health.expect("health Some by guard");
                        let wait_result = tokio::select! {
                            biased;
                            () = cancel.cancelled() => {
                                debug!(
                                    target: "assistd::agent",
                                    iteration,
                                    "cancellation fired while waiting for LLM restart"
                                );
                                return Ok(());
                            }
                            res = probe.wait_for_ready(REPLAY_WAIT_BUDGET) => res,
                        };

                        match wait_result {
                            Ok(()) => {
                                crate::recovery_event!(
                                    crate::RecoverySeverity::Info,
                                    crate::Component::Llm,
                                    "replay_ready",
                                    iteration = iteration,
                                    "supervisor reported Ready; replaying user query"
                                );
                                let _ = tx
                                    .send(LlmEvent::Status {
                                        severity: crate::RecoverySeverity::Info
                                            .as_str()
                                            .to_string(),
                                        component: crate::Component::Llm.as_str().to_string(),
                                        event: "replaying".to_string(),
                                        message: "LLM restored: replaying your query".to_string(),
                                    })
                                    .await;
                                continue;
                            }
                            Err(wait_err) => {
                                crate::recovery_event!(
                                    crate::RecoverySeverity::Error,
                                    crate::Component::Llm,
                                    "replay_abandoned",
                                    iteration = iteration,
                                    wait_error = %wait_err,
                                    "supervisor did not return to Ready; abandoning replay"
                                );
                                let final_msg = match wait_err {
                                    HealthWaitError::Timeout => {
                                        "LLM did not recover before timeout"
                                    }
                                    HealthWaitError::Degraded => {
                                        "LLM supervisor entered degraded state; restart abandoned"
                                    }
                                    HealthWaitError::NoService => {
                                        "LLM service is not currently attached"
                                    }
                                };
                                let _ = tx
                                    .send(LlmEvent::Status {
                                        severity: crate::RecoverySeverity::Error
                                            .as_str()
                                            .to_string(),
                                        component: crate::Component::Llm.as_str().to_string(),
                                        event: "degraded".to_string(),
                                        message: final_msg.to_string(),
                                    })
                                    .await;
                                let _ = tx
                                    .send(LlmEvent::Delta {
                                        text: format!("\n[agent error: {final_msg}]\n"),
                                    })
                                    .await;
                                let _ = tx.send(LlmEvent::Done).await;
                                return Err(anyhow::Error::new(LlmError::ServerRestarting(
                                    final_msg.to_string(),
                                )));
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
                        let _ = tx
                            .send(LlmEvent::Delta {
                                text: format!("\n[agent error: {label}: {e}]\n"),
                            })
                            .await;
                        let _ = tx.send(LlmEvent::Done).await;
                        return Err(anyhow::Error::new(e));
                    }
                }
            };

            match outcome {
                StepOutcome::Final => {
                    let _ = tx.send(LlmEvent::Done).await;
                    return Ok(());
                }
                StepOutcome::ToolCalls(calls) if tools_withdrawn => {
                    warn!(
                        target: "assistd::agent",
                        iteration,
                        "model requested tools after they were withdrawn; ending turn"
                    );
                    let results = calls
                        .iter()
                        .map(|call| cancelled_tool_result(call, "ended; tools are unavailable").0)
                        .collect();
                    backend
                        .push_tool_results(results)
                        .await
                        .map_err(anyhow::Error::new)?;
                    let _ = tx.send(LlmEvent::Done).await;
                    return Ok(());
                }
                StepOutcome::ToolCalls(calls) => {
                    let mut results = Vec::with_capacity(calls.len());
                    let mut stuck = false;
                    for call in calls {
                        stuck |= streak.record(&call) >= DUPLICATE_CALL_LIMIT;
                        if tx.is_closed() || cancel.is_cancelled() {
                            debug!(
                                target: "assistd::agent",
                                iteration,
                                tool = %call.name,
                                cancelled = cancel.is_cancelled(),
                                "stopping mid-call (client gone or explicit cancel) without dispatch"
                            );
                            let (payload, _) =
                                cancelled_tool_result(&call, "cancelled before dispatch");
                            results.push(payload);
                            break;
                        }

                        let _ = tx
                            .send(LlmEvent::ToolCall {
                                id: call.id.clone(),
                                name: call.name.clone(),
                                arguments: call.arguments.clone(),
                            })
                            .await;

                        let dispatched = tokio::select! {
                            biased;
                            () = cancel.cancelled() => None,
                            r = dispatch_tool_call(tools, &call, iteration) => Some(r),
                        };

                        let (payload, raw_result) = match dispatched {
                            Some(r) => r,
                            None => {
                                warn!(
                                    target: "assistd::agent",
                                    iteration,
                                    tool = %call.name,
                                    "cancellation fired during tool dispatch; abandoning tool"
                                );
                                cancelled_tool_result(&call, "cancelled during dispatch")
                            }
                        };
                        let cancelled_mid_dispatch = cancel.is_cancelled();

                        let _ = tx
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
                    backend
                        .push_tool_results(results)
                        .await
                        .map_err(anyhow::Error::new)?;

                    iteration += 1;
                    let exhausted = if stuck {
                        Some(ToolBudgetExhausted::Repeating)
                    } else if iteration >= MAX_TOOL_STEPS {
                        Some(ToolBudgetExhausted::StepCeiling)
                    } else {
                        None
                    };
                    if let Some(why) = exhausted {
                        crate::recovery_event!(
                            crate::RecoverySeverity::Warning,
                            crate::Component::Agent,
                            "tools_withdrawn",
                            iteration = iteration,
                            reason = %why.reason(),
                            "withdrawing tools; asking model to answer from what it has"
                        );
                        let _ = tx
                            .send(LlmEvent::Status {
                                severity: crate::RecoverySeverity::Warning.as_str().to_string(),
                                component: crate::Component::Agent.as_str().to_string(),
                                event: "tools_withdrawn".to_string(),
                                message: format!(
                                    "Agent stopped using tools ({}); answering with what it has",
                                    why.reason()
                                ),
                            })
                            .await;
                        if let Err(e) = backend.set_transient_context(why.model_note()).await {
                            warn!(
                                target: "assistd::agent",
                                error = %e,
                                "set_transient_context failed; answering without the note"
                            );
                        }
                        schemas = Vec::new();
                        tools_withdrawn = true;
                    }
                }
            }
        }
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
    let content = format!(
        "[error] {}: agent turn {reason}.\n[exit:-1 | 0ms]",
        call.name
    );
    let raw = serde_json::json!({
        "output": content,
        "exit_code": -1,
        "duration_ms": 0,
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
        let content = format!(
            "[error] agent: unknown tool '{}'. Available: {}. [exit:-1 | {}ms]",
            call.name,
            tools.names().collect::<Vec<_>>().join(", "),
            duration_ms
        );
        warn!(
            target: "assistd::agent",
            iteration,
            tool = %call.name,
            duration_ms = duration_ms,
            "unknown tool"
        );
        let raw = serde_json::json!({
            "output": content,
            "exit_code": -1,
            "duration_ms": duration_ms,
            "truncated": false,
        });
        return (
            ToolResultPayload {
                call_id: call.id.clone(),
                name: call.name.clone(),
                content,
                attachments: Vec::new(),
            },
            raw,
        );
    };

    let result = tool.invoke(call.arguments.clone()).await;
    let duration = start.elapsed();
    let duration_ms = duration.as_millis();

    let raw = match result {
        Ok(v) => v,
        Err(e) => {
            let content = format!(
                "[error] {}: tool invocation failed. Check: {e}. \
                 Try: a different command.\n[exit:-1 | {duration_ms}ms]",
                call.name
            );
            warn!(
                target: "assistd::agent",
                iteration,
                tool = %call.name,
                duration_ms = duration_ms,
                error = %e,
                "tool invocation errored"
            );
            let raw = serde_json::json!({
                "output": content,
                "exit_code": -1,
                "duration_ms": duration_ms,
                "truncated": false,
            });
            return (
                ToolResultPayload {
                    call_id: call.id.clone(),
                    name: call.name.clone(),
                    content,
                    attachments: Vec::new(),
                },
                raw,
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
