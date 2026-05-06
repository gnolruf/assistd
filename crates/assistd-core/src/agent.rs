//! Single-tool agent loop.
//!
//! Orchestrates one "user turn": push the user's prompt into the backend's
//! conversation state, call [`LlmBackend::step`] with the current tool
//! schemas, dispatch any tool calls the model requests through the
//! shared `ToolRegistry`, feed results back, and iterate. Emits
//! [`LlmEvent::ToolCall`] / [`LlmEvent::ToolResult`] on `tx` so callers
//! (daemon IPC, TUI) can surface the intermediate state to users.
//!
//! Invariant: the caller holds whatever turn-level lock is needed to
//! prevent concurrent agent turns from interleaving in the backend's
//! conversation state. `AppState::handle_query` owns that lock.

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

/// Run one agent turn end-to-end.
///
/// Lifecycle:
/// 1. Push `user_text` into conversation state.
/// 2. Call `backend.step(tools_schemas, tx)`.
/// 3. If the step returned text, send [`LlmEvent::Done`] and stop.
/// 4. Otherwise: for each requested tool call, emit
///    [`LlmEvent::ToolCall`], dispatch it via the registry, emit
///    [`LlmEvent::ToolResult`], and collect a [`ToolResultPayload`].
///    After the loop, push the collected results back with
///    `backend.push_tool_results(...)` and go to step 2.
/// 5. If `max_iterations` is exhausted, emit a user-visible error delta
///    and [`LlmEvent::Done`].
///
/// Cancellation: if `tx` is closed (client disconnected) between
/// iterations, the loop returns `Ok(())` without running further tool
/// calls or LLM round trips. Callers can also explicitly cancel by
/// signalling the [`CancellationToken`] — useful when the daemon needs
/// to stop a long-running LLM step or tool call promptly (e.g. a slow
/// MCP tool when the user disconnects). The token is checked between
/// iterations and the LLM step / tool dispatch run inside a
/// `select!` against it so cancellation propagates without waiting
/// for the inner future to complete on its own.
///
/// Pass [`CancellationToken::new`] (a fresh, never-cancelled token) if
/// the caller doesn't need explicit cancellation — the
/// `tx.is_closed()` path keeps existing behaviour for that case.
///
/// `health` is the optional restart probe used to recover from
/// llama-server crashes mid-turn. When `Some`, the loop reacts to
/// [`LlmError::ServerRestarting`] by emitting a `Status` event,
/// waiting for `ReadyState::Ready`, and re-running the same step
/// once. Pass `None` for tests / mock backends that never crash.
#[allow(clippy::too_many_arguments)]
#[instrument(skip_all, name = "agent_turn")]
pub async fn run_agent_turn(
    backend: Arc<dyn LlmBackend>,
    tools: Arc<ToolRegistry>,
    max_iterations: u32,
    user_text: String,
    user_attachments: Vec<Attachment>,
    tx: mpsc::Sender<LlmEvent>,
    cancel: CancellationToken,
    health: Option<Arc<dyn LlmHealthProbe>>,
) -> Result<()> {
    backend
        .push_user(user_text, user_attachments)
        .await
        .map_err(anyhow::Error::new)?;
    let schemas = tools.openai_schemas();

    for iteration in 0..max_iterations {
        if tx.is_closed() || cancel.is_cancelled() {
            debug!(
                target: "assistd::agent",
                iteration,
                cancelled = cancel.is_cancelled(),
                "stopping between iterations (client gone or explicit cancel)"
            );
            return Ok(());
        }

        // Per-iteration replay budget: at most one restart-driven
        // retry. The supervisor's own consecutive-failure cap (5) is
        // the daemon-level circuit breaker; this keeps the user-visible
        // latency bounded if the supervisor recovers but the model
        // still crashes on the same prompt.
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
                            message: "LLM crashed — restarting and replaying your query"
                                .to_string(),
                        })
                        .await;

                    // Probe is guaranteed Some by the match guard.
                    let probe = health.as_ref().expect("health Some by guard");
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
                                    severity: crate::RecoverySeverity::Info.as_str().to_string(),
                                    component: crate::Component::Llm.as_str().to_string(),
                                    event: "replaying".to_string(),
                                    message: "LLM restored — replaying your query".to_string(),
                                })
                                .await;
                            // Retry the same iteration.
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
                                HealthWaitError::Timeout => "LLM did not recover before timeout",
                                HealthWaitError::Degraded => {
                                    "LLM supervisor entered degraded state; restart abandoned"
                                }
                                HealthWaitError::NoService => {
                                    "LLM service is not currently attached"
                                }
                            };
                            let _ = tx
                                .send(LlmEvent::Status {
                                    severity: crate::RecoverySeverity::Error.as_str().to_string(),
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
                    // Surface the typed error category to the model so
                    // a transient HTTP hiccup doesn't read the same as
                    // a persistent backend-unavailable state. The
                    // human-readable Display is still the primary
                    // signal in the user-facing delta.
                    let label = match &e {
                        LlmError::Chat(_) => "chat backend",
                        LlmError::ToolCallParse(_) => "tool-call parse",
                        LlmError::Unavailable(_) => "backend unavailable",
                        // Already-attempted restart, or no probe wired.
                        // Fall through as a terminal failure.
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
            StepOutcome::ToolCalls(calls) => {
                let mut results = Vec::with_capacity(calls.len());
                for call in calls {
                    if tx.is_closed() || cancel.is_cancelled() {
                        debug!(
                            target: "assistd::agent",
                            iteration,
                            tool = %call.name,
                            cancelled = cancel.is_cancelled(),
                            "stopping mid-call (client gone or explicit cancel) without dispatch"
                        );
                        // Unwind: push synthetic errors for all pending calls
                        // so conversation state isn't left with a dangling
                        // assistant-with-tool_calls on the next turn.
                        results.push(ToolResultPayload {
                            call_id: call.id,
                            name: call.name,
                            content: "[error] run: agent turn cancelled before dispatch.\n\
                                       [exit:-1 | 0ms]"
                                .to_string(),
                            attachments: Vec::new(),
                        });
                        break;
                    }

                    // Surface the in-flight call to clients before it runs.
                    let _ = tx
                        .send(LlmEvent::ToolCall {
                            id: call.id.clone(),
                            name: call.name.clone(),
                            arguments: call.arguments.clone(),
                        })
                        .await;

                    let (payload, raw_result) = dispatch_tool_call(&tools, &call, iteration).await;

                    // Emit the result for downstream observers. Ignore
                    // send errors — the next `tx.is_closed()` check will
                    // catch the disconnect.
                    let _ = tx
                        .send(LlmEvent::ToolResult {
                            id: payload.call_id.clone(),
                            name: payload.name.clone(),
                            result: raw_result,
                        })
                        .await;

                    results.push(payload);
                }
                backend
                    .push_tool_results(results)
                    .await
                    .map_err(anyhow::Error::new)?;
            }
        }
    }

    warn!(
        target: "assistd::agent",
        max_iterations,
        "agent turn exceeded max iterations; giving up"
    );
    let _ = tx
        .send(LlmEvent::Delta {
            text: format!("\n[agent exceeded max_iterations={max_iterations}; stopping]\n"),
        })
        .await;
    let _ = tx.send(LlmEvent::Done).await;
    Ok(())
}

/// Dispatch one tool call and produce both the LLM-facing payload and the
/// raw JSON result (for event emission).
///
/// Error handling: failures at any level become synthetic tool results
/// formatted with the Layer 2 navigational-hint convention
/// (`[error] <cmd>: <what>. <Hint>: <recovery>`) so the model can
/// self-correct on the next step. Only fundamental errors (like a
/// `send` failure on `tx`) bubble up to the loop — and even those
/// result in graceful shutdown, not a hung history.
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

/// Decode one `attachments[i]` entry from a RunTool result. Currently
/// only handles `{"type": "image", "mime": "...", "data": <base64>}`.
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
mod tests {
    use super::*;
    use assistd_config::ToolsOutputConfig;
    use assistd_llm::{LlmBackend, LlmEvent, StepOutcome, ToolCall};
    use assistd_tools::{CommandRegistry, RunTool};
    use async_trait::async_trait;
    use std::sync::Mutex as StdMutex;

    /// Scripted mock backend: returns queued step outcomes in order.
    /// Records what was pushed so tests can assert the loop fed results
    /// back correctly.
    struct MockBackend {
        outcomes: StdMutex<Vec<StepOutcome>>,
        pushed_users: StdMutex<Vec<String>>,
        pushed_attachments: StdMutex<Vec<Vec<Attachment>>>,
        pushed_results: StdMutex<Vec<Vec<ToolResultPayload>>>,
        /// Counts every entry into `step` (including ones cancelled
        /// before they finish). Used by cancellation tests to verify
        /// the loop didn't iterate past a cancellation point.
        step_calls: std::sync::atomic::AtomicUsize,
        /// Optional artificial delay inside `step` so cancellation
        /// tests can fire while the step is still pending.
        slow_step: StdMutex<Option<std::time::Duration>>,
    }

    impl MockBackend {
        fn with(outcomes: Vec<StepOutcome>) -> Arc<Self> {
            Arc::new(Self {
                outcomes: StdMutex::new(outcomes),
                pushed_users: StdMutex::new(Vec::new()),
                pushed_attachments: StdMutex::new(Vec::new()),
                pushed_results: StdMutex::new(Vec::new()),
                step_calls: std::sync::atomic::AtomicUsize::new(0),
                slow_step: StdMutex::new(None),
            })
        }

        fn slow_step_ms(self: Arc<Self>, ms: u64) -> Arc<Self> {
            *self.slow_step.lock().unwrap() = Some(std::time::Duration::from_millis(ms));
            self
        }
    }

    #[async_trait]
    impl LlmBackend for MockBackend {
        async fn generate(
            &self,
            _prompt: String,
            _tx: mpsc::Sender<LlmEvent>,
        ) -> assistd_llm::LlmResult<()> {
            unimplemented!("mock uses step path only")
        }

        async fn push_user(
            &self,
            text: String,
            attachments: Vec<Attachment>,
        ) -> assistd_llm::LlmResult<()> {
            self.pushed_users.lock().unwrap().push(text);
            self.pushed_attachments.lock().unwrap().push(attachments);
            Ok(())
        }

        async fn push_tool_results(
            &self,
            results: Vec<ToolResultPayload>,
        ) -> assistd_llm::LlmResult<()> {
            self.pushed_results.lock().unwrap().push(results);
            Ok(())
        }

        async fn step(
            &self,
            _tools: Vec<Value>,
            tx: mpsc::Sender<LlmEvent>,
        ) -> assistd_llm::LlmResult<StepOutcome> {
            self.step_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // Honour any configured slow-step delay BEFORE consuming
            // an outcome — cancellation tests rely on this future
            // being long-lived so the cancel signal can race it.
            let delay = *self.slow_step.lock().unwrap();
            if let Some(d) = delay {
                tokio::time::sleep(d).await;
            }
            let outcome = {
                let mut q = self.outcomes.lock().unwrap();
                if q.is_empty() {
                    StepOutcome::Final
                } else {
                    q.remove(0)
                }
            };
            if matches!(outcome, StepOutcome::Final) {
                // Emulate a real step sending a text delta on Final so
                // callers can observe streaming.
                let _ = tx.send(LlmEvent::Delta { text: "ok".into() }).await;
            }
            Ok(outcome)
        }
    }

    fn call(id: &str, command: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: "run".into(),
            arguments: serde_json::json!({ "command": command }),
        }
    }

    fn tools_with_echo() -> Arc<ToolRegistry> {
        use assistd_tools::commands::EchoCommand;
        let mut reg = CommandRegistry::new();
        reg.register(EchoCommand);
        let mut tools = ToolRegistry::new();
        tools.register(RunTool::new(
            Arc::new(reg),
            &ToolsOutputConfig::default(),
            std::env::temp_dir().join(format!("assistd-agent-test-{}", std::process::id())),
        ));
        Arc::new(tools)
    }

    async fn collect(rx: &mut mpsc::Receiver<LlmEvent>) -> Vec<LlmEvent> {
        let mut out = Vec::new();
        while let Some(ev) = rx.recv().await {
            out.push(ev);
        }
        out
    }

    #[tokio::test]
    async fn simple_query_one_step_final() {
        let backend = MockBackend::with(vec![StepOutcome::Final]);
        let tools = tools_with_echo();
        let (tx, mut rx) = mpsc::channel(16);
        run_agent_turn(
            backend.clone(),
            tools,
            10,
            "what is 2+2?".into(),
            Vec::new(),
            tx,
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();
        let events = collect(&mut rx).await;
        assert!(matches!(events.last(), Some(LlmEvent::Done)));
        assert_eq!(backend.pushed_users.lock().unwrap().len(), 1);
        // No tool calls dispatched → no tool_results pushed.
        assert!(backend.pushed_results.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn multi_step_tool_then_final() {
        let backend = MockBackend::with(vec![
            StepOutcome::ToolCalls(vec![call("c-1", "echo hello")]),
            StepOutcome::Final,
        ]);
        let tools = tools_with_echo();
        let (tx, mut rx) = mpsc::channel(16);
        run_agent_turn(
            backend.clone(),
            tools,
            10,
            "say hello".into(),
            Vec::new(),
            tx,
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();
        let events = collect(&mut rx).await;

        // Expected event sequence (intermixed with deltas):
        // ToolCall, ToolResult, Delta("ok"), Done
        assert!(
            events
                .iter()
                .any(|e| matches!(e, LlmEvent::ToolCall { name, .. } if name == "run"))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, LlmEvent::ToolResult { name, .. } if name == "run"))
        );
        assert!(matches!(events.last(), Some(LlmEvent::Done)));

        // And the backend saw the echoed result pushed back before the
        // second step.
        let pushed = backend.pushed_results.lock().unwrap();
        assert_eq!(pushed.len(), 1);
        assert_eq!(pushed[0].len(), 1);
        assert!(
            pushed[0][0].content.contains("hello"),
            "result content should contain echo output: {:?}",
            pushed[0][0].content
        );
    }

    #[tokio::test]
    async fn piped_command_completes_in_one_iteration() {
        // A single `run` call with pipes — no need for multiple iterations.
        let backend = MockBackend::with(vec![
            StepOutcome::ToolCalls(vec![call(
                "c-1",
                "echo \"alpha\nbeta\nalpha\" | grep alpha | wc -l",
            )]),
            StepOutcome::Final,
        ]);
        use assistd_tools::commands::{EchoCommand, GrepCommand, WcCommand};
        let mut reg = CommandRegistry::new();
        reg.register(EchoCommand);
        reg.register(GrepCommand);
        reg.register(WcCommand);
        let mut tools = ToolRegistry::new();
        tools.register(RunTool::new(
            Arc::new(reg),
            &ToolsOutputConfig::default(),
            std::env::temp_dir().join(format!("assistd-agent-test-pipe-{}", std::process::id())),
        ));
        let tools = Arc::new(tools);

        let (tx, mut rx) = mpsc::channel(16);
        run_agent_turn(
            backend.clone(),
            tools,
            10,
            "how many?".into(),
            Vec::new(),
            tx,
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();
        let events = collect(&mut rx).await;
        // Exactly one ToolCall/ToolResult pair.
        let calls = events
            .iter()
            .filter(|e| matches!(e, LlmEvent::ToolCall { .. }))
            .count();
        assert_eq!(calls, 1);

        let results = backend.pushed_results.lock().unwrap();
        assert_eq!(results.len(), 1);
        // `echo "alpha\nbeta\nalpha"` prints 1 line; grep alpha filters
        // the full body; wc -l counts it. The exact count depends on
        // how echo interprets \n — but the content should start with an
        // integer and contain an [exit:0] footer.
        assert!(
            results[0][0].content.contains("[exit:0"),
            "expected success footer: {:?}",
            results[0][0].content
        );
    }

    #[tokio::test]
    async fn exceeds_max_iterations_emits_error_delta_and_done() {
        // Keep requesting tool calls forever.
        let outcomes: Vec<StepOutcome> = (0..10)
            .map(|i| {
                StepOutcome::ToolCalls(vec![call(&format!("c-{i}"), &format!("echo iter{i}"))])
            })
            .collect();
        let backend = MockBackend::with(outcomes);
        let tools = tools_with_echo();
        let (tx, mut rx) = mpsc::channel(64);
        run_agent_turn(
            backend,
            tools,
            3,
            "loop it".into(),
            Vec::new(),
            tx,
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();
        let events = collect(&mut rx).await;

        let delta_text: String = events
            .iter()
            .filter_map(|e| match e {
                LlmEvent::Delta { text } => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert!(
            delta_text.contains("max_iterations=3"),
            "expected cap message: {delta_text:?}"
        );
        assert!(matches!(events.last(), Some(LlmEvent::Done)));
    }

    #[tokio::test]
    async fn unknown_tool_passes_error_to_next_step() {
        // Model calls a non-existent tool; dispatch produces an error
        // payload rather than aborting; next step sees it and goes Final.
        let backend = MockBackend::with(vec![
            StepOutcome::ToolCalls(vec![ToolCall {
                id: "c-1".into(),
                name: "nonexistent".into(),
                arguments: serde_json::json!({}),
            }]),
            StepOutcome::Final,
        ]);
        let tools = tools_with_echo();
        let (tx, mut rx) = mpsc::channel(16);
        run_agent_turn(
            backend.clone(),
            tools,
            10,
            "go".into(),
            Vec::new(),
            tx,
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();
        drop(collect(&mut rx).await);
        let pushed = backend.pushed_results.lock().unwrap();
        assert_eq!(pushed.len(), 1);
        let payload = &pushed[0][0];
        assert!(
            payload.content.starts_with("[error] agent: unknown tool"),
            "expected unknown-tool error prefix: {:?}",
            payload.content
        );
    }

    #[tokio::test]
    async fn tool_invoke_err_becomes_synthetic_error_result() {
        // A hand-written tool that errors at the Rust level (rather than
        // via a non-zero exit code) triggers the dispatch `Err` path and
        // must be converted into a synthetic tool-result payload so the
        // next step can self-correct.
        struct ErrTool;
        #[async_trait]
        impl assistd_tools::Tool for ErrTool {
            fn name(&self) -> &str {
                "run"
            }
            fn description(&self) -> &str {
                "errors on invoke"
            }
            fn parameters_schema(&self) -> Value {
                serde_json::json!({"type":"object"})
            }
            async fn invoke(&self, _args: Value) -> anyhow::Result<Value> {
                anyhow::bail!("boom")
            }
        }
        let mut reg = ToolRegistry::new();
        reg.register(ErrTool);
        let reg = Arc::new(reg);

        let backend = MockBackend::with(vec![
            StepOutcome::ToolCalls(vec![call("c-1", "whatever")]),
            StepOutcome::Final,
        ]);
        let (tx, mut rx) = mpsc::channel(16);
        run_agent_turn(
            backend.clone(),
            reg,
            10,
            "go".into(),
            Vec::new(),
            tx,
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();
        drop(collect(&mut rx).await);
        let pushed = backend.pushed_results.lock().unwrap();
        let payload = &pushed[0][0];
        assert!(
            payload
                .content
                .starts_with("[error] run: tool invocation failed"),
            "expected tool-error prefix: {:?}",
            payload.content
        );
    }

    #[tokio::test]
    async fn client_disconnect_between_iterations_stops_loop() {
        let backend = MockBackend::with(vec![
            StepOutcome::ToolCalls(vec![call("c-1", "echo a")]),
            // If we ever reach this outcome it's a bug.
            StepOutcome::ToolCalls(vec![call("c-2", "echo b")]),
        ]);
        let tools = tools_with_echo();
        let (tx, rx) = mpsc::channel::<LlmEvent>(16);
        drop(rx);
        // Channel is closed from the start, so the very first is_closed
        // check bails the loop before any step runs.
        run_agent_turn(
            backend.clone(),
            tools,
            10,
            "go".into(),
            Vec::new(),
            tx,
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();
        // With the receiver already dropped, no step was called.
        let pushed = backend.pushed_results.lock().unwrap();
        assert!(
            pushed.is_empty(),
            "no tool results should be pushed after disconnect: {pushed:?}"
        );
    }

    #[tokio::test]
    async fn explicit_cancel_before_first_step_stops_loop_immediately() {
        // The token is cancelled before the loop starts. The very
        // first iteration's pre-step check sees `cancel.is_cancelled()`
        // and bails — no step is ever called.
        let backend = MockBackend::with(vec![
            StepOutcome::Final,
            // If we get here the cancellation didn't take effect.
            StepOutcome::Final,
        ]);
        let tools = tools_with_echo();
        let (tx, _rx) = mpsc::channel::<LlmEvent>(16);
        let token = CancellationToken::new();
        token.cancel();
        run_agent_turn(
            backend.clone(),
            tools,
            10,
            "go".into(),
            Vec::new(),
            tx,
            token,
            None,
        )
        .await
        .unwrap();
        // user push happened (it's the first thing in the loop), but
        // no step ran.
        assert_eq!(backend.pushed_users.lock().unwrap().len(), 1);
        assert_eq!(
            backend.step_calls.load(std::sync::atomic::Ordering::SeqCst),
            0
        );
    }

    #[tokio::test]
    async fn agent_loop_routes_remember_then_recall() {
        // Drive the full loop through the same `RememberTool` /
        // `RecallTool` impls the daemon registers, against a real
        // SQLite-backed `MemoryOps`. Remember on the first step, recall
        // on the second, then assert both calls were routed and that
        // the save landed in the underlying store. With the embedding
        // subsystem disabled (NoEmbedder/NoSemanticStore + empty model
        // name), recall short-circuits to the `(no memories)` sentinel —
        // the agent-loop wiring is what's under test here, not the
        // semantic round-trip (which is exercised in the embedder
        // integration tests).
        use assistd_memory::{
            ConversationStore, MemoryStore, SqliteConversationStore, SqliteHandle,
            SqliteMemoryStore,
        };
        use assistd_tools::{MemoryOps, RecallTool, RememberTool};
        use tokio::sync::watch;

        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("memory.db");
        // Leak the tempdir for the test — `path` must outlive the
        // handle, and cleanup happens at process exit.
        std::mem::forget(temp);
        let (_tx, rx) = watch::channel(false);
        let (handle, _writer) = SqliteHandle::open(&path, rx).await.unwrap();
        let handle = Arc::new(handle);
        let mem: Arc<dyn MemoryStore> = Arc::new(SqliteMemoryStore::new(handle.clone()));
        let conv: Arc<dyn ConversationStore> = Arc::new(SqliteConversationStore::new(handle));
        let memory_ops = Arc::new(MemoryOps::new(mem.clone(), conv));

        let mut tools = ToolRegistry::new();
        let (embed_tx, embed_rx) = tokio::sync::mpsc::channel::<assistd_embed::EmbedJob>(1);
        drop(embed_rx);
        let no_embedder: Arc<dyn assistd_embed::Embedder> = Arc::new(assistd_embed::NoEmbedder);
        let no_semantic: Arc<dyn assistd_memory::SemanticStore> =
            Arc::new(assistd_memory::NoSemanticStore);
        tools.register(RememberTool::new(memory_ops, embed_tx));
        tools.register(RecallTool::new(no_embedder, no_semantic, String::new()));
        let tools = Arc::new(tools);

        // Step 1: model calls remember(...) — Step 2: model calls
        // recall(...) — Step 3: Final.
        let backend = MockBackend::with(vec![
            StepOutcome::ToolCalls(vec![ToolCall {
                id: "c-1".into(),
                name: "remember".into(),
                arguments: serde_json::json!({
                    "key": "editor_preference",
                    "value": "vim",
                }),
            }]),
            StepOutcome::ToolCalls(vec![ToolCall {
                id: "c-2".into(),
                name: "recall".into(),
                arguments: serde_json::json!({"query": "what editor do I prefer"}),
            }]),
            StepOutcome::Final,
        ]);

        let (tx, mut rx) = mpsc::channel(32);
        run_agent_turn(
            backend.clone(),
            tools,
            10,
            "I prefer vim over emacs".into(),
            Vec::new(),
            tx,
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();
        let events = collect(&mut rx).await;

        // Two ToolCalls emitted, one for each step.
        let names: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                LlmEvent::ToolCall { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(names, vec!["remember", "recall"]);

        // Save landed in the underlying SQLite store (the wire-level
        // contract for `remember`).
        assert_eq!(
            mem.load("editor_preference").await.unwrap().as_deref(),
            Some("vim")
        );

        // Recall's ToolResult and the second push_tool_results carry
        // the no-memories sentinel — the embedder is disabled, so
        // semantic search has nothing to return. The point of this
        // assertion is that the recall result *was* round-tripped back
        // to the model on the next step (proving the loop wiring), not
        // that the data was actually retrievable.
        let recall_output = events
            .iter()
            .find_map(|e| match e {
                LlmEvent::ToolResult { name, result, .. } if name == "recall" => Some(
                    result
                        .get("output")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                ),
                _ => None,
            })
            .expect("expected a recall ToolResult");
        assert_eq!(recall_output, "(no memories)");

        let pushed = backend.pushed_results.lock().unwrap();
        assert_eq!(pushed.len(), 2);
        assert_eq!(pushed[1][0].content, "(no memories)");
    }

    /// Fake `Tool` shaped like `McpToolAdapter`: returns the exact
    /// envelope the live adapter produces (`type` / `output` /
    /// `exit_code` / `duration_ms` / `truncated`) so the agent loop
    /// dispatch path is exercised end-to-end without bringing
    /// `assistd-mcp` in as a dev-dep. Carries a fixed scripted result
    /// so the test can assert what flowed back to the model.
    struct FakeMcpTool {
        name: String,
        result: Value,
    }

    #[async_trait]
    impl assistd_tools::Tool for FakeMcpTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "fake mcp tool"
        }
        fn parameters_schema(&self) -> Value {
            serde_json::json!({"type": "object"})
        }
        async fn invoke(&self, _args: Value) -> anyhow::Result<Value> {
            Ok(self.result.clone())
        }
    }

    /// Acceptance: a turn that mixes an MCP-shaped tool call and a
    /// native `run` call drives both through `dispatch_tool_call`,
    /// emits a `ToolCall`/`ToolResult` pair for each, and pushes both
    /// results back to the backend in order. This is the wire-level
    /// proof that the agent loop is tool-source-agnostic — MCP tools
    /// and native tools share the same dispatch/result/feedback path.
    #[tokio::test]
    async fn agent_loop_mixes_native_and_mcp_calls() {
        use assistd_tools::commands::EchoCommand;
        let mut reg = CommandRegistry::new();
        reg.register(EchoCommand);
        let mut tools = ToolRegistry::new();
        tools.register(RunTool::new(
            Arc::new(reg),
            &ToolsOutputConfig::default(),
            std::env::temp_dir().join(format!("assistd-agent-test-mix-{}", std::process::id())),
        ));
        // Same envelope shape `McpToolAdapter::tool_result_to_json`
        // produces for a `ToolResult::Text("calendar entries")`.
        tools.register(FakeMcpTool {
            name: "mcp__google_calendar__list_events".into(),
            result: serde_json::json!({
                "type": "text",
                "output": "10:00 standup\n14:00 review",
                "exit_code": 0,
                "duration_ms": 12,
                "truncated": false,
            }),
        });
        let tools = Arc::new(tools);

        let backend = MockBackend::with(vec![
            // Step 1: model calls the MCP tool.
            StepOutcome::ToolCalls(vec![ToolCall {
                id: "c-mcp".into(),
                name: "mcp__google_calendar__list_events".into(),
                arguments: serde_json::json!({"date": "tomorrow"}),
            }]),
            // Step 2: model follows up with a native `run` call —
            // proving the loop accepts a different tool on the next
            // step without state leaking from the prior MCP call.
            StepOutcome::ToolCalls(vec![call("c-run", "echo hello")]),
            StepOutcome::Final,
        ]);

        let (tx, mut rx) = mpsc::channel(32);
        run_agent_turn(
            backend.clone(),
            tools,
            10,
            "what's on my calendar tomorrow?".into(),
            Vec::new(),
            tx,
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();
        let events = collect(&mut rx).await;

        let names: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                LlmEvent::ToolCall { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            names,
            vec!["mcp__google_calendar__list_events", "run"],
            "tool calls didn't fire in the scripted order"
        );

        let pushed = backend.pushed_results.lock().unwrap();
        assert_eq!(pushed.len(), 2, "two push_tool_results round-trips");
        assert!(
            pushed[0][0].content.contains("standup"),
            "MCP body must reach next step: {:?}",
            pushed[0][0].content
        );
        assert!(
            pushed[1][0].content.contains("hello"),
            "native echo body must reach next step: {:?}",
            pushed[1][0].content
        );
    }

    /// Acceptance: when the MCP adapter returns its error envelope
    /// (`exit_code: -1`, convention-compliant `output` line), the
    /// agent loop threads it back to the model unchanged so the
    /// recovery hint survives. Without this, a brittle catch-all
    /// elsewhere in the dispatch path would clobber the line and the
    /// model would lose the actionable next step.
    #[tokio::test]
    async fn agent_loop_propagates_mcp_error_envelope_to_next_step() {
        let mut tools = ToolRegistry::new();
        tools.register(FakeMcpTool {
            name: "mcp__google_calendar__list_events".into(),
            // Mirrors `error_envelope` in `assistd-mcp/src/lib.rs` —
            // the line carries `Check:` which the model needs to see
            // intact.
            result: serde_json::json!({
                "type": "error",
                "output": "[error] mcp__google_calendar__list_events: \
                           MCP server returned error code -32602: Invalid params. \
                           Check: the arguments and try again\n",
                "exit_code": -1,
                "duration_ms": 7,
                "truncated": false,
            }),
        });
        let tools = Arc::new(tools);

        let backend = MockBackend::with(vec![
            StepOutcome::ToolCalls(vec![ToolCall {
                id: "c-mcp".into(),
                name: "mcp__google_calendar__list_events".into(),
                arguments: serde_json::json!({}),
            }]),
            StepOutcome::Final,
        ]);
        let (tx, mut rx) = mpsc::channel(16);
        run_agent_turn(
            backend.clone(),
            tools,
            10,
            "what's on my calendar?".into(),
            Vec::new(),
            tx,
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();
        drop(collect(&mut rx).await);

        let pushed = backend.pushed_results.lock().unwrap();
        assert_eq!(pushed.len(), 1);
        let payload = &pushed[0][0];
        assert!(
            payload
                .content
                .starts_with("[error] mcp__google_calendar__list_events: "),
            "convention prefix lost: {:?}",
            payload.content
        );
        assert!(
            payload.content.contains("Check:"),
            "recovery hint stripped before reaching the model: {:?}",
            payload.content
        );
        assert!(
            payload.content.contains("-32602"),
            "rpc code lost: {:?}",
            payload.content
        );
    }

    #[tokio::test]
    async fn cancellation_during_slow_step_preempts_loop() {
        // Fire a cancellation while the LLM step is still pending,
        // and verify run_agent_turn returns promptly via the
        // tokio::select! against `cancel.cancelled()` rather than
        // waiting for the step future to complete on its own.
        let backend = MockBackend::with(vec![StepOutcome::Final]).slow_step_ms(2_000);
        let tools = tools_with_echo();
        let (tx, _rx) = mpsc::channel::<LlmEvent>(16);
        let token = CancellationToken::new();
        let token_for_kicker = token.clone();
        // Spawn a task that fires cancel after a short delay.
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            token_for_kicker.cancel();
        });
        let started = std::time::Instant::now();
        run_agent_turn(
            backend.clone(),
            tools,
            10,
            "go".into(),
            Vec::new(),
            tx,
            token,
            None,
        )
        .await
        .unwrap();
        let elapsed = started.elapsed();
        // The slow step is 2s; if cancellation didn't preempt we'd
        // be waiting that long. Allow a generous 500ms buffer for
        // CI scheduling variance.
        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "cancellation did not preempt slow step (elapsed: {elapsed:?})"
        );
    }

    /// Mock probe that lets a test script the wait_for_ready outcome.
    /// Used to drive the replay-on-ServerRestarting path without
    /// standing up a real PresenceManager.
    struct MockProbe {
        pid: StdMutex<Option<u32>>,
        state: StdMutex<assistd_llm::ReadyState>,
        wait_result: StdMutex<Option<Result<(), assistd_llm::HealthWaitError>>>,
        wait_calls: std::sync::atomic::AtomicUsize,
    }

    impl MockProbe {
        fn ready_with_wait_ok(pid: u32) -> Arc<Self> {
            Arc::new(Self {
                pid: StdMutex::new(Some(pid)),
                state: StdMutex::new(assistd_llm::ReadyState::Ready),
                wait_result: StdMutex::new(Some(Ok(()))),
                wait_calls: std::sync::atomic::AtomicUsize::new(0),
            })
        }

        fn ready_with_wait_err(err: assistd_llm::HealthWaitError) -> Arc<Self> {
            Arc::new(Self {
                pid: StdMutex::new(Some(99)),
                state: StdMutex::new(assistd_llm::ReadyState::Degraded),
                wait_result: StdMutex::new(Some(Err(err))),
                wait_calls: std::sync::atomic::AtomicUsize::new(0),
            })
        }
    }

    #[async_trait]
    impl LlmHealthProbe for MockProbe {
        fn pid(&self) -> Option<u32> {
            *self.pid.lock().unwrap()
        }

        fn state(&self) -> Option<assistd_llm::ReadyState> {
            Some(*self.state.lock().unwrap())
        }

        async fn wait_for_ready(
            &self,
            _budget: std::time::Duration,
        ) -> Result<(), assistd_llm::HealthWaitError> {
            self.wait_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            (*self.wait_result.lock().unwrap()).unwrap_or(Ok(()))
        }
    }

    /// Mock backend that injects errors before falling through to the
    /// existing outcome queue. Used to exercise error-handling paths
    /// (server-restart replay, terminal failures) without needing a
    /// real LlamaChatClient.
    struct ErrorInjectingBackend {
        errors: StdMutex<Vec<LlmError>>,
        outcomes: StdMutex<Vec<StepOutcome>>,
        step_calls: std::sync::atomic::AtomicUsize,
        pushed_users: StdMutex<Vec<String>>,
    }

    impl ErrorInjectingBackend {
        fn new(errors: Vec<LlmError>, outcomes: Vec<StepOutcome>) -> Arc<Self> {
            Arc::new(Self {
                errors: StdMutex::new(errors),
                outcomes: StdMutex::new(outcomes),
                step_calls: std::sync::atomic::AtomicUsize::new(0),
                pushed_users: StdMutex::new(Vec::new()),
            })
        }
    }

    #[async_trait]
    impl LlmBackend for ErrorInjectingBackend {
        async fn generate(
            &self,
            _prompt: String,
            _tx: mpsc::Sender<LlmEvent>,
        ) -> assistd_llm::LlmResult<()> {
            unimplemented!("mock uses step path only")
        }

        async fn push_user(
            &self,
            text: String,
            _attachments: Vec<Attachment>,
        ) -> assistd_llm::LlmResult<()> {
            self.pushed_users.lock().unwrap().push(text);
            Ok(())
        }

        async fn push_tool_results(
            &self,
            _results: Vec<ToolResultPayload>,
        ) -> assistd_llm::LlmResult<()> {
            Ok(())
        }

        async fn step(
            &self,
            _tools: Vec<Value>,
            tx: mpsc::Sender<LlmEvent>,
        ) -> assistd_llm::LlmResult<StepOutcome> {
            self.step_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // Errors first; once they're exhausted, fall through to
            // outcomes. Lets a single test script "first attempt
            // crashes, second attempt succeeds".
            let next_err = self.errors.lock().unwrap().pop();
            if let Some(e) = next_err {
                return Err(e);
            }
            let outcome = self
                .outcomes
                .lock()
                .unwrap()
                .pop()
                .unwrap_or(StepOutcome::Final);
            if matches!(outcome, StepOutcome::Final) {
                let _ = tx.send(LlmEvent::Delta { text: "ok".into() }).await;
            }
            Ok(outcome)
        }
    }

    #[tokio::test]
    async fn replay_once_on_server_restarting() {
        // First attempt: ServerRestarting. Probe says Ready returns Ok.
        // Second attempt: Final.
        // Expected: exactly one step retry; Status events for restarting
        // and replaying; final Done; no terminal Error.
        let backend = ErrorInjectingBackend::new(
            vec![LlmError::ServerRestarting("boom".into())],
            vec![StepOutcome::Final],
        );
        let probe = MockProbe::ready_with_wait_ok(123);
        let tools = tools_with_echo();
        let (tx, mut rx) = mpsc::channel(32);
        run_agent_turn(
            backend.clone(),
            tools,
            10,
            "hello".into(),
            Vec::new(),
            tx,
            CancellationToken::new(),
            Some(probe.clone()),
        )
        .await
        .expect("replay should succeed");

        let events = collect(&mut rx).await;
        let restart_count = events
            .iter()
            .filter(|e| matches!(e, LlmEvent::Status { event, .. } if event == "restarting"))
            .count();
        let replaying_count = events
            .iter()
            .filter(|e| matches!(e, LlmEvent::Status { event, .. } if event == "replaying"))
            .count();
        assert_eq!(restart_count, 1, "expected one restarting Status event");
        assert_eq!(replaying_count, 1, "expected one replaying Status event");
        assert!(matches!(events.last(), Some(LlmEvent::Done)));
        assert_eq!(
            probe.wait_calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "wait_for_ready should be called exactly once"
        );
        // Two step calls: the failing one and the successful retry.
        assert_eq!(
            backend.step_calls.load(std::sync::atomic::Ordering::SeqCst),
            2
        );
    }

    #[tokio::test]
    async fn replay_does_not_loop_on_repeated_server_restarting() {
        // Two ServerRestarting in a row: the loop should retry once,
        // see the second crash, and surface a terminal error rather
        // than continue replaying. The supervisor's circuit-breaker
        // (Degraded after 5 consecutive failures) is the daemon-level
        // backstop; this test asserts the per-iteration cap.
        let backend = ErrorInjectingBackend::new(
            vec![
                LlmError::ServerRestarting("second".into()),
                LlmError::ServerRestarting("first".into()),
            ],
            vec![],
        );
        let probe = MockProbe::ready_with_wait_ok(123);
        let tools = tools_with_echo();
        let (tx, mut rx) = mpsc::channel(32);
        let result = run_agent_turn(
            backend.clone(),
            tools,
            10,
            "hello".into(),
            Vec::new(),
            tx,
            CancellationToken::new(),
            Some(probe),
        )
        .await;
        assert!(result.is_err(), "second ServerRestarting must be terminal");

        let events = collect(&mut rx).await;
        let restart_count = events
            .iter()
            .filter(|e| matches!(e, LlmEvent::Status { event, .. } if event == "restarting"))
            .count();
        assert_eq!(
            restart_count, 1,
            "should attempt replay only once, then surface error"
        );
        assert!(matches!(events.last(), Some(LlmEvent::Done)));
        // Two step calls: original + one replay attempt.
        assert_eq!(
            backend.step_calls.load(std::sync::atomic::Ordering::SeqCst),
            2
        );
    }

    #[tokio::test]
    async fn replay_abandons_on_degraded_supervisor() {
        // ServerRestarting once, but the probe reports Degraded.
        // Expected: a `degraded` Status event followed by a terminal
        // error. No second step call.
        let backend =
            ErrorInjectingBackend::new(vec![LlmError::ServerRestarting("dead".into())], vec![]);
        let probe = MockProbe::ready_with_wait_err(assistd_llm::HealthWaitError::Degraded);
        let tools = tools_with_echo();
        let (tx, mut rx) = mpsc::channel(32);
        let result = run_agent_turn(
            backend.clone(),
            tools,
            10,
            "hello".into(),
            Vec::new(),
            tx,
            CancellationToken::new(),
            Some(probe),
        )
        .await;
        assert!(result.is_err(), "Degraded probe must surface as error");

        let events = collect(&mut rx).await;
        assert!(
            events
                .iter()
                .any(|e| matches!(e, LlmEvent::Status { event, .. } if event == "degraded")),
            "expected a degraded Status event"
        );
        assert!(matches!(events.last(), Some(LlmEvent::Done)));
        // Only one step call — wait_for_ready returned Err, so no
        // replay step was attempted.
        assert_eq!(
            backend.step_calls.load(std::sync::atomic::Ordering::SeqCst),
            1
        );
    }
}
