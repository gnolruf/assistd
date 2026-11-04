//! Confirmation gates and the per-connection router that carries their
//! prompts to an IPC client.

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot};
use tracing::{info, warn};

use assistd_ipc::{ConfirmSource, Event, PermissionMode};

use super::judge::{JudgedCommand, SafetyJudge, Verdict};

/// Cap on confirmation prompts in flight on one connection.
pub const MAX_PENDING_CONFIRMS: usize = 32;

/// How long a prompt waits for the client's answer before it is denied.
pub const CONFIRM_TIMEOUT: Duration = Duration::from_secs(120);

/// A request for the user's confirmation before a command runs.
#[derive(Debug, Clone)]
pub struct ConfirmationRequest {
    /// Tool name, e.g. `"bash"`.
    pub tool: String,
    /// Verbatim script the tool is about to execute.
    pub script: String,
    /// Why the command needs confirmation, for display.
    pub matched_pattern: String,
    /// Programs an [`Approval::Always`] answer adds to the allowlist.
    pub always_allow: Vec<String>,
    /// How much it could do beyond what the prompt shows.
    pub risk: Risk,
    /// Which permission modes may approve it without asking.
    pub unattended: Unattended,
    /// The agent whose command it is.
    pub source: ConfirmSource,
}

/// Which permission modes may approve a [`ConfirmationRequest`] without asking the user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unattended {
    /// Every mode asks.
    Never,
    /// [`PermissionMode::AcceptEdits`] and [`PermissionMode::Auto`] approve it: an edit of
    /// ordinary paths inside the edit scope.
    ScopedEdit,
    /// [`PermissionMode::Auto`] approves it when the safety judge allows.
    Judged(JudgeContext),
}

/// What the safety judge is told about a command besides its text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JudgeContext {
    pub class: JudgedClass,
    pub cwd: Option<PathBuf>,
    /// Why the policy could not settle it, shown to the judge.
    pub why: String,
}

/// The class of a command put to the safety judge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JudgedClass {
    /// An edit with a target outside the edit scope.
    Edit,
    Unvetted,
    /// A `wm open` launch the user has not approved for good.
    Launch,
    /// A call to an MCP tool the user has not approved for good.
    McpTool,
}

impl fmt::Display for JudgedClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Edit => "edit",
            Self::Unvetted => "unvetted",
            Self::Launch => "launch",
            Self::McpTool => "mcp tool",
        })
    }
}

/// How much a [`ConfirmationRequest`] could do beyond what its tool shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Risk {
    /// Edits files the prompt names, or launches an app.
    Routine,
    /// Anything else: an unvetted command, a sensitive path, a write too long to show in full, a
    /// new web host or an unapproved MCP tool.
    Elevated,
}

/// The user's answer to a [`ConfirmationRequest`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Approval {
    Deny,
    Once,
    /// Run, and add the request's `always_allow` programs to the allowlist.
    Always,
}

impl Approval {
    /// The approval an IPC client's answer stands for.
    pub fn from_answer(allow: bool, always: bool) -> Self {
        match (allow, always) {
            (false, _) => Self::Deny,
            (true, false) => Self::Once,
            (true, true) => Self::Always,
        }
    }
}

/// Decides whether a command that needs confirmation may run.
#[async_trait]
pub trait ConfirmationGate: fmt::Debug + Send + Sync + 'static {
    /// Ask for confirmation. Every failure mode (channel drop, shutdown,
    /// timeout) must become [`Approval::Deny`] so a turn never hangs.
    async fn confirm(&self, req: ConfirmationRequest) -> Approval;
}

/// Gate that never approves, logging each denial.
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug, Default)]
pub struct DenyAllGate;

#[cfg(any(test, feature = "test-support"))]
#[async_trait]
impl ConfirmationGate for DenyAllGate {
    async fn confirm(&self, req: ConfirmationRequest) -> Approval {
        warn!(
            target: "assistd::policy",
            tool = %req.tool,
            pattern = %req.matched_pattern,
            "command denied: no interactive confirmation gate attached"
        );
        Approval::Deny
    }
}

/// Gate that approves every request once.
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug, Default)]
pub struct AlwaysAllowGate;

#[cfg(any(test, feature = "test-support"))]
#[async_trait]
impl ConfirmationGate for AlwaysAllowGate {
    async fn confirm(&self, _req: ConfirmationRequest) -> Approval {
        Approval::Once
    }
}

#[derive(Debug, Default)]
struct PendingPrompts {
    /// The client can no longer answer; every later ask is denied.
    closed: bool,
    prompts: HashMap<String, oneshot::Sender<Approval>>,
}

/// An answer named a `confirm_id` with no prompt in flight.
#[derive(Debug, thiserror::Error)]
#[error("no pending confirm for this confirm_id")]
pub struct NoPendingConfirm;

/// Per-connection routing table for in-flight confirmation prompts,
/// reached through the [`CONFIRM_ROUTER`] task-local. Asks beyond
/// [`MAX_PENDING_CONFIRMS`] in flight are denied rather than queued.
#[derive(Debug)]
pub struct ConfirmRouter {
    /// Id of the connection's originating request.
    request_id: String,
    wire: mpsc::Sender<Event>,
    timeout: Duration,
    pending: Mutex<PendingPrompts>,
    mode: Mutex<PermissionMode>,
    /// Answers for the user in [`PermissionMode::Auto`], which is refused without one.
    judge: Option<SafetyJudge>,
    /// Replaces [`ConfirmSource::Main`] on every prompt asked through this router.
    main_source: Option<ConfirmSource>,
}

impl ConfirmRouter {
    /// A router in [`PermissionMode::Manual`] whose prompts are denied after `timeout` without
    /// an answer, and which honors [`PermissionMode::Auto`] only with a `judge`.
    pub fn new(
        request_id: String,
        wire: mpsc::Sender<Event>,
        timeout: Duration,
        judge: Option<SafetyJudge>,
    ) -> Arc<Self> {
        Arc::new(Self {
            request_id,
            wire,
            timeout,
            pending: Mutex::new(PendingPrompts::default()),
            mode: Mutex::new(PermissionMode::Manual),
            judge,
            main_source: None,
        })
    }

    /// A router with no client behind it, which denies every prompt and labels the main
    /// agent's with `source`.
    pub fn unattended(request_id: String, source: ConfirmSource) -> Arc<Self> {
        let (wire, _) = mpsc::channel(1);
        Arc::new(Self {
            request_id,
            wire,
            timeout: CONFIRM_TIMEOUT,
            pending: Mutex::new(PendingPrompts {
                closed: true,
                prompts: HashMap::new(),
            }),
            mode: Mutex::new(PermissionMode::Manual),
            judge: None,
            main_source: Some(source),
        })
    }

    /// Answer later asks under `mode`, or under [`PermissionMode::Manual`] when it is Auto and
    /// there is no judge; prompts already shown stay open.
    pub fn set_mode(&self, mode: PermissionMode) {
        let mode = if mode == PermissionMode::Auto && self.judge.is_none() {
            warn!(
                target: "assistd::policy",
                "auto mode is off (tools.allow_auto_mode); using manual mode"
            );
            PermissionMode::Manual
        } else {
            mode
        };
        *self.mode.lock() = mode;
    }

    /// Approve `req` once when the mode lets it run unasked; otherwise prompt the client and
    /// await the answer. Every failure is [`Approval::Deny`].
    pub async fn ask(&self, mut req: ConfirmationRequest) -> Approval {
        if let Some(source) = &self.main_source
            && req.source.is_main()
        {
            req.source = source.clone();
        }
        let mode = *self.mode.lock();
        if self.approves_unasked(mode, &mut req).await {
            return Approval::Once;
        }
        self.ask_client(req).await
    }

    /// Whether `mode` lets `req` run unasked, logging each approval. A judge's reason to ask is
    /// added to the prompt.
    async fn approves_unasked(&self, mode: PermissionMode, req: &mut ConfirmationRequest) -> bool {
        match (mode, &req.unattended, &self.judge) {
            (PermissionMode::AcceptEdits | PermissionMode::Auto, Unattended::ScopedEdit, _) => {
                info!(
                    target: "assistd::policy",
                    tool = %req.tool,
                    ?mode,
                    class = "edit",
                    "approved without asking"
                );
                true
            }
            (PermissionMode::Auto, Unattended::Judged(context), Some(judge)) => {
                let judged = JudgedCommand {
                    command: &req.script,
                    cwd: context.cwd.as_deref(),
                    unvetted: &context.why,
                };
                match judge.judge(judged).await {
                    Verdict::Allow(reason) => {
                        info!(
                            target: "assistd::policy",
                            tool = %req.tool,
                            ?mode,
                            class = %context.class,
                            reason,
                            "approved without asking"
                        );
                        true
                    }
                    Verdict::Ask(reason) => {
                        req.matched_pattern =
                            format!("{}; the safety judge asks: {reason}", req.matched_pattern);
                        false
                    }
                }
            }
            _ => false,
        }
    }

    async fn ask_client(&self, req: ConfirmationRequest) -> Approval {
        let Some((prompt, answer)) = self.register(&req) else {
            return Approval::Deny;
        };
        let event = Event::ConfirmRequest {
            id: self.request_id.clone(),
            confirm_id: prompt.confirm_id.clone(),
            tool: req.tool.clone(),
            script: req.script.clone(),
            matched_pattern: req.matched_pattern.clone(),
            always_allow: req.always_allow.clone(),
            source: req.source.clone(),
        };
        if self.wire.send(event).await.is_err() {
            warn!(
                target: "assistd::policy",
                tool = %req.tool,
                "command denied: client disconnected before confirm"
            );
            return Approval::Deny;
        }
        self.await_answer(&req, answer).await
    }

    /// Record a pending prompt under a fresh id, or `None` when the
    /// router is closed or full.
    fn register(
        &self,
        req: &ConfirmationRequest,
    ) -> Option<(PendingPrompt<'_>, oneshot::Receiver<Approval>)> {
        let mut pending = self.pending.lock();
        if pending.closed {
            warn!(
                target: "assistd::policy",
                tool = %req.tool,
                pattern = %req.matched_pattern,
                source = ?req.source,
                "command denied: client cannot answer prompts"
            );
            return None;
        }
        if pending.prompts.len() >= MAX_PENDING_CONFIRMS {
            warn!(
                target: "assistd::policy",
                tool = %req.tool,
                in_flight = pending.prompts.len(),
                cap = MAX_PENDING_CONFIRMS,
                "command denied: pending-confirm cap reached"
            );
            return None;
        }
        let confirm_id = uuid::Uuid::new_v4().to_string();
        let (tx, rx) = oneshot::channel();
        pending.prompts.insert(confirm_id.clone(), tx);
        let prompt = PendingPrompt {
            router: self,
            confirm_id,
        };
        Some((prompt, rx))
    }

    async fn await_answer(
        &self,
        req: &ConfirmationRequest,
        answer: oneshot::Receiver<Approval>,
    ) -> Approval {
        match tokio::time::timeout(self.timeout, answer).await {
            Ok(Ok(approval)) => approval,
            Ok(Err(_)) => {
                warn!(
                    target: "assistd::policy",
                    tool = %req.tool,
                    "command denied: confirmation channel dropped"
                );
                Approval::Deny
            }
            Err(_) => {
                warn!(
                    target: "assistd::policy",
                    tool = %req.tool,
                    timeout_secs = self.timeout.as_secs(),
                    "command denied: no answer before timeout"
                );
                Approval::Deny
            }
        }
    }

    /// Deliver a client's answer to the matching pending prompt.
    /// Fails with [`NoPendingConfirm`] when no prompt with that id is in flight.
    pub fn route_response(
        &self,
        confirm_id: &str,
        approval: Approval,
    ) -> Result<(), NoPendingConfirm> {
        let tx = self
            .pending
            .lock()
            .prompts
            .remove(confirm_id)
            .ok_or(NoPendingConfirm)?;
        let _ = tx.send(approval);
        Ok(())
    }

    /// Deny every prompt in flight and every later ask.
    pub fn close(&self) {
        let drained = {
            let mut pending = self.pending.lock();
            pending.closed = true;
            std::mem::take(&mut pending.prompts)
        };
        for (_, tx) in drained {
            let _ = tx.send(Approval::Deny);
        }
    }
}

/// A prompt registered with a [`ConfirmRouter`], removed from its
/// routing table when dropped.
struct PendingPrompt<'router> {
    router: &'router ConfirmRouter,
    confirm_id: String,
}

impl Drop for PendingPrompt<'_> {
    fn drop(&mut self) {
        self.router.pending.lock().prompts.remove(&self.confirm_id);
    }
}

tokio::task_local! {
    /// The [`ConfirmRouter`] of the IPC connection whose request is
    /// being dispatched.
    pub static CONFIRM_ROUTER: Arc<ConfirmRouter>;
}

/// Wrap `fut` so it runs under the caller's [`CONFIRM_ROUTER`], if any.
/// Task-locals do not survive `tokio::spawn`, so call this at the spawn site.
pub fn inherit_confirm_router<F: Future>(fut: F) -> impl Future<Output = F::Output> {
    let router = CONFIRM_ROUTER.try_with(Arc::clone).ok();
    async move {
        match router {
            Some(router) => CONFIRM_ROUTER.scope(router, fut).await,
            None => fut.await,
        }
    }
}

/// Gate that round-trips prompts through the [`CONFIRM_ROUTER`] in scope,
/// denying when there is none.
#[derive(Debug, Default)]
pub struct IpcConfirmationGate;

#[async_trait]
impl ConfirmationGate for IpcConfirmationGate {
    async fn confirm(&self, req: ConfirmationRequest) -> Approval {
        match CONFIRM_ROUTER.try_with(Arc::clone) {
            Ok(router) => router.ask(req).await,
            Err(_) => {
                warn!(
                    target: "assistd::policy",
                    tool = %req.tool,
                    pattern = %req.matched_pattern,
                    "command denied: no IPC client attached to ask"
                );
                Approval::Deny
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::test_support::FixedJudge;
    use PermissionMode::{AcceptEdits, Auto, Manual};

    fn sample_request() -> ConfirmationRequest {
        ConfirmationRequest {
            tool: "bash".into(),
            script: "rm -rf foo".into(),
            matched_pattern: "rm -rf".into(),
            always_allow: Vec::new(),
            risk: Risk::Elevated,
            unattended: Unattended::Never,
            source: ConfirmSource::Main,
        }
    }

    fn judged(class: JudgedClass) -> Unattended {
        Unattended::Judged(JudgeContext {
            class,
            cwd: None,
            why: "stub".into(),
        })
    }

    async fn recv_confirm_id(rx: &mut mpsc::Receiver<Event>) -> String {
        match rx.recv().await.expect("prompt on the wire") {
            Event::ConfirmRequest { confirm_id, .. } => confirm_id,
            other => panic!("expected ConfirmRequest, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn each_mode_runs_unasked_only_what_its_class_allows() {
        let unattended = [
            Unattended::Never,
            Unattended::ScopedEdit,
            judged(JudgedClass::Edit),
            judged(JudgedClass::Unvetted),
        ];
        for judge in [None, Some("allow"), Some("ask")] {
            for mode in [Manual, AcceptEdits, Auto] {
                for class in &unattended {
                    let effective = match (mode, judge) {
                        (Auto, None) => Manual,
                        _ => mode,
                    };
                    let prompted = match (effective, class) {
                        (_, Unattended::Never) | (Manual, _) => true,
                        (_, Unattended::ScopedEdit) => false,
                        (AcceptEdits, Unattended::Judged(_)) => true,
                        _ => judge != Some("allow"),
                    };
                    let (tx, mut rx) = mpsc::channel(4);
                    let safety_judge = judge.map(|decision| {
                        SafetyJudge::new(Arc::new(FixedJudge(decision)), Duration::from_secs(5))
                    });
                    let router =
                        ConfirmRouter::new("r".into(), tx, Duration::from_millis(10), safety_judge);
                    router.set_mode(mode);
                    let approval = router
                        .ask(ConfirmationRequest {
                            unattended: class.clone(),
                            ..sample_request()
                        })
                        .await;
                    let case = format!("{mode:?} {class:?}");
                    assert_eq!(rx.try_recv().is_ok(), prompted, "{case}");
                    let expected = if prompted {
                        Approval::Deny
                    } else {
                        Approval::Once
                    };
                    assert_eq!(approval, expected, "{case}");
                }
            }
        }
    }

    #[tokio::test]
    async fn router_denies_and_forgets_prompt_after_timeout() {
        let (tx, mut rx) = mpsc::channel(4);
        let router = ConfirmRouter::new("r".into(), tx, Duration::from_millis(50), None);
        let ask = router.ask(sample_request());
        let (approval, confirm_id) = tokio::join!(ask, recv_confirm_id(&mut rx));
        assert_eq!(approval, Approval::Deny);
        assert_eq!(router.pending.lock().prompts.len(), 0);
        router
            .route_response(&confirm_id, Approval::Once)
            .expect_err("a timed-out prompt is no longer routable");
    }

    #[tokio::test]
    async fn cancelled_ask_forgets_its_prompt() {
        let (tx, mut rx) = mpsc::channel(4);
        let router = ConfirmRouter::new("r".into(), tx, Duration::from_secs(60), None);
        let asker = Arc::clone(&router);
        let in_flight = tokio::spawn(async move { asker.ask(sample_request()).await });
        let confirm_id = recv_confirm_id(&mut rx).await;

        in_flight.abort();
        in_flight.await.expect_err("ask was aborted");
        assert_eq!(router.pending.lock().prompts.len(), 0);
        router
            .route_response(&confirm_id, Approval::Once)
            .expect_err("a cancelled prompt is no longer routable");
    }

    #[tokio::test]
    async fn router_close_denies_in_flight_prompt_and_later_asks() {
        let (tx, mut rx) = mpsc::channel(4);
        let router = ConfirmRouter::new("r".into(), tx, Duration::from_secs(60), None);
        let asker = Arc::clone(&router);
        let in_flight = tokio::spawn(async move { asker.ask(sample_request()).await });
        recv_confirm_id(&mut rx).await;

        router.close();
        assert_eq!(in_flight.await.expect("ask task"), Approval::Deny);
        assert_eq!(router.pending.lock().prompts.len(), 0);

        assert_eq!(router.ask(sample_request()).await, Approval::Deny);
        rx.try_recv()
            .expect_err("a closed router must not put prompts on the wire");
    }

    #[tokio::test]
    async fn router_denies_without_asking_once_pending_cap_is_reached() {
        let (tx, mut rx) = mpsc::channel(4);
        let router = ConfirmRouter::new("r".into(), tx, CONFIRM_TIMEOUT, None);
        {
            let mut pending = router.pending.lock();
            for i in 0..MAX_PENDING_CONFIRMS {
                pending
                    .prompts
                    .insert(format!("preloaded-{i}"), oneshot::channel().0);
            }
        }

        assert_eq!(router.ask(sample_request()).await, Approval::Deny);
        assert_eq!(router.pending.lock().prompts.len(), MAX_PENDING_CONFIRMS);
        rx.try_recv()
            .expect_err("a denied ask must not put a prompt on the wire");
    }

    #[tokio::test]
    async fn inherit_confirm_router_carries_router_across_spawn() {
        let (tx, mut rx) = mpsc::channel(4);
        let router = ConfirmRouter::new("r".into(), tx, Duration::from_secs(60), None);
        let answer = CONFIRM_ROUTER.sync_scope(Arc::clone(&router), || {
            tokio::spawn(inherit_confirm_router(async {
                IpcConfirmationGate.confirm(sample_request()).await
            }))
        });
        let confirm_id = recv_confirm_id(&mut rx).await;
        router
            .route_response(&confirm_id, Approval::Always)
            .expect("routed");
        assert_eq!(answer.await.expect("spawned gate"), Approval::Always);
    }

    #[tokio::test]
    async fn bare_spawn_loses_router_and_gate_denies() {
        let (tx, mut rx) = mpsc::channel(4);
        let router = ConfirmRouter::new("r".into(), tx, Duration::from_secs(60), None);
        let answer = CONFIRM_ROUTER.sync_scope(router, || {
            tokio::spawn(async { IpcConfirmationGate.confirm(sample_request()).await })
        });
        assert_eq!(answer.await.expect("spawned gate"), Approval::Deny);
        rx.try_recv()
            .expect_err("a gate with no router must not reach the wire");
    }
}
