//! Confirmation gates and the per-connection router that carries their
//! prompts to an IPC client.

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot};
use tracing::warn;

use assistd_ipc::Event;

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
pub trait ConfirmationGate: Send + Sync + 'static {
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

#[derive(Default)]
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
pub struct ConfirmRouter {
    /// Id of the connection's originating request.
    request_id: String,
    wire: mpsc::Sender<Event>,
    timeout: Duration,
    pending: Mutex<PendingPrompts>,
}

impl ConfirmRouter {
    /// A router whose prompts are denied after `timeout` without an answer.
    pub fn new(request_id: String, wire: mpsc::Sender<Event>, timeout: Duration) -> Arc<Self> {
        Arc::new(Self {
            request_id,
            wire,
            timeout,
            pending: Mutex::new(PendingPrompts::default()),
        })
    }

    /// Forward the prompt to the client and await the answer. Every
    /// failure mode is [`Approval::Deny`].
    pub async fn ask(&self, req: ConfirmationRequest) -> Approval {
        let confirm_id = uuid::Uuid::new_v4().to_string();
        let Some(answer) = self.register(&confirm_id, &req) else {
            return Approval::Deny;
        };
        let event = Event::ConfirmRequest {
            id: self.request_id.clone(),
            confirm_id: confirm_id.clone(),
            tool: req.tool.clone(),
            script: req.script.clone(),
            matched_pattern: req.matched_pattern.clone(),
            always_allow: req.always_allow.clone(),
        };
        if self.wire.send(event).await.is_err() {
            self.forget(&confirm_id);
            warn!(
                target: "assistd::policy",
                tool = %req.tool,
                "command denied: client disconnected before confirm"
            );
            return Approval::Deny;
        }
        self.await_answer(&confirm_id, &req, answer).await
    }

    /// Record a pending prompt under `confirm_id`, or `None` when the
    /// router is closed or full.
    fn register(
        &self,
        confirm_id: &str,
        req: &ConfirmationRequest,
    ) -> Option<oneshot::Receiver<Approval>> {
        let mut pending = self.pending.lock();
        if pending.closed {
            warn!(
                target: "assistd::policy",
                tool = %req.tool,
                pattern = %req.matched_pattern,
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
        let (tx, rx) = oneshot::channel();
        pending.prompts.insert(confirm_id.to_string(), tx);
        Some(rx)
    }

    async fn await_answer(
        &self,
        confirm_id: &str,
        req: &ConfirmationRequest,
        answer: oneshot::Receiver<Approval>,
    ) -> Approval {
        match tokio::time::timeout(self.timeout, answer).await {
            Ok(Ok(approval)) => approval,
            Ok(Err(_)) => {
                self.forget(confirm_id);
                warn!(
                    target: "assistd::policy",
                    tool = %req.tool,
                    "command denied: confirmation channel dropped"
                );
                Approval::Deny
            }
            Err(_) => {
                self.forget(confirm_id);
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

    fn forget(&self, confirm_id: &str) {
        self.pending.lock().prompts.remove(confirm_id);
    }

    /// Deliver a client's answer to the matching pending prompt.
    ///
    /// # Errors
    /// [`NoPendingConfirm`] when no prompt with that id is in flight.
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

    #[cfg(test)]
    fn pending_len(&self) -> usize {
        self.pending.lock().prompts.len()
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

    fn sample_request() -> ConfirmationRequest {
        ConfirmationRequest {
            tool: "bash".into(),
            script: "rm -rf foo".into(),
            matched_pattern: "rm -rf".into(),
            always_allow: Vec::new(),
        }
    }

    async fn recv_confirm_id(rx: &mut mpsc::Receiver<Event>) -> String {
        match rx.recv().await.expect("prompt on the wire") {
            Event::ConfirmRequest { confirm_id, .. } => confirm_id,
            other => panic!("expected ConfirmRequest, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn router_denies_and_forgets_prompt_after_timeout() {
        let (tx, mut rx) = mpsc::channel(4);
        let router = ConfirmRouter::new("r".into(), tx, Duration::from_millis(50));
        let ask = router.ask(sample_request());
        let (approval, confirm_id) = tokio::join!(ask, recv_confirm_id(&mut rx));
        assert_eq!(approval, Approval::Deny);
        assert_eq!(router.pending_len(), 0);
        router
            .route_response(&confirm_id, Approval::Once)
            .expect_err("a timed-out prompt is no longer routable");
    }

    #[tokio::test]
    async fn router_close_denies_in_flight_prompt_and_later_asks() {
        let (tx, mut rx) = mpsc::channel(4);
        let router = ConfirmRouter::new("r".into(), tx, Duration::from_secs(60));
        let asker = Arc::clone(&router);
        let in_flight = tokio::spawn(async move { asker.ask(sample_request()).await });
        recv_confirm_id(&mut rx).await;

        router.close();
        assert_eq!(in_flight.await.expect("ask task"), Approval::Deny);
        assert_eq!(router.pending_len(), 0);

        assert_eq!(router.ask(sample_request()).await, Approval::Deny);
        rx.try_recv()
            .expect_err("a closed router must not put prompts on the wire");
    }

    #[tokio::test]
    async fn router_denies_without_asking_once_pending_cap_is_reached() {
        let (tx, mut rx) = mpsc::channel(4);
        let router = ConfirmRouter::new("r".into(), tx, CONFIRM_TIMEOUT);
        {
            let mut pending = router.pending.lock();
            for i in 0..MAX_PENDING_CONFIRMS {
                pending
                    .prompts
                    .insert(format!("preloaded-{i}"), oneshot::channel().0);
            }
        }

        assert_eq!(router.ask(sample_request()).await, Approval::Deny);
        assert_eq!(router.pending_len(), MAX_PENDING_CONFIRMS);
        rx.try_recv()
            .expect_err("a denied ask must not put a prompt on the wire");
    }

    #[tokio::test]
    async fn inherit_confirm_router_carries_router_across_spawn() {
        let (tx, mut rx) = mpsc::channel(4);
        let router = ConfirmRouter::new("r".into(), tx, Duration::from_secs(60));
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
        let router = ConfirmRouter::new("r".into(), tx, Duration::from_secs(60));
        let answer = CONFIRM_ROUTER.sync_scope(router, || {
            tokio::spawn(async { IpcConfirmationGate.confirm(sample_request()).await })
        });
        assert_eq!(answer.await.expect("spawned gate"), Approval::Deny);
        rx.try_recv()
            .expect_err("a gate with no router must not reach the wire");
    }

    #[test]
    fn an_answer_is_always_only_when_it_allows() {
        assert_eq!(Approval::from_answer(false, true), Approval::Deny);
        assert_eq!(Approval::from_answer(true, false), Approval::Once);
        assert_eq!(Approval::from_answer(true, true), Approval::Always);
    }
}
