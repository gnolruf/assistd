//! TUI-backed [`ConfirmationGate`].
//!
//! The gate forwards each confirmation request to the chat app's event
//! loop through an `mpsc` channel, awaits a `oneshot` decision, and
//! converts every failure mode (channel drop, UI shutdown) to `false`.
//! Drop-equals-deny is essential: if the TUI quits mid-prompt, the
//! agent loop must not hang on a pending oneshot.

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};

use assistd_tools::{ConfirmationGate, ConfirmationRequest};

/// One pending confirmation request in flight between the gate and the
/// TUI event loop. The TUI fills in the `responder` when the user
/// presses Y/N (or quits — dropping the responder is interpreted as a
/// denial by the gate).
#[derive(Debug)]
pub struct PendingConfirmation {
    pub request: ConfirmationRequest,
    pub responder: oneshot::Sender<bool>,
}

/// [`ConfirmationGate`] implementation that serializes requests over an
/// `mpsc` channel to the TUI event loop.
pub struct TuiGate {
    inbox: mpsc::Sender<PendingConfirmation>,
}

impl TuiGate {
    pub fn new(inbox: mpsc::Sender<PendingConfirmation>) -> Self {
        Self { inbox }
    }
}

#[async_trait]
impl ConfirmationGate for TuiGate {
    async fn confirm(&self, req: ConfirmationRequest) -> bool {
        let (tx, rx) = oneshot::channel();
        if self
            .inbox
            .send(PendingConfirmation {
                request: req,
                responder: tx,
            })
            .await
            .is_err()
        {
            // TUI event loop has hung up — deny by default.
            return false;
        }
        // Dropped responder (TUI quit mid-prompt) also denies.
        rx.await.unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn approve_from_responder_returns_true() {
        let (tx, mut rx) = mpsc::channel::<PendingConfirmation>(1);
        let gate = TuiGate::new(tx);
        let handle = tokio::spawn(async move {
            gate.confirm(ConfirmationRequest {
                tool: "bash".into(),
                script: "rm -rf foo".into(),
                matched_pattern: "rm -rf".into(),
            })
            .await
        });
        let pending = rx.recv().await.expect("request");
        let _ = pending.responder.send(true);
        assert!(handle.await.unwrap());
    }

    #[tokio::test]
    async fn deny_from_responder_returns_false() {
        let (tx, mut rx) = mpsc::channel::<PendingConfirmation>(1);
        let gate = TuiGate::new(tx);
        let handle = tokio::spawn(async move {
            gate.confirm(ConfirmationRequest {
                tool: "bash".into(),
                script: "rm -rf foo".into(),
                matched_pattern: "rm -rf".into(),
            })
            .await
        });
        let pending = rx.recv().await.expect("request");
        let _ = pending.responder.send(false);
        assert!(!handle.await.unwrap());
    }

    #[tokio::test]
    async fn dropped_responder_returns_false() {
        let (tx, mut rx) = mpsc::channel::<PendingConfirmation>(1);
        let gate = TuiGate::new(tx);
        let handle = tokio::spawn(async move {
            gate.confirm(ConfirmationRequest {
                tool: "bash".into(),
                script: "rm -rf foo".into(),
                matched_pattern: "rm -rf".into(),
            })
            .await
        });
        let pending = rx.recv().await.expect("request");
        drop(pending.responder);
        assert!(!handle.await.unwrap());
    }

    #[tokio::test]
    async fn closed_inbox_returns_false_immediately() {
        let (tx, rx) = mpsc::channel::<PendingConfirmation>(1);
        drop(rx); // TUI event loop gone
        let gate = TuiGate::new(tx);
        let approved = gate
            .confirm(ConfirmationRequest {
                tool: "bash".into(),
                script: "rm -rf foo".into(),
                matched_pattern: "rm -rf".into(),
            })
            .await;
        assert!(!approved);
    }
}
