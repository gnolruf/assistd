//! `SessionContext`: the session and branch the user's conversation is on.

use tokio::sync::{RwLock, watch};

use assistd_memory::{BranchId, SessionId};

/// The active session, the branch it lives on once its first message is
/// saved, and whether its recall is still pending; always read and replaced together.
#[derive(Debug)]
pub struct SessionContext {
    inner: RwLock<SessionContextInner>,
    /// Lets holders that cannot await the lock read the session
    /// synchronously.
    session: watch::Sender<SessionId>,
}

impl SessionContext {
    /// Point at `branch_id` of `session_id`; `None` means the session has
    /// no rows yet and is saved with its first message. Only such a new
    /// session has recall pending.
    pub fn new(session_id: SessionId, branch_id: Option<BranchId>) -> Self {
        Self {
            inner: RwLock::new(SessionContextInner {
                session_id,
                branch_id,
                recall_pending: branch_id.is_none(),
            }),
            session: watch::channel(session_id).0,
        }
    }

    /// Watch the active session id.
    pub fn session_updates(&self) -> watch::Receiver<SessionId> {
        self.session.subscribe()
    }

    /// The active session and its branch, `None` while the session is unsaved.
    pub async fn current(&self) -> (SessionId, Option<BranchId>) {
        let active = self.inner.read().await;
        (active.session_id, active.branch_id)
    }

    /// Make `branch_id` of `session_id` active and notify
    /// [`Self::session_updates`] watchers. Switching to a new session makes
    /// recall pending; resuming a saved one does not.
    pub async fn replace(&self, session_id: SessionId, branch_id: Option<BranchId>) {
        let mut active = self.inner.write().await;
        if active.session_id != session_id {
            active.recall_pending = branch_id.is_none();
        }
        active.session_id = session_id;
        active.branch_id = branch_id;
        self.session.send_replace(session_id);
    }

    /// Whether the active session has not yet recalled past context.
    pub async fn recall_pending(&self) -> bool {
        self.inner.read().await.recall_pending
    }

    /// Mark the active session's recall as done, returning whether it was
    /// still pending, so only one caller recalls per session.
    pub async fn take_pending_recall(&self) -> bool {
        std::mem::take(&mut self.inner.write().await.recall_pending)
    }
}

#[derive(Debug)]
struct SessionContextInner {
    session_id: SessionId,
    branch_id: Option<BranchId>,
    recall_pending: bool,
}
