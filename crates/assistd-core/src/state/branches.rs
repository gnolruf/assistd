//! Handlers for `/fork`, `/branches`, `/switch`, `/undo`, `/resume`,
//! `/new`, plus the LLM-driven session-title generator and a handful
//! of small lookup helpers used by the above.

use super::AppState;
use anyhow::Result;
use assistd_ipc::Event;
use assistd_memory::{BranchId, PersistedRole, SessionId};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Translate a [`assistd_memory::PersistedRole`] into the equivalent
/// [`assistd_llm::HistoryRole`] for branch-history replay. Identity
/// mapping; lives in this crate because `assistd-llm` doesn't depend
/// on `assistd-memory`.
fn persisted_role_to_history_role(role: PersistedRole) -> assistd_llm::HistoryRole {
    match role {
        PersistedRole::System => assistd_llm::HistoryRole::System,
        PersistedRole::User => assistd_llm::HistoryRole::User,
        PersistedRole::Assistant => assistd_llm::HistoryRole::Assistant,
        PersistedRole::Tool => assistd_llm::HistoryRole::Tool,
    }
}

pub(super) fn clean_generated_title(raw: &str) -> String {
    const MAX_TITLE_CHARS: usize = 80;
    let first_line = raw
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    let stripped = first_line
        .trim_matches(|c: char| matches!(c, '"' | '\'' | '`' | '*' | '_' | '#' | ' ' | '\t' | '.'));
    stripped.chars().take(MAX_TITLE_CHARS).collect::<String>()
}

impl AppState {
    /// `/fork <name>`: snapshot the current branch into a new branch
    /// and switch to it. The shared agent_turn_lock + a drain of
    /// `persistence_tracker` guarantees no in-flight turn races the
    /// snapshot. The new branch shares conversation rows with its
    /// parent (no row duplication); only branch_messages get copied.
    #[tracing::instrument(skip_all, fields(correlation_id = %id, branch = %name))]
    pub(super) async fn handle_fork(
        self: Arc<Self>,
        id: String,
        name: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        if name.trim().is_empty() {
            let _ = tx
                .send(Event::Error {
                    id,
                    message: "/fork: name must not be empty".into(),
                })
                .await;
            return Ok(());
        }
        // Lock first, then drain; order matters: holding the lock
        // prevents new persistence tasks from being spawned, then we
        // wait for already-spawned ones to finish.
        let _agent_guard = self.runtime.agent_turn_lock.clone().lock_owned().await;
        self.drain_persistence_inflight().await;

        let (session, current_branch) = self.runtime.conversation_ctx.current().await;
        let new_branch = match self
            .memory
            .conversations
            .fork_branch(current_branch, &name)
            .await
        {
            Ok(b) => b,
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("/fork: {e:#}"),
                    })
                    .await;
                return Ok(());
            }
        };
        if let Err(e) = self
            .memory
            .conversations
            .set_current_branch(&session, new_branch)
            .await
        {
            let _ = tx
                .send(Event::Error {
                    id,
                    message: format!("/fork: failed to update current branch: {e:#}"),
                })
                .await;
            return Ok(());
        }
        self.runtime
            .conversation_ctx
            .replace(session.clone(), new_branch)
            .await;

        let parent_name = self.lookup_branch_name(current_branch).await;
        let fork_point_seq = self.lookup_branch_tail_seq(current_branch).await;
        let session_title = self
            .memory
            .conversations
            .get_session_title(&session)
            .await
            .ok()
            .flatten();
        let _ = tx
            .send(Event::BranchSwitched {
                id: id.clone(),
                branch_id: new_branch.0,
                session_id: session.0.clone(),
                session_title,
                name,
                parent_branch_name: parent_name,
                fork_point_seq,
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    /// `/resume`: enumerate every branch across every session, with
    /// the active session's branches surfaced first. The TUI feeds the
    /// resulting `BranchInfo` events into its branch picker.
    #[tracing::instrument(skip_all, fields(correlation_id = %id))]
    pub(super) async fn handle_branches(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        let branches = match self.memory.conversations.list_branches().await {
            Ok(v) => v,
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("/resume: {e:#}"),
                    })
                    .await;
                return Ok(());
            }
        };
        let (active_session, _) = self.runtime.conversation_ctx.current().await;
        // Active-session-first ordering: keep the original list_branches
        // sort but pull active-session branches to the top.
        let mut active = Vec::new();
        let mut other = Vec::new();
        for b in branches {
            if b.session_id == active_session.0 {
                active.push(b);
            } else {
                other.push(b);
            }
        }
        for b in active.into_iter().chain(other) {
            let is_active_session = b.session_id == active_session.0;
            let _ = tx
                .send(Event::BranchInfo {
                    id: id.clone(),
                    branch_id: b.branch_id.0,
                    session_id: b.session_id,
                    session_started_at: b.session_started_at,
                    session_ended_at: b.session_ended_at,
                    session_title: b.session_title,
                    name: b.name,
                    parent_branch_name: b.parent_branch_name,
                    fork_point_seq: b.fork_point_seq,
                    created_at: b.created_at,
                    message_count: b.message_count,
                    is_current_in_session: b.is_current_in_session,
                    is_active_session,
                })
                .await;
        }
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    /// Background hook: if `session` has no `title` yet, ask the LLM
    /// for a short summary of `user_text` and write it back via
    /// `set_session_title`. Spawns onto `persistence_tracker` so daemon
    /// shutdown drains the task; survives `complete_oneshot` failures
    /// silently because a missing title is a UX downgrade, not a bug.
    pub(super) fn spawn_session_title_generation(
        self: Arc<Self>,
        session: Arc<SessionId>,
        user_text: String,
    ) {
        const MAX_PROMPT_CHARS: usize = 1024;
        let trimmed: String = user_text.chars().take(MAX_PROMPT_CHARS).collect();
        self.runtime.persistence_tracker.clone().spawn(async move {
            match self.memory.conversations.get_session_title(&session).await {
                Ok(Some(_)) => return,
                Ok(None) => {}
                Err(e) => {
                    tracing::debug!(
                        target: "assistd::memory",
                        error = %e,
                        "get_session_title failed; skipping title generation"
                    );
                    return;
                }
            }
            let prompt = format!(
                "Summarize this conversation in 4 to 6 words for use as a UI title. \
                Reply with only the title — no quotes, no punctuation, no leading verbs \
                like \"chat about\". Conversation:\n\n{trimmed}"
            );
            let raw = match self.subsystems.llm.complete_oneshot(prompt).await {
                Ok(s) => s,
                Err(e) => {
                    tracing::debug!(
                        target: "assistd::chat",
                        error = %e,
                        "title generation LLM call failed"
                    );
                    return;
                }
            };
            let title = clean_generated_title(&raw);
            if title.is_empty() {
                return;
            }
            if let Err(e) = self
                .memory
                .conversations
                .set_session_title(&session, &title)
                .await
            {
                tracing::warn!(
                    target: "assistd::memory",
                    error = %e,
                    "set_session_title failed"
                );
            }
        });
    }

    /// `/switch <target>`: drain in-flight writes, swap the active
    /// (session, branch) pointer, replay the target branch's history
    /// into the LLM backend, and stream the loaded turns back to the
    /// client so the TUI can repaint the chat pane.
    #[tracing::instrument(skip_all, fields(correlation_id = %id, target = %target))]
    pub(super) async fn handle_switch(
        self: Arc<Self>,
        id: String,
        target: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        let _agent_guard = self.runtime.agent_turn_lock.clone().lock_owned().await;
        self.drain_persistence_inflight().await;

        let (active_session, _active_branch) = self.runtime.conversation_ctx.current().await;
        let resolved = match self
            .memory
            .conversations
            .resolve_branch(&target, Some(&active_session))
            .await
        {
            Ok(Some(pair)) => pair,
            Ok(None) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("/switch: no branch named {target:?}"),
                    })
                    .await;
                return Ok(());
            }
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("/switch: {e:#}"),
                    })
                    .await;
                return Ok(());
            }
        };
        let (target_session, target_branch) = resolved;

        // Update DB pointer first; if the daemon crashes between this
        // and the in-memory swap, the next startup resumes the *new*
        // active branch (consistent), not the old one.
        if let Err(e) = self
            .memory
            .conversations
            .set_current_branch(&target_session, target_branch)
            .await
        {
            let _ = tx
                .send(Event::Error {
                    id,
                    message: format!("/switch: failed to update current branch: {e:#}"),
                })
                .await;
            return Ok(());
        }
        let target_session_arc = Arc::new(target_session.clone());
        self.runtime
            .conversation_ctx
            .replace(target_session_arc.clone(), target_branch)
            .await;

        let rows = match self
            .memory
            .conversations
            .load_branch_history(target_branch)
            .await
        {
            Ok(v) => v,
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("/switch: load_branch_history: {e:#}"),
                    })
                    .await;
                return Ok(());
            }
        };
        let entries: Vec<assistd_llm::HistoryEntry> = rows
            .iter()
            .map(|r| assistd_llm::HistoryEntry {
                role: persisted_role_to_history_role(r.role),
                content: r.content.clone(),
                tool_calls_json: r.tool_calls.clone(),
                tool_call_id: r.tool_call_id.clone(),
                tool_name: r.tool_name.clone(),
            })
            .collect();
        if let Err(e) = self.subsystems.llm.replace_history(entries).await {
            tracing::warn!(
                target: "assistd::state",
                error = %e,
                "replace_history failed (non-fatal)"
            );
        }

        let (branch_name, parent_name, fork_point_seq) =
            self.lookup_branch_meta(target_branch).await;
        let session_title = self
            .memory
            .conversations
            .get_session_title(&target_session)
            .await
            .ok()
            .flatten();

        let _ = tx
            .send(Event::BranchSwitched {
                id: id.clone(),
                branch_id: target_branch.0,
                session_id: target_session.0.clone(),
                session_title,
                name: branch_name.unwrap_or_default(),
                parent_branch_name: parent_name,
                fork_point_seq,
            })
            .await;
        for r in rows {
            let _ = tx
                .send(Event::HistoryEntry {
                    id: id.clone(),
                    seq: r.seq,
                    role: r.role.as_wire().to_string(),
                    content: r.content,
                    tool_name: r.tool_name,
                })
                .await;
        }
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    /// `/undo`: drop the latest user prompt and the entire assistant
    /// reply that followed it from the current branch. Both DB and
    /// in-memory state are updated atomically with the agent_turn_lock
    /// held.
    #[tracing::instrument(skip_all, fields(correlation_id = %id))]
    pub(super) async fn handle_undo(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        let _agent_guard = self.runtime.agent_turn_lock.clone().lock_owned().await;
        self.drain_persistence_inflight().await;
        let (_, branch) = self.runtime.conversation_ctx.current().await;
        let outcome = match self.memory.conversations.undo_last_turn(branch).await {
            Ok(o) => o,
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("/undo: {e:#}"),
                    })
                    .await;
                return Ok(());
            }
        };
        if outcome.removed_messages > 0 {
            // Mirror the deletion in the in-memory conversation. We
            // call `truncate_to_last_real_user` rather than recompute
            // from the DB because the DB is now authoritative; the
            // backend's job is just to align.
            if let Err(e) = self.subsystems.llm.truncate_to_last_real_user().await {
                tracing::warn!(
                    target: "assistd::state",
                    error = %e,
                    "truncate_to_last_real_user failed (non-fatal)"
                );
            }
        }
        let _ = tx
            .send(Event::UndoApplied {
                id: id.clone(),
                removed_messages: outcome.removed_messages,
                last_user_text: outcome.last_user_text,
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    /// TUI-startup branch decision. If the current branch's most
    /// recent message landed within `recency_secs`, keep the branch
    /// and stream its history so the client can repaint the chat
    /// pane. Otherwise, begin a fresh session with an empty `main`
    /// branch, swap it in, and emit a `BranchSwitched` so the client
    /// can clear its output.
    #[tracing::instrument(skip_all, fields(correlation_id = %id, recency_secs = recency_secs))]
    pub(super) async fn handle_resume_or_new(
        self: Arc<Self>,
        id: String,
        recency_secs: u64,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        let _agent_guard = self.runtime.agent_turn_lock.clone().lock_owned().await;
        self.drain_persistence_inflight().await;

        let (session, branch) = self.runtime.conversation_ctx.current().await;
        let latest = self
            .memory
            .conversations
            .latest_branch_activity(branch)
            .await
            .ok()
            .flatten();
        // Keep the current branch when it's either empty (nothing to
        // stale-out) or its latest message landed within the window.
        let keep_current = match latest.as_deref() {
            None => true,
            Some(s) => chrono::DateTime::parse_from_rfc3339(s)
                .ok()
                .map(|t| {
                    let age =
                        chrono::Utc::now().signed_duration_since(t.with_timezone(&chrono::Utc));
                    age >= chrono::Duration::zero()
                        && age <= chrono::Duration::seconds(recency_secs as i64)
                })
                .unwrap_or(false),
        };

        if keep_current {
            let rows = match self.memory.conversations.load_branch_history(branch).await {
                Ok(v) => v,
                Err(e) => {
                    let _ = tx
                        .send(Event::Error {
                            id,
                            message: format!("/resume: load_branch_history: {e:#}"),
                        })
                        .await;
                    return Ok(());
                }
            };
            let entries: Vec<assistd_llm::HistoryEntry> = rows
                .iter()
                .map(|r| assistd_llm::HistoryEntry {
                    role: persisted_role_to_history_role(r.role),
                    content: r.content.clone(),
                    tool_calls_json: r.tool_calls.clone(),
                    tool_call_id: r.tool_call_id.clone(),
                    tool_name: r.tool_name.clone(),
                })
                .collect();
            if let Err(e) = self.subsystems.llm.replace_history(entries).await {
                tracing::warn!(
                    target: "assistd::state",
                    error = %e,
                    "replace_history failed during resume (non-fatal)"
                );
            }
            let (branch_name, parent_name, fork_point_seq) = self.lookup_branch_meta(branch).await;
            let session_title = self
                .memory
                .conversations
                .get_session_title(&session)
                .await
                .ok()
                .flatten();
            let _ = tx
                .send(Event::BranchSwitched {
                    id: id.clone(),
                    branch_id: branch.0,
                    session_id: session.0.clone(),
                    session_title,
                    name: branch_name.unwrap_or_default(),
                    parent_branch_name: parent_name,
                    fork_point_seq,
                })
                .await;
            for r in rows {
                let _ = tx
                    .send(Event::HistoryEntry {
                        id: id.clone(),
                        seq: r.seq,
                        role: r.role.as_wire().to_string(),
                        content: r.content,
                        tool_name: r.tool_name,
                    })
                    .await;
            }
            let _ = tx.send(Event::Done { id }).await;
            return Ok(());
        }

        // Fresh chat: new session + empty main branch.
        let (new_session, new_branch) = match self
            .memory
            .conversations
            .begin_session_with_main_branch(std::process::id())
            .await
        {
            Ok(pair) => pair,
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("/resume: begin_session_with_main_branch: {e:#}"),
                    })
                    .await;
                return Ok(());
            }
        };
        self.runtime
            .conversation_ctx
            .replace(Arc::new(new_session.clone()), new_branch)
            .await;
        if let Err(e) = self.subsystems.llm.replace_history(Vec::new()).await {
            tracing::warn!(
                target: "assistd::state",
                error = %e,
                "replace_history(empty) failed during fresh session (non-fatal)"
            );
        }
        let _ = tx
            .send(Event::BranchSwitched {
                id: id.clone(),
                branch_id: new_branch.0,
                session_id: new_session.0.clone(),
                session_title: None,
                name: "main".to_string(),
                parent_branch_name: None,
                fork_point_seq: None,
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    /// `/new`: unconditionally begin a fresh session with an empty
    /// `main` branch and swap it in. Mirrors the fresh-chat half of
    /// [`Self::handle_resume_or_new`] but without the recency check
    /// so the user always gets a blank canvas.
    #[tracing::instrument(skip_all, fields(correlation_id = %id))]
    pub(super) async fn handle_new_session(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        let _agent_guard = self.runtime.agent_turn_lock.clone().lock_owned().await;
        self.drain_persistence_inflight().await;

        let (new_session, new_branch) = match self
            .memory
            .conversations
            .begin_session_with_main_branch(std::process::id())
            .await
        {
            Ok(pair) => pair,
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("/new: begin_session_with_main_branch: {e:#}"),
                    })
                    .await;
                return Ok(());
            }
        };
        self.runtime
            .conversation_ctx
            .replace(Arc::new(new_session.clone()), new_branch)
            .await;
        if let Err(e) = self.subsystems.llm.replace_history(Vec::new()).await {
            tracing::warn!(
                target: "assistd::state",
                error = %e,
                "replace_history(empty) failed during /new (non-fatal)"
            );
        }
        let _ = tx
            .send(Event::BranchSwitched {
                id: id.clone(),
                branch_id: new_branch.0,
                session_id: new_session.0.clone(),
                session_title: None,
                name: "main".to_string(),
                parent_branch_name: None,
                fork_point_seq: None,
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    pub(super) async fn lookup_branch_name(&self, branch: BranchId) -> Option<String> {
        for b in self.memory.conversations.list_branches().await.ok()? {
            if b.branch_id == branch {
                return Some(b.name);
            }
        }
        None
    }

    pub(super) async fn lookup_branch_tail_seq(&self, branch: BranchId) -> Option<i64> {
        let rows = self
            .memory
            .conversations
            .load_branch_history(branch)
            .await
            .ok()?;
        rows.last().map(|r| r.seq)
    }

    pub(super) async fn lookup_branch_meta(
        &self,
        branch: BranchId,
    ) -> (Option<String>, Option<String>, Option<i64>) {
        let Ok(branches) = self.memory.conversations.list_branches().await else {
            return (None, None, None);
        };
        for b in branches {
            if b.branch_id == branch {
                return (Some(b.name), b.parent_branch_name, b.fork_point_seq);
            }
        }
        (None, None, None)
    }
}
