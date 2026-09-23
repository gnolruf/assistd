//! Branch and session handlers, plus the session-title generator.

use super::{AppState, send_error, wire_role};
use assistd_ipc::Event;
use assistd_llm::{HistoryEntry, HistoryRole};
use assistd_memory::{BranchId, HistoryRow, PersistedRole, SessionId};
use std::sync::Arc;
use tokio::sync::mpsc;

fn persisted_role_to_history_role(role: PersistedRole) -> HistoryRole {
    match role {
        PersistedRole::System => HistoryRole::System,
        PersistedRole::User => HistoryRole::User,
        PersistedRole::Assistant => HistoryRole::Assistant,
        PersistedRole::Tool => HistoryRole::Tool,
    }
}

/// Rows of a branch, as the LLM backend's history type.
pub fn history_entries(rows: &[HistoryRow]) -> Vec<HistoryEntry> {
    rows.iter()
        .map(|r| HistoryEntry {
            role: persisted_role_to_history_role(r.role),
            content: r.content.clone(),
            tool_calls_json: r.tool_calls.clone(),
            tool_call_id: r.tool_call_id.clone(),
            tool_name: r.tool_name.clone(),
        })
        .collect()
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
    /// `/fork <name>`: snapshot the current branch into a new branch and
    /// switch to it. The new branch shares conversation rows with its
    /// parent; only the branch membership is copied.
    #[tracing::instrument(skip_all, fields(correlation_id = %id, branch = %name))]
    pub(super) async fn handle_fork(
        self: Arc<Self>,
        id: String,
        name: String,
        tx: mpsc::Sender<Event>,
    ) {
        if name.trim().is_empty() {
            send_error(&tx, id, "/fork: name must not be empty".into()).await;
            return;
        }
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
                send_error(&tx, id, format!("/fork: {e}")).await;
                return;
            }
        };
        if let Err(e) = self
            .memory
            .conversations
            .set_current_branch(&session, new_branch)
            .await
        {
            send_error(
                &tx,
                id,
                format!("/fork: failed to update current branch: {e}"),
            )
            .await;
            return;
        }
        self.runtime
            .conversation_ctx
            .replace(session.clone(), new_branch)
            .await;

        let (parent_name, _, _) = self.lookup_branch_meta(current_branch).await;
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
    }

    /// Enumerate every branch across every session, active session first.
    #[tracing::instrument(skip_all, fields(correlation_id = %id))]
    pub(super) async fn handle_branches(self: Arc<Self>, id: String, tx: mpsc::Sender<Event>) {
        let branches = match self.memory.conversations.list_branches().await {
            Ok(v) => v,
            Err(e) => {
                send_error(&tx, id, format!("/resume: {e}")).await;
                return;
            }
        };
        let (active_session, _) = self.runtime.conversation_ctx.current().await;
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
    }

    /// If `session` has no title yet, ask the LLM for one in the
    /// background, persist it, and broadcast [`Event::SessionTitle`].
    /// Failures are logged and retried on the session's next turn.
    pub(super) fn spawn_session_title_generation(
        self: Arc<Self>,
        id: String,
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
            let raw = match self
                .subsystems
                .llm
                .complete_oneshot(prompt, assistd_llm::Thinking::Disabled)
                .await
            {
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
                tracing::debug!(
                    target: "assistd::chat",
                    raw_len = raw.len(),
                    "title generation produced no usable text; session stays untitled"
                );
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
                return;
            }
            let _ = self.runtime.events_bus().send(Event::SessionTitle {
                id,
                session_id: session.0.clone(),
                title,
            });
        });
    }

    /// `/switch <target>`: make the target branch active, replay its
    /// history into the LLM backend, and stream it to the client.
    #[tracing::instrument(skip_all, fields(correlation_id = %id, target = %target))]
    pub(super) async fn handle_switch(
        self: Arc<Self>,
        id: String,
        target: String,
        tx: mpsc::Sender<Event>,
    ) {
        let _agent_guard = self.runtime.agent_turn_lock.clone().lock_owned().await;
        self.drain_persistence_inflight().await;

        let (active_session, _active_branch) = self.runtime.conversation_ctx.current().await;
        let (target_session, target_branch) = match self
            .memory
            .conversations
            .resolve_branch(&target, Some(&active_session))
            .await
        {
            Ok(Some(pair)) => pair,
            Ok(None) => {
                send_error(&tx, id, format!("/switch: no branch named {target:?}")).await;
                return;
            }
            Err(e) => {
                send_error(&tx, id, format!("/switch: {e}")).await;
                return;
            }
        };

        if let Err(e) = self
            .memory
            .conversations
            .set_current_branch(&target_session, target_branch)
            .await
        {
            send_error(
                &tx,
                id,
                format!("/switch: failed to update current branch: {e}"),
            )
            .await;
            return;
        }
        let target_session = Arc::new(target_session);
        self.runtime
            .conversation_ctx
            .replace(target_session.clone(), target_branch)
            .await;

        self.replay_branch(id, &target_session, target_branch, &tx, "/switch")
            .await;
    }

    /// `/undo`: drop the latest user prompt and the assistant reply that
    /// followed it from the current branch.
    #[tracing::instrument(skip_all, fields(correlation_id = %id))]
    pub(super) async fn handle_undo(self: Arc<Self>, id: String, tx: mpsc::Sender<Event>) {
        let _agent_guard = self.runtime.agent_turn_lock.clone().lock_owned().await;
        self.drain_persistence_inflight().await;
        let (_, branch) = self.runtime.conversation_ctx.current().await;
        let outcome = match self.memory.conversations.undo_last_turn(branch).await {
            Ok(o) => o,
            Err(e) => {
                send_error(&tx, id, format!("/undo: {e}")).await;
                return;
            }
        };
        if outcome.removed_messages > 0
            && let Err(e) = self.subsystems.llm.truncate_to_last_real_user().await
        {
            tracing::warn!(
                target: "assistd::state",
                error = %e,
                "truncate_to_last_real_user failed (non-fatal)"
            );
        }
        let _ = tx
            .send(Event::UndoApplied {
                id: id.clone(),
                removed_messages: outcome.removed_messages,
                last_user_text: outcome.last_user_text,
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
    }

    /// Keep the current branch and stream its history if its latest
    /// message landed within `recency_secs`; otherwise begin a fresh
    /// session.
    #[tracing::instrument(skip_all, fields(correlation_id = %id, recency_secs = recency_secs))]
    pub(super) async fn handle_resume_or_new(
        self: Arc<Self>,
        id: String,
        recency_secs: u64,
        tx: mpsc::Sender<Event>,
    ) {
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
        let window = i64::try_from(recency_secs)
            .ok()
            .and_then(chrono::Duration::try_seconds)
            .unwrap_or(chrono::Duration::MAX);
        let keep_current = match latest.as_deref() {
            None => true,
            Some(s) => chrono::DateTime::parse_from_rfc3339(s)
                .ok()
                .map(|t| {
                    let age =
                        chrono::Utc::now().signed_duration_since(t.with_timezone(&chrono::Utc));
                    age >= chrono::Duration::zero() && age <= window
                })
                .unwrap_or(false),
        };

        if keep_current {
            self.replay_branch(id, &session, branch, &tx, "/resume")
                .await;
        } else {
            self.begin_fresh_session(id, &tx, "/resume").await;
        }
    }

    /// `/new`: begin a fresh session with an empty `main` branch.
    #[tracing::instrument(skip_all, fields(correlation_id = %id))]
    pub(super) async fn handle_new_session(self: Arc<Self>, id: String, tx: mpsc::Sender<Event>) {
        let _agent_guard = self.runtime.agent_turn_lock.clone().lock_owned().await;
        self.drain_persistence_inflight().await;
        self.begin_fresh_session(id, &tx, "/new").await;
    }

    /// Load `branch`, replace the LLM history with it, and stream it to
    /// the client as `BranchSwitched`, `HistoryEntry` rows, and `Done`.
    async fn replay_branch(
        &self,
        id: String,
        session: &SessionId,
        branch: BranchId,
        tx: &mpsc::Sender<Event>,
        label: &str,
    ) {
        let rows = match self.memory.conversations.load_branch_history(branch).await {
            Ok(v) => v,
            Err(e) => {
                send_error(tx, id, format!("{label}: load_branch_history: {e}")).await;
                return;
            }
        };
        if let Err(e) = self
            .subsystems
            .llm
            .replace_history(history_entries(&rows))
            .await
        {
            tracing::warn!(
                target: "assistd::state",
                error = %e,
                "replace_history failed during {label} (non-fatal)"
            );
        }

        let (branch_name, parent_name, fork_point_seq) = self.lookup_branch_meta(branch).await;
        let session_title = self
            .memory
            .conversations
            .get_session_title(session)
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
                    role: wire_role(r.role),
                    content: r.content,
                    tool_name: r.tool_name,
                })
                .await;
        }
        let _ = tx.send(Event::Done { id }).await;
    }

    /// Start a new session with an empty `main` branch, make it active,
    /// clear the LLM history, and report `BranchSwitched` then `Done`.
    async fn begin_fresh_session(&self, id: String, tx: &mpsc::Sender<Event>, label: &str) {
        let (new_session, new_branch) = match self
            .memory
            .conversations
            .begin_session_with_main_branch(std::process::id())
            .await
        {
            Ok(pair) => pair,
            Err(e) => {
                send_error(
                    tx,
                    id,
                    format!("{label}: begin_session_with_main_branch: {e}"),
                )
                .await;
                return;
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
                "replace_history(empty) failed during {label} (non-fatal)"
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

    /// `(name, parent name, fork point)` of `branch`, all `None` when the
    /// listing fails or the branch is unknown.
    pub(super) async fn lookup_branch_meta(
        &self,
        branch: BranchId,
    ) -> (Option<String>, Option<String>, Option<i64>) {
        let Ok(branches) = self.memory.conversations.list_branches().await else {
            return (None, None, None);
        };
        branches
            .into_iter()
            .find(|b| b.branch_id == branch)
            .map(|b| (Some(b.name), b.parent_branch_name, b.fork_point_seq))
            .unwrap_or((None, None, None))
    }
}
