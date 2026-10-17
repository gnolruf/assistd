//! Branch and session handlers, plus the session-title generator.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, TimeDelta, Utc};
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::{debug, warn};

use assistd_ipc::Event;
use assistd_llm::{HistoryEntry, HistoryRole, Thinking, ToolCallRecord};
use assistd_memory::{
    BranchId, BranchSummary, HistoryRow, MessageKind, PersistedToolCall, Seq, SessionId,
    UndoOutcome,
};

use super::{AppState, send_error, wire_role, wire_timestamp};

const MAX_TITLE_CHARS: usize = 80;
const MAX_TITLE_PROMPT_CHARS: usize = 1024;

/// A branch's persisted history as the LLM replays it: the compaction
/// summary, if any, then only the rows after the ones it stands in for.
#[derive(Debug, Default)]
pub struct LlmReplay {
    pub summary: Option<String>,
    pub entries: Vec<HistoryEntry>,
}

impl LlmReplay {
    /// The replay of `rows`, a whole branch, under its `summary`.
    pub fn new(rows: &[HistoryRow], summary: Option<BranchSummary>) -> Self {
        let through_seq = summary.as_ref().map(|s| s.through_seq);
        Self {
            summary: summary.map(|s| s.content),
            entries: rows
                .iter()
                .filter(|row| through_seq.is_none_or(|through| row.seq > through))
                .map(history_entry)
                .collect(),
        }
    }
}

impl AppState {
    /// `/fork <name>`: snapshot the current branch into a new branch and
    /// switch to it. The new branch shares message rows with its
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

        let (session, current_branch) = match self.saved_session().await {
            Ok(pair) => pair,
            Err(e) => {
                send_error(&tx, id, format!("/fork: {e}")).await;
                return;
            }
        };
        let new_branch = match self
            .memory
            .transcripts
            .fork_and_switch(current_branch, &name)
            .await
        {
            Ok(branch) => branch,
            Err(e) => {
                send_error(&tx, id, format!("/fork: {e}")).await;
                return;
            }
        };
        self.runtime
            .session_ctx
            .replace(session, Some(new_branch))
            .await;

        let (_, parent_name, fork_point_seq) = self.lookup_branch_meta(new_branch).await;
        let session_title = self
            .memory
            .transcripts
            .session_title(session)
            .await
            .ok()
            .flatten();
        let _ = tx
            .send(Event::BranchSwitched {
                id: id.clone(),
                branch_id: Some(new_branch.0),
                session_id: session.to_string(),
                session_title,
                name,
                parent_branch_name: parent_name,
                fork_point_seq: fork_point_seq.map(|seq| seq.0),
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
    }

    /// Enumerate every branch across every session, active session first.
    #[tracing::instrument(skip_all, fields(correlation_id = %id))]
    pub(super) async fn handle_branches(self: Arc<Self>, id: String, tx: mpsc::Sender<Event>) {
        let branches = match self.memory.transcripts.list_branches().await {
            Ok(branches) => branches,
            Err(e) => {
                send_error(&tx, id, format!("/resume: {e}")).await;
                return;
            }
        };
        let (active_session, _) = self.runtime.session_ctx.current().await;
        let (active, other): (Vec<_>, Vec<_>) = branches
            .into_iter()
            .partition(|branch| branch.session_id == active_session);
        for branch in active.into_iter().chain(other) {
            let is_active_session = branch.session_id == active_session;
            let _ = tx
                .send(Event::BranchInfo {
                    id: id.clone(),
                    branch_id: branch.branch_id.0,
                    session_id: branch.session_id.to_string(),
                    session_started_at: wire_timestamp(branch.session_started_at),
                    session_ended_at: branch.session_ended_at.map(wire_timestamp),
                    session_title: branch.session_title,
                    name: branch.name,
                    parent_branch_name: branch.parent_branch_name,
                    fork_point_seq: branch.fork_point_seq.map(|seq| seq.0),
                    created_at: wire_timestamp(branch.created_at),
                    message_count: branch.message_count,
                    is_current_in_session: branch.is_current_in_session,
                    is_active_session,
                    last_active_at: wire_timestamp(branch.last_active_at),
                })
                .await;
        }
        let _ = tx.send(Event::Done { id }).await;
    }

    /// Broadcast `session`'s title as [`Event::SessionTitle`] for turn `id`,
    /// in the background. An untitled session first gets one from the LLM;
    /// failures are logged and retried on the session's next turn.
    pub(super) fn spawn_session_title_broadcast(
        self: Arc<Self>,
        id: String,
        session: SessionId,
        user_text: &str,
    ) {
        let trimmed: String = user_text.chars().take(MAX_TITLE_PROMPT_CHARS).collect();
        self.runtime.persistence_tracker.clone().spawn(async move {
            let Some(title) = self.session_title_for_turn(session, &trimmed).await else {
                return;
            };
            let _ = self.runtime.events_bus().send(Event::SessionTitle {
                id,
                session_id: session.to_string(),
                title,
            });
        });
    }

    /// `session`'s stored title, else a newly generated and saved one.
    /// `None` when any step fails.
    async fn session_title_for_turn(&self, session: SessionId, user_text: &str) -> Option<String> {
        match self.memory.transcripts.session_title(session).await {
            Ok(Some(title)) => return Some(title),
            Ok(None) => {}
            Err(e) => {
                debug!(
                    target: "assistd::memory",
                    error = %e,
                    "session_title failed; skipping the session title"
                );
                return None;
            }
        }
        let title = self.generate_title(user_text).await?;
        if let Err(e) = self
            .memory
            .transcripts
            .set_session_title(session, &title)
            .await
        {
            warn!(
                target: "assistd::memory",
                error = %e,
                "set_session_title failed"
            );
            return None;
        }
        Some(title)
    }

    /// Ask the LLM for a title summarising `user_text`. `None` when the
    /// call fails or yields no usable text.
    async fn generate_title(&self, user_text: &str) -> Option<String> {
        let prompt = format!(
            "Summarize this conversation in 4 to 6 words for use as a UI title. \
                Reply with only the title — no quotes, no punctuation, no leading verbs \
                like \"chat about\". Conversation:\n\n{user_text}"
        );
        let raw = match self
            .subsystems
            .llm
            .complete_oneshot(prompt, Thinking::Disabled)
            .await
        {
            Ok(raw) => raw,
            Err(e) => {
                debug!(
                    target: "assistd::chat",
                    error = %e,
                    "title generation LLM call failed"
                );
                return None;
            }
        };
        let title = clean_generated_title(&raw);
        if title.is_empty() {
            debug!(
                target: "assistd::chat",
                raw_len = raw.len(),
                "title generation produced no usable text; session stays untitled"
            );
            return None;
        }
        Some(title)
    }

    /// Make the `target` branch active, replay its
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

        let (active_session, _active_branch) = self.runtime.session_ctx.current().await;
        let (target_session, target_branch) = match self
            .memory
            .transcripts
            .resolve_branch(&target, Some(active_session))
            .await
        {
            Ok(Some(pair)) => pair,
            Ok(None) => {
                send_error(&tx, id, format!("/resume: no branch named {target:?}")).await;
                return;
            }
            Err(e) => {
                send_error(&tx, id, format!("/resume: {e}")).await;
                return;
            }
        };

        if let Err(e) = self
            .memory
            .transcripts
            .set_current_branch(target_session, target_branch)
            .await
        {
            send_error(
                &tx,
                id,
                format!("/resume: failed to update current branch: {e}"),
            )
            .await;
            return;
        }
        self.runtime
            .session_ctx
            .replace(target_session, Some(target_branch))
            .await;

        self.replay_branch(id, target_session, target_branch, &tx, "/resume")
            .await;
    }

    /// `/undo`: drop the latest user prompt and the assistant reply that
    /// followed it from the current branch.
    #[tracing::instrument(skip_all, fields(correlation_id = %id))]
    pub(super) async fn handle_undo(self: Arc<Self>, id: String, tx: mpsc::Sender<Event>) {
        let _agent_guard = self.runtime.agent_turn_lock.clone().lock_owned().await;
        self.drain_persistence_inflight().await;
        let (_, branch) = self.runtime.session_ctx.current().await;
        let outcome = match branch {
            None => UndoOutcome::default(),
            Some(branch) => match self.memory.transcripts.undo_last_turn(branch).await {
                Ok(outcome) => outcome,
                Err(e) => {
                    send_error(&tx, id, format!("/undo: {e}")).await;
                    return;
                }
            },
        };
        if outcome.removed_messages > 0
            && let Err(e) = self.subsystems.llm.truncate_to_last_real_user().await
        {
            warn!(
                target: "assistd::state",
                error = %e,
                "truncate_to_last_real_user failed (non-fatal)"
            );
        }
        let _ = tx
            .send(Event::UndoApplied {
                id: id.clone(),
                removed_messages: u32::try_from(outcome.removed_messages).unwrap_or(u32::MAX),
                last_user_text: outcome.last_user_text,
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
    }

    /// Keep the current branch and stream its history if its latest
    /// message landed within `recency_secs`; otherwise begin a fresh
    /// session. An unsaved session is kept as is.
    #[tracing::instrument(skip_all, fields(correlation_id = %id, recency_secs = recency_secs))]
    pub(super) async fn handle_resume_or_new(
        self: Arc<Self>,
        id: String,
        recency_secs: u64,
        tx: mpsc::Sender<Event>,
    ) {
        let _agent_guard = self.runtime.agent_turn_lock.clone().lock_owned().await;
        self.drain_persistence_inflight().await;

        let (session, branch) = self.runtime.session_ctx.current().await;
        let Some(branch) = branch else {
            announce_unsaved_session(id, session, &tx).await;
            return;
        };
        let latest = self
            .memory
            .transcripts
            .latest_branch_activity(branch)
            .await
            .ok()
            .flatten();
        if is_within_recency(latest, recency_secs) {
            self.replay_branch(id, session, branch, &tx, "/resume")
                .await;
        } else {
            self.begin_unsaved_session(id, &tx, "/resume").await;
        }
    }

    /// `/new`: begin a fresh session, saved with its first message.
    #[tracing::instrument(skip_all, fields(correlation_id = %id))]
    pub(super) async fn handle_new_session(self: Arc<Self>, id: String, tx: mpsc::Sender<Event>) {
        let _agent_guard = self.runtime.agent_turn_lock.clone().lock_owned().await;
        self.drain_persistence_inflight().await;
        self.begin_unsaved_session(id, &tx, "/new").await;
    }

    /// Load `branch`, replace the LLM history with it, and stream it to
    /// the client as `BranchSwitched`, `HistoryEntry` rows, and `Done`.
    async fn replay_branch(
        &self,
        id: String,
        session: SessionId,
        branch: BranchId,
        tx: &mpsc::Sender<Event>,
        label: &str,
    ) {
        let rows = match self.memory.transcripts.load_branch_history(branch).await {
            Ok(rows) => rows,
            Err(e) => {
                send_error(tx, id, format!("{label}: load_branch_history: {e}")).await;
                return;
            }
        };
        let summary = self.load_branch_summary(branch, label).await;
        let compaction = summary.as_ref().map(|s| (s.through_seq, s.duration_ms));
        let replay = LlmReplay::new(&rows, summary);
        if let Err(e) = self
            .subsystems
            .llm
            .replace_history(replay.summary, replay.entries)
            .await
        {
            warn!(
                target: "assistd::state",
                error = %e,
                "replace_history failed during {label} (non-fatal)"
            );
        }

        let (branch_name, parent_name, fork_point_seq) = self.lookup_branch_meta(branch).await;
        let session_title = self
            .memory
            .transcripts
            .session_title(session)
            .await
            .ok()
            .flatten();
        let _ = tx
            .send(Event::BranchSwitched {
                id: id.clone(),
                branch_id: Some(branch.0),
                session_id: session.to_string(),
                session_title,
                name: branch_name.unwrap_or_default(),
                parent_branch_name: parent_name,
                fork_point_seq: fork_point_seq.map(|seq| seq.0),
            })
            .await;
        let mut call_arguments = tool_call_arguments(&rows);
        for row in rows {
            let role = wire_role(row.message.role());
            let (reasoning, tool_name, tool_args, outcome) = match row.message.kind {
                MessageKind::Tool {
                    call_id,
                    tool_name,
                    outcome,
                } => (
                    None,
                    Some(tool_name),
                    call_arguments.remove(&call_id),
                    Some(outcome),
                ),
                MessageKind::Assistant { reasoning, .. } => (reasoning, None, None, None),
                MessageKind::User => (None, None, None, None),
            };
            let reasoning_ms = reasoning.as_ref().map(|r| r.duration_ms);
            let _ = tx
                .send(Event::HistoryEntry {
                    id: id.clone(),
                    seq: row.seq.0,
                    role,
                    content: row.message.content,
                    reasoning: reasoning.map(|r| r.text).unwrap_or_default(),
                    reasoning_ms,
                    tool_name,
                    tool_args,
                    exit_code: outcome.map(|o| o.exit_code),
                    duration_ms: outcome.map(|o| o.duration_ms),
                    compaction_ms: compaction
                        .filter(|(through_seq, _)| *through_seq == row.seq)
                        .map(|(_, duration_ms)| duration_ms),
                })
                .await;
        }
        let _ = tx.send(Event::Done { id }).await;
    }

    /// The branch's compaction summary; `None` when there is none or the
    /// lookup fails, which only costs a fresh compaction later.
    async fn load_branch_summary(&self, branch: BranchId, label: &str) -> Option<BranchSummary> {
        self.memory
            .transcripts
            .load_branch_summary(branch)
            .await
            .inspect_err(|e| {
                warn!(
                    target: "assistd::state",
                    error = %e,
                    "load_branch_summary failed during {label}; replaying without it"
                );
            })
            .ok()
            .flatten()
    }

    /// Make a new, unsaved session active, clear the LLM history, and
    /// report `BranchSwitched` then `Done`. The session and its `main`
    /// branch are inserted with its first message.
    async fn begin_unsaved_session(&self, id: String, tx: &mpsc::Sender<Event>, label: &str) {
        let new_session = SessionId::new();
        self.runtime.session_ctx.replace(new_session, None).await;
        if let Err(e) = self.subsystems.llm.replace_history(None, Vec::new()).await {
            warn!(
                target: "assistd::state",
                error = %e,
                "replace_history(empty) failed during {label} (non-fatal)"
            );
        }
        announce_unsaved_session(id, new_session, tx).await;
    }

    /// `(name, parent name, fork point)` of `branch`, all `None` when the
    /// lookup fails or the branch is unknown.
    pub(super) async fn lookup_branch_meta(
        &self,
        branch: BranchId,
    ) -> (Option<String>, Option<String>, Option<Seq>) {
        let Ok(Some(info)) = self.memory.transcripts.branch_info(branch).await else {
            return (None, None, None);
        };
        (
            Some(info.name),
            info.parent_branch_name,
            info.fork_point_seq,
        )
    }
}

/// Report the unsaved `session` as active on its future `main` branch:
/// `BranchSwitched` without a branch id, then `Done`.
async fn announce_unsaved_session(id: String, session: SessionId, tx: &mpsc::Sender<Event>) {
    let _ = tx
        .send(Event::BranchSwitched {
            id: id.clone(),
            branch_id: None,
            session_id: session.to_string(),
            session_title: None,
            name: "main".to_string(),
            parent_branch_name: None,
            fork_point_seq: None,
        })
        .await;
    let _ = tx.send(Event::Done { id }).await;
}

fn history_entry(row: &HistoryRow) -> HistoryEntry {
    let content = row.message.content.clone();
    match &row.message.kind {
        MessageKind::User => HistoryEntry {
            role: HistoryRole::User,
            content,
            reasoning: String::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_name: None,
        },
        MessageKind::Assistant {
            reasoning,
            tool_calls,
        } => HistoryEntry {
            role: HistoryRole::Assistant,
            content,
            reasoning: reasoning
                .as_ref()
                .map(|r| r.text.clone())
                .unwrap_or_default(),
            tool_calls: tool_calls.iter().map(tool_call_record).collect(),
            tool_call_id: None,
            tool_name: None,
        },
        MessageKind::Tool {
            call_id, tool_name, ..
        } => HistoryEntry {
            role: HistoryRole::Tool,
            content,
            reasoning: String::new(),
            tool_calls: Vec::new(),
            tool_call_id: Some(call_id.clone()),
            tool_name: Some(tool_name.clone()),
        },
    }
}

/// `call` as the LLM replays it, with its arguments as the JSON text the model sent.
fn tool_call_record(call: &PersistedToolCall) -> ToolCallRecord {
    let arguments = match &call.arguments {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    };
    ToolCallRecord {
        id: call.id.clone(),
        name: call.name.clone(),
        arguments,
    }
}

/// The arguments of every tool call recorded on `rows`, by call id.
fn tool_call_arguments(rows: &[HistoryRow]) -> HashMap<String, Value> {
    rows.iter()
        .flat_map(|row| match &row.message.kind {
            MessageKind::Assistant { tool_calls, .. } => tool_calls.as_slice(),
            MessageKind::User | MessageKind::Tool { .. } => &[],
        })
        .map(|call| (call.id.clone(), call.arguments.clone()))
        .collect()
}

pub(super) fn clean_generated_title(raw: &str) -> String {
    let first_line = raw
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("");
    let stripped = first_line
        .trim_matches(|c: char| matches!(c, '"' | '\'' | '`' | '*' | '_' | '#' | ' ' | '\t' | '.'));
    stripped.chars().take(MAX_TITLE_CHARS).collect::<String>()
}

/// Whether the `latest` activity is no older than `recency_secs`.
/// A branch with no activity counts as recent; a future timestamp does not.
fn is_within_recency(latest: Option<DateTime<Utc>>, recency_secs: u64) -> bool {
    let window = i64::try_from(recency_secs)
        .ok()
        .and_then(TimeDelta::try_seconds)
        .unwrap_or(TimeDelta::MAX);
    match latest {
        None => true,
        Some(at) => {
            let age = Utc::now().signed_duration_since(at);
            age >= TimeDelta::zero() && age <= window
        }
    }
}
