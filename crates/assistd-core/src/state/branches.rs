//! Branch and session handlers, plus the session-title generator.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, TimeDelta, Utc};
use serde_json::Value;
use tracing::{debug, warn};

use assistd_ipc::{AgentId, Event, EventBody, RelayStepEnd, RelayStepPlan};
use assistd_llm::{
    HistoryEntry, HistoryReplay, HistoryRole, LlmBackend, OneShotReply, SlotHolder, SlotStash,
    Thinking, ToolCallRecord,
};
use assistd_memory::{
    AgentRunEnd, AgentRunRecord, BranchHistory, BranchId, BranchSummary, HistoryRow, HistoryScope,
    MessageId, MessageKind, PersistedAgentRun, PersistedMessage, PersistedRole, PersistedToolCall,
    Seq, SessionId, TranscriptStore, TurnStats, UndoOutcome,
};

use super::{DispatchError, FailedStep, wire_role, wire_timestamp};
use crate::admission::TurnAdmission;
use crate::bus::EventBus;
use crate::bus::EventSink;
use crate::presence::PresenceManager;
use crate::relay::delete_answer_files;
use crate::transcript::{SessionContext, TranscriptLog};

const MAX_TITLE_CHARS: usize = 80;
const MAX_TITLE_PROMPT_CHARS: usize = 1024;

/// Where a replayed history draws its compaction line, and how long the compaction took.
#[derive(Clone, Copy)]
struct CompactionLine {
    after: Seq,
    duration_ms: u64,
}

/// What a replayed message is followed by when its history's compaction line or its
/// turn's end falls on it.
#[derive(Clone, Copy)]
struct ReplayedEnds {
    compaction: Option<CompactionLine>,
    turn_stats: Option<TurnStats>,
}

/// Where a branch row lands when its history is replayed into the LLM.
enum ReplayPart {
    /// Stood in for by the summary.
    Summarized,
    /// A tool step of the kept turn, folded into its question's call log.
    FoldedStep,
    Kept,
}

/// Handles the fork, switch, undo and session requests on the main backend, and
/// titles its sessions.
#[derive(Debug)]
pub struct BranchService {
    pub(super) session_ctx: Arc<SessionContext>,
    pub(super) transcripts: Arc<dyn TranscriptStore>,
    pub(super) transcript: Arc<TranscriptLog>,
    pub(super) admission: Arc<TurnAdmission>,
    pub(super) llm: Arc<dyn LlmBackend>,
    pub(super) presence: Arc<PresenceManager>,
    pub(super) slot_stash: Option<Arc<SlotStash>>,
    pub(super) bus: EventBus,
}

impl BranchService {
    /// `/fork <name>`: snapshot the current branch into a new branch and
    /// switch to it. The new branch shares message rows with its
    /// parent; only the branch membership is copied.
    #[tracing::instrument(skip_all, fields(correlation_id = %id, branch = %name))]
    pub(super) async fn handle_fork(
        &self,
        id: String,
        name: String,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        if name.trim().is_empty() {
            return Err(DispatchError::EmptyBranchName);
        }
        let _user_turn = self.admission.user_turn_lock.clone().lock_owned().await;
        self.transcript.drain_inflight().await;

        let (session, current_branch) = self.transcript.saved_session().await.context("/fork")?;
        let new_branch = self
            .transcripts
            .fork_and_switch(current_branch, &name)
            .await
            .context("/fork")?;
        self.session_ctx.replace(session, Some(new_branch)).await;

        let (_, parent_name, fork_point_seq) = self.lookup_branch_meta(new_branch).await;
        let session_title = self.transcripts.session_title(session).await.ok().flatten();
        let switched = EventBody::BranchSwitched {
            branch_id: Some(new_branch.0),
            session_id: session.to_string(),
            session_title,
            name,
            parent_branch_name: parent_name,
            fork_point_seq: fork_point_seq.map(|seq| seq.0),
        };
        tx.reply(id, switched).await;
        Ok(())
    }

    /// Enumerate every branch across every session, active session first.
    #[tracing::instrument(skip_all, fields(correlation_id = %id))]
    pub(super) async fn handle_branches(
        &self,
        id: String,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let branches = self.transcripts.list_branches().await.context("/resume")?;
        let (active_session, _) = self.session_ctx.current().await;
        let (active, other): (Vec<_>, Vec<_>) = branches
            .into_iter()
            .partition(|branch| branch.session_id == active_session);
        for branch in active.into_iter().chain(other) {
            let is_active_session = branch.session_id == active_session;
            let _ = tx
                .send(Event::new(
                    id.clone(),
                    EventBody::BranchInfo {
                        branch_id: branch.branch_id.0,
                        session_id: branch.session_id.to_string(),
                        session_started_at: wire_timestamp(branch.session_started_at),
                        session_title: branch.session_title,
                        name: branch.name,
                        parent_branch_name: branch.parent_branch_name,
                        fork_point_seq: branch.fork_point_seq.map(|seq| seq.0),
                        created_at: wire_timestamp(branch.created_at),
                        message_count: branch.message_count,
                        is_current_in_session: branch.is_current_in_session,
                        is_active_session,
                        last_active_at: wire_timestamp(branch.last_active_at),
                        task_run: branch.task_run,
                    },
                ))
                .await;
        }
        Ok(())
    }

    /// Broadcast `session`'s title as [`EventBody::SessionTitle`] for turn `id`,
    /// in the background. An untitled session first gets one from the LLM;
    /// failures are logged and retried on the session's next turn.
    pub(super) fn spawn_session_title_broadcast(
        self: &Arc<Self>,
        id: String,
        session: SessionId,
        user_text: &str,
    ) {
        let trimmed: String = user_text.chars().take(MAX_TITLE_PROMPT_CHARS).collect();
        let branches = Arc::clone(self);
        self.transcript.spawn_tracked(async move {
            let Some(title) = branches.session_title_for_turn(session, &trimmed).await else {
                return;
            };
            let _ = branches.bus.sender().send(Event::new(
                id,
                EventBody::SessionTitle {
                    session_id: session.to_string(),
                    title,
                },
            ));
        });
    }

    /// `session`'s stored title, else a newly generated and saved one.
    /// `None` when any step fails.
    async fn session_title_for_turn(&self, session: SessionId, user_text: &str) -> Option<String> {
        match self.transcripts.session_title(session).await {
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
        if let Err(e) = self.transcripts.set_session_title(session, &title).await {
            warn!(
                target: "assistd::memory",
                error = %e,
                "set_session_title failed"
            );
            return None;
        }
        Some(title)
    }

    /// Ask the LLM for a title summarising `user_text`, between turns and
    /// only while `Active`. `None` when asleep, or the call fails or yields
    /// no usable text.
    async fn generate_title(&self, user_text: &str) -> Option<String> {
        let _lane = self.admission.task_queue.queue_title().granted().await?;
        let Some(_request_guard) = self.presence.request_guard_if_active().await else {
            debug!(
                target: "assistd::completion",
                "daemon not active; session stays untitled until its next turn"
            );
            return None;
        };
        let prompt = format!(
            "Summarize this conversation in 4 to 6 words for use as a UI title. \
                Reply with only the title — no quotes, no punctuation, no leading verbs \
                like \"chat about\". Conversation:\n\n{user_text}"
        );
        let raw = match self
            .llm
            .complete_oneshot(prompt, OneShotReply::Text(Thinking::Disabled))
            .await
        {
            Ok(raw) => raw,
            Err(e) => {
                debug!(
                    target: "assistd::completion",
                    error = %e,
                    "title generation LLM call failed"
                );
                return None;
            }
        };
        let title = clean_generated_title(&raw);
        if title.is_empty() {
            debug!(
                target: "assistd::completion",
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
        &self,
        id: String,
        target: String,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let _user_turn = self.admission.user_turn_lock.clone().lock_owned().await;
        self.transcript.drain_inflight().await;

        let (active_session, _active_branch) = self.session_ctx.current().await;
        let (target_session, target_branch) = self
            .transcripts
            .resolve_branch(&target, Some(active_session))
            .await
            .context("/resume")?
            .ok_or(DispatchError::UnknownBranch(target))?;

        self.transcripts
            .activate_branch(target_session, target_branch)
            .await
            .context("/resume: failed to update current branch")?;
        self.session_ctx
            .replace(target_session, Some(target_branch))
            .await;

        self.replay_branch(id, target_session, target_branch, &tx)
            .await
    }

    /// `/undo`: drop the latest user prompt and the assistant reply that
    /// followed it from the current branch.
    #[tracing::instrument(skip_all, fields(correlation_id = %id))]
    pub(super) async fn handle_undo(&self, id: String, tx: EventSink) -> Result<(), DispatchError> {
        let _user_turn = self.admission.user_turn_lock.clone().lock_owned().await;
        self.transcript.drain_inflight().await;
        let (_, branch) = self.session_ctx.current().await;
        let outcome = match branch {
            None => UndoOutcome::default(),
            Some(branch) => self
                .transcripts
                .undo_last_turn(branch)
                .await
                .context("/undo")?,
        };
        if outcome.removed_messages > 0 {
            self.transcript.forget_last_user_turn();
            if let Err(e) = self.llm.truncate_to_last_real_user().await {
                warn!(
                    target: "assistd::state",
                    error = %e,
                    "truncate_to_last_real_user failed (non-fatal)"
                );
            }
        }
        delete_answer_files(outcome.released_answer_files).await;
        let applied = EventBody::UndoApplied {
            removed_messages: u32::try_from(outcome.removed_messages).unwrap_or(u32::MAX),
            last_user_text: outcome.last_user_text,
        };
        tx.reply(id, applied).await;
        Ok(())
    }

    /// Keep the current branch and stream its history if its latest
    /// message landed within `recency_secs`; otherwise begin a fresh
    /// session. An unsaved session is kept as is.
    #[tracing::instrument(skip_all, fields(correlation_id = %id, recency_secs = recency_secs))]
    pub(super) async fn handle_resume_or_new(
        &self,
        id: String,
        recency_secs: u64,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let _user_turn = self.admission.user_turn_lock.clone().lock_owned().await;
        self.transcript.drain_inflight().await;

        let (session, branch) = self.session_ctx.current().await;
        let Some(branch) = branch else {
            report_unsaved_session(id, session, &tx).await;
            return Ok(());
        };
        let latest = self
            .transcripts
            .latest_branch_activity(branch)
            .await
            .ok()
            .flatten();
        if is_within_recency(latest, recency_secs) {
            self.replay_branch(id, session, branch, &tx).await
        } else {
            self.begin_unsaved_session(id, &tx, "/resume").await;
            Ok(())
        }
    }

    /// Stream the current branch's history however old it is. An unsaved
    /// session is reported as is.
    #[tracing::instrument(skip_all, fields(correlation_id = %id))]
    pub(super) async fn handle_replay_branch(
        &self,
        id: String,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let _user_turn = self.admission.user_turn_lock.clone().lock_owned().await;
        self.transcript.drain_inflight().await;

        let (session, branch) = self.session_ctx.current().await;
        match branch {
            Some(branch) => self.replay_branch(id, session, branch, &tx).await,
            None => {
                report_unsaved_session(id, session, &tx).await;
                Ok(())
            }
        }
    }

    /// `/new`: begin a fresh session, saved with its first message.
    #[tracing::instrument(skip_all, fields(correlation_id = %id))]
    pub(super) async fn handle_new_session(
        &self,
        id: String,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        self.admission.interaction.end();
        let _user_turn = self.admission.user_turn_lock.clone().lock_owned().await;
        self.transcript.drain_inflight().await;
        if let Some(stash) = &self.slot_stash {
            stash.end(SlotHolder::Main).await;
        }
        self.begin_unsaved_session(id, &tx, "/new").await;
        Ok(())
    }

    /// Load `branch`, replace the LLM history with it, and stream it to
    /// the client as `BranchSwitched` and `HistoryEntry` rows.
    async fn replay_branch(
        &self,
        id: String,
        session: SessionId,
        branch: BranchId,
        tx: &EventSink,
    ) -> Result<(), DispatchError> {
        let BranchHistory { summary, rows } = self
            .transcripts
            .load_branch(branch, HistoryScope::Whole)
            .await
            .context("/resume: load_branch")?;
        let compaction = summary
            .as_ref()
            .and_then(|s| compaction_line(rows.iter().map(|row| row.seq), s));
        let replay = branch_replay(&rows, summary);
        self.transcript.forget_user_history();
        if let Err(e) = self.llm.replace_history(replay).await {
            warn!(
                target: "assistd::state",
                error = %e,
                "replace_history failed during /resume (non-fatal)"
            );
        }

        let (branch_name, parent_name, fork_point_seq) = self.lookup_branch_meta(branch).await;
        let session_title = self.transcripts.session_title(session).await.ok().flatten();
        let _ = tx
            .send(Event::new(
                id.clone(),
                EventBody::BranchSwitched {
                    branch_id: Some(branch.0),
                    session_id: session.to_string(),
                    session_title,
                    name: branch_name.unwrap_or_default(),
                    parent_branch_name: parent_name,
                    fork_point_seq: fork_point_seq.map(|seq| seq.0),
                },
            ))
            .await;
        let mut call_arguments = tool_call_arguments(rows.iter().map(|row| &row.message));
        let mut agent_runs = self.agent_runs_of(branch).await;
        let stored = tx.unpublishing();
        for row in rows {
            let marker = compaction.filter(|line| line.after == row.seq);
            let runs = agent_runs.remove(&row.message_id);
            let ends = ReplayedEnds {
                compaction: marker,
                turn_stats: row.turn_stats,
            };
            let body = replayed_message(row.seq, row.message, &mut call_arguments, ends);
            let _ = stored.send(Event::new(id.clone(), body)).await;
            if let Some(runs) = runs {
                replay_agent_runs(&id, runs, &stored).await;
            }
        }
        Ok(())
    }

    /// The sub-agent runs stored on `branch`'s rows; none when they cannot be read.
    async fn agent_runs_of(&self, branch: BranchId) -> HashMap<MessageId, Vec<PersistedAgentRun>> {
        self.transcripts
            .load_agent_runs(branch)
            .await
            .unwrap_or_else(|e| {
                warn!(
                    target: "assistd::state",
                    error = %e,
                    "load_agent_runs failed; replaying relays without their sub-agents"
                );
                HashMap::new()
            })
    }

    /// Begin a fresh session for the user turn `id`, which is in flight and holds the
    /// user-turn lock, and report `BranchSwitched`; a session with nothing saved is kept.
    pub(super) async fn begin_session_for_turn(&self, id: &str, tx: &EventSink) {
        if self.session_ctx.current().await.1.is_none() {
            return;
        }
        self.transcript.drain_inflight().await;
        if let Some(stash) = &self.slot_stash {
            stash.end(SlotHolder::Main).await;
        }
        let session = self.start_unsaved_session("a lapsed voice session").await;
        report_unsaved_session(id.to_string(), session, tx).await;
    }

    /// Make a new, unsaved session active, clear the LLM history, and
    /// report `BranchSwitched`. The session and its `main`
    /// branch are inserted with its first message.
    async fn begin_unsaved_session(&self, id: String, tx: &EventSink, label: &str) {
        let new_session = self.start_unsaved_session(label).await;
        report_unsaved_session(id, new_session, tx).await;
    }

    /// Make a new, unsaved session active and clear the LLM history.
    async fn start_unsaved_session(&self, label: &str) -> SessionId {
        let new_session = SessionId::new();
        if let Err(e) = self.transcripts.clear_active_session().await {
            warn!(
                target: "assistd::memory",
                error = %e,
                "clear_active_session failed during {label}; a restart resumes the session left"
            );
        }
        self.session_ctx.replace(new_session, None).await;
        let empty = HistoryReplay::default();
        self.transcript.forget_user_history();
        if let Err(e) = self.llm.replace_history(empty).await {
            warn!(
                target: "assistd::state",
                error = %e,
                "replace_history(empty) failed during {label} (non-fatal)"
            );
        }
        new_session
    }

    /// `(name, parent name, fork point)` of `branch`, all `None` when the
    /// lookup fails or the branch is unknown.
    pub(super) async fn lookup_branch_meta(
        &self,
        branch: BranchId,
    ) -> (Option<String>, Option<String>, Option<Seq>) {
        let Ok(Some(info)) = self.transcripts.branch_info(branch).await else {
            return (None, None, None);
        };
        (
            Some(info.name),
            info.parent_branch_name,
            info.fork_point_seq,
        )
    }
}

/// `rows` of a branch, whole or from its summary's cut, as the LLM replays them
/// under its `summary`.
pub fn branch_replay(rows: &[HistoryRow], summary: Option<BranchSummary>) -> HistoryReplay {
    let mut folded_steps = Vec::new();
    let mut entries = Vec::new();
    for row in rows {
        match replay_part(summary.as_ref(), row.seq) {
            ReplayPart::Summarized => {}
            ReplayPart::FoldedStep => folded_steps.push(history_entry(row)),
            ReplayPart::Kept => entries.push(history_entry(row)),
        }
    }
    HistoryReplay {
        summary: summary.map(|s| s.content),
        folded_steps,
        entries,
    }
}

/// Report the unsaved `session` as active on its future `main` branch.
async fn report_unsaved_session(id: String, session: SessionId, tx: &EventSink) {
    let _ = tx
        .send(Event::new(
            id,
            EventBody::BranchSwitched {
                branch_id: None,
                session_id: session.to_string(),
                session_title: None,
                name: "main".to_string(),
                parent_branch_name: None,
                fork_point_seq: None,
            },
        ))
        .await;
}

fn replay_part(summary: Option<&BranchSummary>, seq: Seq) -> ReplayPart {
    let Some(summary) = summary else {
        return ReplayPart::Kept;
    };
    let kept_from = summary.kept_from_seq;
    if seq < kept_from {
        ReplayPart::Summarized
    } else if seq > kept_from && summary.steps_folded_through.is_some_and(|end| seq <= end) {
        ReplayPart::FoldedStep
    } else {
        ReplayPart::Kept
    }
}

/// The compaction line of a history whose rows have `seqs`, oldest first: after the
/// newest row left of those it held when `summary` was made.
fn compaction_line(
    seqs: impl Iterator<Item = Seq>,
    summary: &BranchSummary,
) -> Option<CompactionLine> {
    let after = seqs
        .take_while(|seq| *seq <= summary.compacted_after_seq)
        .last()?;
    Some(CompactionLine {
        after,
        duration_ms: summary.duration_ms,
    })
}

/// The stored `message` at `seq` of a history as a client replays it, taking the
/// arguments of a tool result's call from `call_arguments`.
fn replayed_message(
    seq: Seq,
    message: PersistedMessage,
    call_arguments: &mut HashMap<String, Value>,
    ends: ReplayedEnds,
) -> EventBody {
    let role = wire_role(message.role());
    let (reasoning, tool_name, tool_args, outcome) = match message.kind {
        MessageKind::Tool {
            call_id,
            tool_name,
            outcome,
            ..
        } => (
            None,
            Some(tool_name),
            call_arguments.remove(&call_id),
            Some(outcome),
        ),
        MessageKind::Assistant { reasoning, .. } => (reasoning, None, None, None),
        MessageKind::User => (None, None, None, None),
    };
    EventBody::HistoryEntry {
        seq: seq.0,
        role,
        content: message.content,
        reasoning_ms: reasoning.as_ref().map(|r| r.duration_ms),
        reasoning: reasoning.map_or_default(|r| r.text),
        tool_name,
        tool_args,
        exit_code: outcome.map(|o| o.exit_code),
        duration_ms: outcome.map(|o| o.duration_ms),
        compaction_ms: ends.compaction.map(|line| line.duration_ms),
        turn_stats: ends.turn_stats.map(wire_turn_stats),
    }
}

/// Replay the stored `runs` of the `relay` row just sent to `tx` as the events the relay
/// streamed: its plan, then each run that started, with its history after the task as
/// sub-agent events.
async fn replay_agent_runs(id: &str, runs: Vec<PersistedAgentRun>, tx: &EventSink) {
    let steps = runs
        .iter()
        .map(|run| RelayStepPlan {
            task: run.task.clone(),
            agent: run.agent.clone(),
        })
        .collect();
    tx.reply(id.to_owned(), EventBody::RelayPlanned { steps })
        .await;
    for (step, run) in (1u32..).zip(runs) {
        if let Some(record) = run.record {
            replay_agent_run(id, step, record, tx).await;
        }
    }
}

/// Replay run `step` of a stored relay as sub-agent `step`'s.
async fn replay_agent_run(id: &str, step: u32, record: AgentRunRecord, tx: &EventSink) {
    let sub_agent = AgentId(step);
    let from_agent = |body| Event::from_agent(id, sub_agent, body);
    tx.reply(
        id.to_owned(),
        EventBody::RelayStepStarted { step, sub_agent },
    )
    .await;
    let seqs = (0i64..).map(Seq);
    let compaction = record
        .summary
        .as_ref()
        .and_then(|s| compaction_line(seqs.clone().take(record.messages.len()), s));
    let mut call_arguments = tool_call_arguments(record.messages.iter().map(|m| &m.message));
    for (seq, stored) in seqs.zip(record.messages) {
        if stored.message.role() == PersistedRole::User {
            continue;
        }
        let ends = ReplayedEnds {
            compaction: compaction.filter(|line| line.after == seq),
            turn_stats: None,
        };
        let body = replayed_message(seq, stored.message, &mut call_arguments, ends);
        let _ = tx.send(from_agent(body)).await;
    }
    if let Some(tokens) = record.tokens {
        let total = EventBody::ContextSize {
            tokens,
            total: tokens,
        };
        let _ = tx.send(from_agent(total)).await;
    }
    let end = match record.end {
        AgentRunEnd::Answered { answer_file } => RelayStepEnd::Answered { answer_file },
        AgentRunEnd::Failed { message } => RelayStepEnd::Failed { message },
    };
    let ended = EventBody::RelayStepEnded {
        step,
        end,
        duration_ms: record.duration_ms,
    };
    tx.reply(id.to_owned(), ended).await;
}

fn wire_turn_stats(stats: TurnStats) -> assistd_ipc::TurnStats {
    assistd_ipc::TurnStats {
        duration_ms: stats.duration_ms,
        tokens: stats.generated_tokens,
    }
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
            reasoning: reasoning.as_ref().map_or_default(|r| r.text.clone()),
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

/// The arguments of every tool call `messages` request, by call id.
fn tool_call_arguments<'a>(
    messages: impl Iterator<Item = &'a PersistedMessage>,
) -> HashMap<String, Value> {
    messages
        .flat_map(|message| match &message.kind {
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
