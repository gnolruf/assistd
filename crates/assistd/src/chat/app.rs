//! Chat application state and reducer. The `on_*` methods mutate state
//! only; I/O lives in the `spawn_*` methods and the attach loader.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use assistd_config::ConfigIssue;
use assistd_core::PresenceState;
use assistd_ipc::{
    ComponentReadiness, Event, ImageMime, IpcClient, PausedPrompt, PermissionMode, Request,
    TaskQueueSnapshot, TaskRunId, TaskRunSummary, VoiceCaptureState,
};
use assistd_tools::{Approval, Attachment, ConfirmationRequest, Risk, Unattended};
use assistd_utils::token_counter::TokenCounter;
use chrono::{DateTime, Utc};
use crossterm::event::KeyEvent;
use ratatui_image::picker::Picker;
use ratatui_image::protocol::StatefulProtocol;
use tokio::sync::mpsc;
use tokio::task::{JoinError, JoinSet};

use super::input::InputLine;
use super::output::OutputPane;
use super::selection::Selection;
use super::settings::SettingsPanel;
use super::tasks::TasksPanel;
use super::text_editor::{EditorShape, TextEditor};
use super::theme::Theme;

mod agents;
mod attach;
mod commands;
mod connection;
mod keys;
mod modal;
mod plan_mode;
mod resume;
mod tasks;
mod wakeword;
mod wire;

pub(super) use agents::AgentPanel;
pub(super) use modal::TypeAheadGuard;
pub(super) use plan_mode::{ChatMode, PlanChoice, PlanReview};
use wakeword::WakeWordStream;
pub(super) use wakeword::{
    ComputerScene, LANE_COLUMNS, Mood, RECENT_RESULTS, StatusState, TestProgress, WakeWordPanel,
    WakeWordStatus, WakeWordView,
};

const SPINNER_CHARS: &[char] = &['⣷', '⣯', '⣟', '⡿', '⢿', '⣻', '⣽', '⣾'];
const NOTICE_HOLD: Duration = Duration::from_secs(3);
/// How recently the daemon's branch must have been written for the chat to resume it.
const RESUME_RECENCY_SECS: u64 = 10;

const CONFIG_ISSUES_HINT: &str =
    "[the daemon will not start with these config errors; fix them in /settings]";
/// Query ids remembered so their copies on the broadcast bus are ignored;
/// the daemon runs one turn at a time, so a few suffice.
const OWN_TURN_MEMORY: usize = 8;

/// An image staged by `/attach` for the next submission.
pub(super) struct PendingAttachment {
    /// File basename.
    pub name: String,
    pub mime: ImageMime,
    pub bytes: Vec<u8>,
    /// `None` on terminals without graphics support.
    pub protocol: Option<StatefulProtocol>,
}

impl PendingAttachment {
    fn into_parts(self) -> (Attachment, Option<StatefulProtocol>, String) {
        (
            Attachment::Image {
                mime: self.mime,
                bytes: self.bytes,
            },
            self.protocol,
            self.name,
        )
    }
}

/// Boxed inside [`ChatEvent`] so the graphics buffers do not bloat the
/// other variants.
pub(super) struct AttachLoadedPayload {
    pub name: String,
    pub mime: ImageMime,
    pub size: usize,
    pub bytes: Vec<u8>,
    pub protocol: Option<StatefulProtocol>,
}

impl std::fmt::Debug for AttachLoadedPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttachLoadedPayload")
            .field("name", &self.name)
            .field("mime", &self.mime)
            .field("size", &self.size)
            .field("has_thumbnail", &self.protocol.is_some())
            .finish_non_exhaustive()
    }
}

/// Which concurrent daemon connection an event arrived on, and so which
/// slice of `App` state its terminal event may retire.
#[derive(Debug, Clone, Copy)]
pub(super) enum WireStream {
    /// A query dialog or a turn seen on the bus; owns the assistant message,
    /// [`App::reply_started_at`] and the query writer.
    Reply,
    /// A branch command; owns the in-flight branch op and its rows.
    Branch,
    /// A request whose reply only updates indicator state.
    Status,
    /// A `/wakeword` request; owns the wake-word dock.
    WakeWord(WakeWordStream),
    /// A `/tasks` request, routed to the panel by its id.
    Tasks,
}

#[derive(Debug)]
pub(super) enum ChatEvent {
    Wire {
        stream: WireStream,
        event: Event,
    },
    /// The connection ended without a terminal event.
    WireError {
        stream: WireStream,
        message: String,
    },
    /// An event from the daemon's broadcast bus subscription.
    Bus(Event),
    /// The bus subscription ended; events sent until it is restored are missed.
    BusLost,
    /// The bus subscription is back after [`ChatEvent::BusLost`].
    BusRestored,
    AttachLoaded(Box<AttachLoadedPayload>),
    AttachFailed {
        path: String,
        message: String,
    },
}

/// An answer the confirmation prompt offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ConfirmChoice {
    Once,
    /// Run, and stop asking for the request's `always_allow` names.
    Always,
    Deny,
}

impl ConfirmChoice {
    fn approval(self) -> Approval {
        match self {
            Self::Once => Approval::Once,
            Self::Always => Approval::Always,
            Self::Deny => Approval::Deny,
        }
    }
}

/// Where a confirmation prompt's answer goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AnswerTo {
    /// The active reply's query connection, as a `ConfirmResponse`.
    Reply,
    /// A `TaskConfirm` for this paused task run, whose "always" allows the host for its task.
    TaskRun(TaskRunId),
}

/// Command-confirmation prompt drawn in the chat panel while the daemon's
/// agent loop or a paused task run waits on the answer.
pub(super) struct ConfirmationPrompt {
    pub request: ConfirmationRequest,
    /// Echoed back with the answer.
    confirm_id: String,
    pub answer_to: AnswerTo,
    /// Index into [`choices`](Self::choices).
    pub selected: Selection,
    /// Every script row is shown, scrolled to `scroll`, rather than the
    /// first few.
    pub expanded: bool,
    /// First script row shown while expanded; each frame clamps it.
    pub scroll: usize,
}

impl ConfirmationPrompt {
    /// A prompt with "No" selected, so a stray Enter denies.
    pub(super) fn new(confirm_id: String, request: ConfirmationRequest) -> Self {
        Self::answered_to(confirm_id, request, AnswerTo::Reply)
    }

    /// The prompt paused task run `run` waits on, offering "always" when it names a host.
    pub(super) fn for_task_run(run: TaskRunId, prompt: PausedPrompt) -> Self {
        let request = ConfirmationRequest {
            tool: prompt.tool,
            script: prompt.script,
            matched_pattern: prompt.matched_pattern,
            always_allow: prompt.host.into_iter().map(String::from).collect(),
            risk: Risk::Elevated,
            unattended: Unattended::Never,
            source: prompt.source,
        };
        Self::answered_to(prompt.confirm_id, request, AnswerTo::TaskRun(run))
    }

    fn answered_to(confirm_id: String, request: ConfirmationRequest, answer_to: AnswerTo) -> Self {
        let mut prompt = Self {
            request,
            confirm_id,
            answer_to,
            selected: Selection::default(),
            expanded: false,
            scroll: 0,
        };
        prompt.selected = Selection::at(prompt.choices().len() - 1);
        prompt
    }

    /// The answers on offer, in display order; "always" only when the
    /// request names something to stop asking about.
    pub(super) fn choices(&self) -> &'static [ConfirmChoice] {
        if self.request.always_allow.is_empty() {
            &[ConfirmChoice::Once, ConfirmChoice::Deny]
        } else {
            &[
                ConfirmChoice::Once,
                ConfirmChoice::Always,
                ConfirmChoice::Deny,
            ]
        }
    }
}

/// What `/resume`, `/wakeword`, `/settings`, `/agents` or `/tasks` draws in place of the
/// transcript.
pub(super) enum Panel {
    Resume(ResumePicker),
    WakeWord(WakeWordPanel),
    Settings(Box<SettingsPanel>),
    Agents(AgentPanel),
    Tasks(Box<TasksPanel>),
}

pub(super) struct App {
    pub output: OutputPane,
    pub input: InputLine,
    pub tokens: TokenCounter,
    pub model_name: String,
    /// When the open reply stream began; `None` between replies.
    pub reply_started_at: Option<Instant>,
    /// Whether the daemon expects to compact history during the open reply.
    pub compacting_soon: bool,
    pub quitting: bool,
    pub spinner: usize,
    /// Transient message shown in the input rule and when it was set.
    pub notice: Option<(String, Instant)>,
    pub last_output_height: u16,
    /// `None` until the bus reports it.
    pub presence_state: Option<PresenceState>,
    /// Whether the bus subscription is up.
    pub daemon_reachable: bool,
    pub vision_enabled: bool,
    pub confirmation: Option<ConfirmationPrompt>,
    /// The plan the active reply waits on the user to accept or send back.
    pub plan_review: Option<PlanReview>,
    /// Keeps keys typed for the input line from answering a prompt that just opened.
    pub type_ahead: TypeAheadGuard,
    pub listening: VoiceCaptureState,
    pub voice_output_enabled: bool,
    /// The wake phrase was heard and its command has not arrived yet.
    pub wake_word_heard: bool,
    pub pending_attachments: Vec<PendingAttachment>,
    picker: Option<Picker>,
    ipc: Arc<IpcClient>,
    /// The reply turn that owns the output pane; `None` between turns.
    active_reply: Option<ActiveReply>,
    /// Ids of the latest queries this chat sent, newest last.
    own_turn_ids: VecDeque<String>,
    /// An utterance transcribed while another turn owned the pane, held
    /// back so it is drawn above its own answer.
    queued_transcription: Option<QueuedTranscription>,
    in_flight_branch_op: Option<BranchOp>,
    /// Rows of a `/resume` listing, handed to the picker on `Done`.
    branches_buffer: Vec<BranchListEntry>,
    chat_tx: mpsc::Sender<ChatEvent>,
    /// Connection and attachment-loading tasks; aborted on drop.
    tasks: JoinSet<()>,
    slash_selected: Selection,
    /// Set by Esc on the slash popup; cleared when the buffer leaves its
    /// `/` prefix.
    slash_dismissed: bool,
    /// The open panel; `confirmation` takes precedence while both are open.
    pub panel: Option<Panel>,
    /// The task queue as last broadcast; `None` until the bus delivers it.
    task_queue: Option<TaskQueueSnapshot>,
    /// The config file `/settings` edits.
    config_path: PathBuf,
    /// Throttles rewraps for a live block's timer to 1 Hz.
    last_live_seconds: Option<u64>,
    pub session_title: Option<String>,
    /// Ctrl+O: force-expands every thinking and tool block without
    /// touching their per-item `expanded` flags.
    pub verbose: bool,
    /// Shift+Tab: whether this chat's messages act, answering prompts under a permission
    /// mode, or plan.
    pub mode: ChatMode,
    /// The daemon honors [`PermissionMode::Auto`], so Shift+Tab offers it.
    pub auto_mode_offered: bool,
    /// Colors from the config as last loaded or saved from `/settings`.
    theme: Theme,
    /// Each MCP server that is unavailable, with why, redrawn under the banner.
    mcp_unavailable: Vec<(String, String)>,
}

/// The reply turn that owns the output pane. The daemon serialises turns,
/// so a turn another client started mid-query is handed the pane afterwards.
struct ActiveReply {
    id: String,
    /// Answers `ConfirmRequest`; `None` for a turn seen on the bus, whose
    /// connection belongs to another client.
    writer: Option<mpsc::Sender<Request>>,
}

struct QueuedTranscription {
    id: String,
    text: String,
}

#[derive(Debug, Clone, Copy)]
enum BranchOp {
    Fork,
    Switch,
    Undo,
    Resume,
    New,
    ResumePicker,
}

#[derive(Debug, Clone)]
pub(super) struct BranchListEntry {
    pub name: String,
    pub parent_branch_name: Option<String>,
    pub fork_point_seq: Option<i64>,
    pub message_count: u64,
    pub is_current_in_session: bool,
    pub is_active_session: bool,
    pub session_short: String,
    pub session_title: Option<String>,
    /// `None` when the daemon's timestamp does not parse.
    pub last_active_at: Option<DateTime<Utc>>,
    /// The task run the session holds; `None` for a user's session.
    pub task_run: Option<TaskRunSummary>,
}

impl BranchListEntry {
    pub(super) fn is_active(&self) -> bool {
        self.is_active_session && self.is_current_in_session
    }

    /// The picker tab that lists this entry.
    fn tab(&self) -> ResumeTab {
        match self.task_run {
            Some(_) => ResumeTab::Tasks,
            None => ResumeTab::Sessions,
        }
    }

    /// Whether every whitespace-separated term of the lowercased `query`
    /// appears in the title, task name, branch names or session id.
    fn matches_query(&self, query: &str) -> bool {
        let haystack = [
            self.session_title.as_deref(),
            self.task_run.as_ref().map(|run| run.task_name.as_str()),
            Some(self.name.as_str()),
            self.parent_branch_name.as_deref(),
            Some(self.session_short.as_str()),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
        query.split_whitespace().all(|term| haystack.contains(term))
    }
}

/// Which kind of session a [`ResumePicker`] lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ResumeTab {
    /// The user's own sessions.
    Sessions,
    /// Sessions that task runs held.
    Tasks,
}

/// Session list `/resume` draws in the chat panel, one tab at a time,
/// narrowed by a typed search.
pub(super) struct ResumePicker {
    entries: Vec<BranchListEntry>,
    tab: ResumeTab,
    query: TextEditor,
    /// Index into [`matches`](Self::matches).
    selected: Selection,
}

impl ResumePicker {
    /// Opens on the Sessions tab, selecting the active branch when it is listed there.
    fn new(entries: Vec<BranchListEntry>) -> Self {
        let mut picker = Self {
            entries,
            tab: ResumeTab::Sessions,
            query: TextEditor::new(String::new(), EditorShape::Line),
            selected: Selection::default(),
        };
        let active = picker.matches().iter().position(|entry| entry.is_active());
        picker.selected = active.map_or_else(Selection::default, Selection::at);
        picker
    }

    pub(super) fn tab(&self) -> ResumeTab {
        self.tab
    }

    pub(super) fn query(&self) -> &TextEditor {
        &self.query
    }

    pub(super) fn selected(&self) -> usize {
        self.selected.index()
    }

    /// How many entries the tab on screen lists before the search.
    pub(super) fn total(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.tab() == self.tab)
            .count()
    }

    /// The tab's entries the search keeps, in listing order.
    pub(super) fn matches(&self) -> Vec<&BranchListEntry> {
        let query = self.query.text().to_lowercase();
        self.entries
            .iter()
            .filter(|entry| entry.tab() == self.tab && entry.matches_query(&query))
            .collect()
    }

    /// Show the other tab, keeping the search.
    fn switch_tab(&mut self) {
        self.tab = match self.tab {
            ResumeTab::Sessions => ResumeTab::Tasks,
            ResumeTab::Tasks => ResumeTab::Sessions,
        };
        self.selected = Selection::default();
    }

    /// The selected match as a session-qualified switch target.
    fn current_target(&self) -> Option<String> {
        self.matches()
            .get(self.selected.index())
            .map(|e| format!("{}/{}", e.session_short, e.name))
    }

    fn select_previous(&mut self) {
        self.selected.previous();
    }

    fn select_next(&mut self) {
        let count = self.matches().len();
        self.selected.next(count);
    }

    /// Edits the search with `ev`, selecting the first match when it changed.
    fn edit_query(&mut self, ev: KeyEvent) {
        let before = self.query.text().len();
        self.query.on_key(ev);
        if self.query.text().len() != before {
            self.selected = Selection::default();
        }
    }
}

impl App {
    pub(super) fn new(
        ipc: Arc<IpcClient>,
        chat_tx: mpsc::Sender<ChatEvent>,
        model_name: String,
        vision_enabled: bool,
        auto_mode_offered: bool,
        picker: Option<Picker>,
        config_path: PathBuf,
    ) -> Self {
        let mut output = OutputPane::new();
        output.push_banner(model_name.clone());
        Self {
            output,
            input: InputLine::new(),
            tokens: TokenCounter::new(),
            model_name,
            reply_started_at: None,
            compacting_soon: false,
            quitting: false,
            spinner: 0,
            notice: None,
            last_output_height: 10,
            presence_state: None,
            daemon_reachable: true,
            vision_enabled,
            confirmation: None,
            plan_review: None,
            type_ahead: TypeAheadGuard::default(),
            listening: VoiceCaptureState::Idle,
            voice_output_enabled: false,
            wake_word_heard: false,
            pending_attachments: Vec::new(),
            picker,
            ipc,
            active_reply: None,
            own_turn_ids: VecDeque::with_capacity(OWN_TURN_MEMORY),
            queued_transcription: None,
            in_flight_branch_op: None,
            branches_buffer: Vec::new(),
            chat_tx,
            tasks: JoinSet::new(),
            slash_selected: Selection::default(),
            slash_dismissed: false,
            panel: None,
            task_queue: None,
            config_path,
            last_live_seconds: None,
            session_title: None,
            verbose: false,
            mode: ChatMode::Act(PermissionMode::Manual),
            auto_mode_offered,
            theme: Theme::default(),
            mcp_unavailable: Vec::new(),
        }
    }

    pub(super) fn with_theme(self, theme: Theme) -> Self {
        Self { theme, ..self }
    }

    pub(super) fn with_daemon_reachable(self, daemon_reachable: bool) -> Self {
        Self {
            daemon_reachable,
            ..self
        }
    }

    /// Show each of the config file's `issues` under the banner, with what they cost.
    pub(super) fn with_config_issues(mut self, issues: &[ConfigIssue]) -> Self {
        for issue in issues {
            self.output.push_error(&format!("[config] {issue}"));
        }
        if !issues.is_empty() {
            self.output.push_info(CONFIG_ISSUES_HINT);
        }
        self
    }

    /// Show each MCP server in `unavailable`, with why it is, under the banner.
    pub(super) fn with_mcp_unavailable(mut self, unavailable: Vec<(String, String)>) -> Self {
        self.mcp_unavailable = unavailable;
        self.push_mcp_notices();
        self
    }

    /// The colors to draw with: the open settings panel's, unsaved edits
    /// included, so changes show as they are made.
    pub(super) fn theme(&self) -> Theme {
        match &self.panel {
            Some(Panel::Settings(panel)) => panel.theme(),
            _ => self.theme,
        }
    }

    pub(super) fn should_quit(&self) -> bool {
        self.quitting
    }

    pub(super) fn generating(&self) -> bool {
        self.reply_started_at.is_some()
    }

    pub(super) fn spinner_char(&self) -> char {
        SPINNER_CHARS[self.spinner % SPINNER_CHARS.len()]
    }

    /// The panel on screen; an open confirmation prompt hides it.
    pub(super) fn shown_panel(&self) -> Option<&Panel> {
        self.panel.as_ref().filter(|_| self.confirmation.is_none())
    }

    pub(super) fn notice(&self) -> Option<&str> {
        self.notice.as_ref().map(|(s, _)| s.as_str())
    }

    pub(super) fn set_output_height(&mut self, h: u16) {
        self.last_output_height = h;
    }

    /// Advance timers and reap finished tasks, returning whether the screen shows any of
    /// them. Fails when a task panicked, since the panic hook has already restored the terminal.
    pub(super) fn on_tick(&mut self) -> Result<bool, JoinError> {
        while let Some(res) = self.tasks.try_join_next() {
            match res {
                Err(e) if e.is_panic() => return Err(e),
                Err(e) => tracing::warn!("chat task failed: {e}"),
                Ok(()) => {}
            }
        }
        let timed = self.shows_time();
        self.spinner = self.spinner.wrapping_add(1);
        if let Some((_, at)) = &self.notice
            && at.elapsed() > NOTICE_HOLD
        {
            self.notice = None;
        }
        if let Some(agent) = self.viewed_agent_mut() {
            agent.transcript_mut().refresh_live_block();
        }
        let live_secs = self.output.live_block_seconds();
        if live_secs != self.last_live_seconds {
            self.last_live_seconds = live_secs;
            if live_secs.is_some() {
                self.output.refresh_live_block();
            }
        }
        Ok(timed)
    }

    /// Whether anything on screen moves with the clock: a spinner, a timer, a notice
    /// about to expire, or a panel, which may show either.
    fn shows_time(&self) -> bool {
        self.generating()
            || self.listening != VoiceCaptureState::Idle
            || self.wake_word_heard
            || self.notice.is_some()
            || self.output.live_block_seconds().is_some()
            || self.panel_shown()
    }

    fn reset_transcript(&mut self) {
        self.close_agent_panel();
        self.output.clear();
        self.output.push_banner(self.model_name.clone());
        self.push_mcp_notices();
    }

    fn push_mcp_notices(&mut self) {
        for (server, reason) in &self.mcp_unavailable {
            self.output.push_info(&mcp_notice(server, reason));
        }
    }

    /// Track MCP server `server`'s new `state`, noting in the transcript when it becomes
    /// unavailable.
    fn on_mcp_readiness(&mut self, server: String, state: ComponentReadiness) {
        self.mcp_unavailable.retain(|(known, _)| *known != server);
        if let ComponentReadiness::Unavailable { reason } = state {
            self.output.push_info(&mcp_notice(&server, &reason));
            self.mcp_unavailable.push((server, reason));
        }
    }

    fn set_notice(&mut self, text: &str) {
        self.notice = Some((text.to_string(), Instant::now()));
    }
}

/// The transcript line for an MCP server that is unavailable for `reason`.
fn mcp_notice(server: &str, reason: &str) -> String {
    format!("[MCP {server}: {reason}]")
}

#[cfg(test)]
mod tests;
