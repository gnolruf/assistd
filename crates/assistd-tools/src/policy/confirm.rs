//! Confirmation gates and the per-connection router that carries their
//! prompts to an IPC client.

use std::collections::HashMap;
use std::ffi::OsStr;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::{Mutex as AsyncMutex, mpsc, oneshot, watch};
use tracing::{info, warn};

use assistd_config::defaults::CONFIRMATION_PROMPT_SECS;
use assistd_ipc::{
    ConfirmSource, Event, EventBody, HostName, PermissionMode, PlanVerdict, PlannedAction, TaskId,
};

use super::grant::{TurnGrant, spoken_needs};
use super::judge::{JudgedCommand, SafetyJudge, Verdict};
use super::on_blocking_pool;
use crate::TurnContext;

/// Cap on confirmation prompts and plan reviews in flight on one connection.
pub const MAX_PENDING_CONFIRMS: usize = 32;

/// How many names a spoken prompt lists before it counts the rest.
const SPOKEN_LIST_ITEMS: usize = 2;

/// How many web hosts a headless turn may ask about on their own before it must plan.
const UNPLANNED_SITES: usize = 1;

/// How long a prompt waits for the client's answer before it is denied.
pub const CONFIRM_TIMEOUT: Duration = Duration::from_secs(CONFIRMATION_PROMPT_SECS);

/// A request for the user's confirmation before a command runs.
#[derive(Debug, Clone)]
pub struct ConfirmationRequest {
    /// Tool name, e.g. `"bash"`.
    pub tool: String,
    /// Verbatim script the tool is about to execute.
    pub script: String,
    /// Why the command needs confirmation, for display.
    pub matched_pattern: String,
    /// The request in a sentence, for a prompt that is spoken instead of shown.
    pub summary: Summary,
    /// Programs an [`Approval::Always`] answer adds to the allowlist.
    pub always_allow: Vec<String>,
    /// How much it could do beyond what the prompt shows.
    pub risk: Risk,
    /// Which permission modes may approve it without asking.
    pub unattended: Unattended,
    /// The agent whose command it is.
    pub source: ConfirmSource,
}

/// What a [`ConfirmationRequest`] would do and why it asks, short enough to say out loud.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Summary {
    /// What would run, worded to follow "Allow": `a fetch from example.com`.
    pub what: String,
    /// Why it asks, as a sentence, when that is more than the tool not being approved yet.
    pub why: Option<String>,
}

impl Summary {
    /// A request for `what` that needs no reason said.
    pub fn of(what: impl Into<String>) -> Self {
        Self {
            what: what.into(),
            why: None,
        }
    }

    /// The same request, asked because of `why`.
    pub fn because(mut self, why: impl Into<String>) -> Self {
        self.why = Some(why.into());
        self
    }
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
    /// A fetch from this host, which a task run approves when its task allows the host; no
    /// mode approves it.
    WebHost(String),
}

/// What the safety judge is told about a command besides its text, and what a turn's grant
/// must hold to approve it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JudgeContext {
    pub class: JudgedClass,
    pub cwd: Option<PathBuf>,
    /// Why the policy could not settle it, shown to the judge.
    pub why: String,
    /// The actions that, all granted to a headless turn, approve it; a write names the path
    /// changed. Empty when no grant can.
    pub grantable: Vec<PlannedAction>,
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

/// What a prompt was answered with: one of its choices, or a refusal with words for the agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Answer {
    Choice(Approval),
    /// Deny, and tell the agent whose command it was this, in the user's own words.
    Message(String),
}

impl Answer {
    /// The answer an IPC client's fields stand for. A `message` that is not blank denies
    /// whatever `allow` says, and `always` alone denies.
    pub fn from_wire(allow: bool, always: bool, message: Option<&str>) -> Self {
        let message = message.map(str::trim).filter(|message| !message.is_empty());
        match (message, allow, always) {
            (Some(message), _, _) => Self::Message(message.to_string()),
            (None, false, _) => Self::Choice(Approval::Deny),
            (None, true, false) => Self::Choice(Approval::Once),
            (None, true, true) => Self::Choice(Approval::Always),
        }
    }

    /// Whether the command may run, and for how long that holds.
    pub fn approval(&self) -> Approval {
        match self {
            Self::Choice(approval) => *approval,
            Self::Message(_) => Approval::Deny,
        }
    }
}

impl From<Approval> for Answer {
    fn from(approval: Approval) -> Self {
        Self::Choice(approval)
    }
}

/// Decides whether a command that needs confirmation may run.
#[async_trait]
pub trait ConfirmationGate: fmt::Debug + Send + Sync + 'static {
    /// Ask whoever answers `turn`'s prompts. Every failure mode (channel drop,
    /// shutdown, timeout) must become [`Approval::Deny`] so a turn never hangs.
    async fn confirm(&self, turn: &TurnContext, req: ConfirmationRequest) -> Approval;
}

/// Holds a task run's prompts, which nobody is present to answer, until the user answers.
#[async_trait]
pub trait PromptParking: fmt::Debug + Send + Sync + 'static {
    /// Pause the run on `req` and resume it once answered; [`Approval::Always`] on a web
    /// fetch also allows its host for the rest of the run. Every failure is [`Approval::Deny`].
    async fn park(&self, req: ConfirmationRequest) -> Answer;
}

/// The user's answer to a prompt asked out loud.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpokenAnswer {
    Confirm,
    Deny,
    /// Neither a plain yes nor a plain no: what was said, for the agent.
    Message(String),
    /// The question could not be asked, or nothing was said to it.
    Unanswered,
}

/// Asks the prompts of a headless turn, one with no client to show them to, out loud.
#[async_trait]
pub trait VoiceGate: fmt::Debug + Send + Sync + 'static {
    /// Say what `req` asks and read the user's spoken answer. Every failure is
    /// [`SpokenAnswer::Deny`] or [`SpokenAnswer::Unanswered`].
    async fn ask_aloud(&self, req: &ConfirmationRequest) -> SpokenAnswer;

    /// Say `proposal`, a plan and the approvals it asks for, and read the user's spoken
    /// answer to it, failing as [`ask_aloud`](Self::ask_aloud) does.
    async fn propose_aloud(&self, proposal: &str) -> SpokenAnswer;
}

/// Who answers a [`ConfirmRouter`]'s prompts.
#[derive(Debug)]
enum Answerer {
    /// The IPC client behind the connection.
    Client,
    /// Nobody: the task's mode and allowed hosts, else its [`PromptParking`].
    TaskRun(TaskRunAsks),
    /// The user, out loud.
    Voice(VoiceAsks),
}

/// A headless turn's spoken prompts: one at a time, and none after one went unanswered.
#[derive(Debug)]
struct VoiceAsks {
    gate: Arc<dyn VoiceGate>,
    /// Whether an earlier prompt of the turn went unanswered; held while one is asked.
    unanswered: AsyncMutex<bool>,
    /// What the user approved together for this turn.
    grant: Mutex<TurnGrant>,
    /// Whether the user agreed to a plan of this turn.
    planned: AtomicBool,
    /// How many web hosts the turn has asked about with no plan agreed.
    unplanned_sites: AtomicUsize,
}

impl VoiceAsks {
    fn new(gate: Arc<dyn VoiceGate>) -> Self {
        Self {
            gate,
            unanswered: AsyncMutex::new(false),
            grant: Mutex::new(TurnGrant::default()),
            planned: AtomicBool::new(false),
            unplanned_sites: AtomicUsize::new(0),
        }
    }

    /// How many more web hosts the turn may ask about before it must plan; `None` once the
    /// user agreed to a plan, after which it may ask about any.
    fn sites_left(&self) -> Option<usize> {
        let asked = self.unplanned_sites.load(Ordering::Acquire);
        (!self.planned.load(Ordering::Acquire)).then(|| UNPLANNED_SITES.saturating_sub(asked))
    }

    /// Ask whether the work `summary` describes may go ahead with `actions` approved, and
    /// grant them to the turn on a yes. A plan with no actions is agreed to unasked.
    async fn propose(&self, summary: &str, actions: Vec<PlannedAction>) -> SpokenAnswer {
        if actions.is_empty() {
            self.planned.store(true, Ordering::Release);
            return SpokenAnswer::Confirm;
        }
        let mut unanswered = self.unanswered.lock().await;
        if *unanswered {
            warn!(
                target: "assistd::policy",
                "plan not proposed: an earlier spoken prompt of this turn went unanswered"
            );
            return SpokenAnswer::Unanswered;
        }
        let proposal = format!(
            "{summary} I need to {}. Shall I go ahead?",
            spoken_needs(&actions)
        );
        let answer = self.gate.propose_aloud(&proposal).await;
        info!(target: "assistd::policy", ?actions, ?answer, "spoken plan answered");
        *unanswered = answer == SpokenAnswer::Unanswered;
        if answer == SpokenAnswer::Confirm {
            let granted = on_blocking_pool(move || TurnGrant::canonical(actions)).await;
            self.grant.lock().merge(granted);
            self.planned.store(true, Ordering::Release);
        }
        answer
    }

    /// Whether the user granted the turn everything `req` needs approved.
    fn granted(&self, req: &ConfirmationRequest) -> bool {
        let granted = self.grant.lock().covers(req);
        if granted {
            info!(
                target: "assistd::policy",
                tool = %req.tool,
                what = %req.summary.what,
                "approved: the user granted it to this turn"
            );
        }
        granted
    }

    /// Approve `req` once when the user says so; an answer never approves anything for good.
    async fn ask(&self, req: &ConfirmationRequest) -> Answer {
        if matches!(req.unattended, Unattended::WebHost(_)) && self.sites_left().is_some() {
            self.unplanned_sites.fetch_add(1, Ordering::AcqRel);
        }
        let mut unanswered = self.unanswered.lock().await;
        if *unanswered {
            warn!(
                target: "assistd::policy",
                tool = %req.tool,
                "command denied: an earlier spoken prompt of this turn went unanswered"
            );
            return Approval::Deny.into();
        }
        let answer = self.gate.ask_aloud(req).await;
        info!(
            target: "assistd::policy",
            tool = %req.tool,
            what = %req.summary.what,
            ?answer,
            "spoken prompt answered"
        );
        *unanswered = answer == SpokenAnswer::Unanswered;
        match answer {
            SpokenAnswer::Confirm => Approval::Once.into(),
            SpokenAnswer::Deny | SpokenAnswer::Unanswered => Approval::Deny.into(),
            SpokenAnswer::Message(said) => Answer::Message(said),
        }
    }
}

/// The hosts a task run fetches from unasked, and where its other prompts wait.
#[derive(Debug)]
struct TaskRunAsks {
    allowed_hosts: Mutex<Vec<HostName>>,
    parking: Arc<dyn PromptParking>,
}

#[derive(Debug, Default)]
struct PendingPrompts {
    /// The client can no longer answer; every later ask is denied.
    closed: bool,
    prompts: HashMap<String, oneshot::Sender<Answer>>,
    reviews: HashMap<String, oneshot::Sender<PlanVerdict>>,
}

impl PendingPrompts {
    fn in_flight(&self) -> usize {
        self.prompts.len() + self.reviews.len()
    }
}

/// An answer named a `confirm_id` with no prompt in flight.
#[derive(Debug, thiserror::Error)]
#[error("no pending confirm for this confirm_id")]
pub struct NoPendingConfirm;

/// A verdict named a `review_id` with no plan review in flight.
#[derive(Debug, thiserror::Error)]
#[error("no pending plan review for this review_id")]
pub struct NoPendingReview;

/// Per-connection routing table for in-flight confirmation prompts,
/// handed to tool calls in their [`TurnContext`]. Asks beyond
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
    answerer: Answerer,
    /// The turn is planning, so every ask is refused without prompting anyone.
    planning: AtomicBool,
    refused_while_planning: AtomicUsize,
    /// Raised while a prompt waits in [`PromptParking`].
    parked: watch::Sender<bool>,
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
            answerer: Answerer::Client,
            planning: AtomicBool::new(false),
            refused_while_planning: AtomicUsize::new(0),
            parked: watch::Sender::new(false),
        })
    }

    /// A router for a task run with no client behind it, labelling its prompts with `source`:
    /// a prompt `mode` approves unasked, as for a client, or a fetch from one of
    /// `allowed_hosts` is approved once, and any other waits in `parking`. Auto mode falls
    /// back to Manual without a `judge`.
    pub fn for_task_run(
        request_id: String,
        source: ConfirmSource,
        mode: PermissionMode,
        judge: Option<SafetyJudge>,
        allowed_hosts: Vec<HostName>,
        parking: Arc<dyn PromptParking>,
    ) -> Arc<Self> {
        let asks = TaskRunAsks {
            allowed_hosts: Mutex::new(allowed_hosts),
            parking,
        };
        let router = Self {
            judge,
            ..Self::unanswered(request_id, Some(source), Answerer::TaskRun(asks))
        };
        router.set_mode(mode);
        Arc::new(router)
    }

    /// A router for a planning turn nobody reviews, which refuses every ask without prompting
    /// anyone, in every mode.
    pub fn for_planning(request_id: String) -> Arc<Self> {
        let router = Self::unanswered(request_id, None, Answerer::Client);
        router.set_planning(true);
        Arc::new(router)
    }

    /// A router for a headless turn, asked by voice with no client to show a prompt to: it
    /// approves what the turn's grant or `mode` runs unasked and asks `gate` every other
    /// prompt out loud, until one goes unanswered; the rest are then denied unasked. Auto
    /// mode falls back to Manual without a `judge`.
    pub fn for_voice_turn(
        request_id: String,
        mode: PermissionMode,
        judge: Option<SafetyJudge>,
        gate: Arc<dyn VoiceGate>,
    ) -> Arc<Self> {
        let asks = Answerer::Voice(VoiceAsks::new(gate));
        let router = Self {
            judge,
            ..Self::unanswered(request_id, None, asks)
        };
        router.set_mode(mode);
        Arc::new(router)
    }

    /// A router no client answers, so it never prompts on the wire.
    fn unanswered(
        request_id: String,
        main_source: Option<ConfirmSource>,
        answerer: Answerer,
    ) -> Self {
        let (wire, _) = mpsc::channel(1);
        Self {
            request_id,
            wire,
            timeout: CONFIRM_TIMEOUT,
            pending: Mutex::new(PendingPrompts {
                closed: true,
                ..PendingPrompts::default()
            }),
            mode: Mutex::new(PermissionMode::Manual),
            judge: None,
            main_source,
            answerer,
            planning: AtomicBool::new(false),
            refused_while_planning: AtomicUsize::new(0),
            parked: watch::Sender::new(false),
        }
    }

    /// Whether the turn's prompts are asked out loud.
    pub fn asks_aloud(&self) -> bool {
        matches!(self.answerer, Answerer::Voice(_))
    }

    /// The saved task the turn is a run of; `None` for any other turn.
    pub fn task_run(&self) -> Option<TaskId> {
        match &self.main_source {
            Some(ConfirmSource::Task { task, .. }) => Some(*task),
            Some(ConfirmSource::Main | ConfirmSource::Relay { .. }) | None => None,
        }
    }

    /// Ask the user of a headless turn out loud whether the work `summary` describes may go
    /// ahead with `actions` approved; a yes grants them for the rest of the turn and nothing
    /// for good, and a plan with no actions is agreed to unasked. `None` when this router's
    /// prompts are not asked out loud.
    pub async fn propose_grant(
        &self,
        summary: &str,
        actions: Vec<PlannedAction>,
    ) -> Option<SpokenAnswer> {
        match &self.answerer {
            Answerer::Voice(asks) => Some(asks.propose(summary, actions).await),
            Answerer::Client | Answerer::TaskRun(_) => None,
        }
    }

    /// The web hosts granted to a headless turn so far; `None` when this router's prompts
    /// are not asked out loud.
    pub fn granted_hosts(&self) -> Option<Vec<String>> {
        match &self.answerer {
            Answerer::Voice(asks) => Some(asks.grant.lock().hosts()),
            Answerer::Client | Answerer::TaskRun(_) => None,
        }
    }

    /// How many more web hosts a headless turn may ask about on their own before its user
    /// must agree to a plan naming the rest. `None` once they have, and for any other turn.
    pub fn unplanned_sites_left(&self) -> Option<usize> {
        match &self.answerer {
            Answerer::Voice(asks) => asks.sites_left(),
            Answerer::Client | Answerer::TaskRun(_) => None,
        }
    }

    /// How many asks this router has refused while planning.
    pub fn refused_while_planning(&self) -> usize {
        self.refused_while_planning.load(Ordering::Acquire)
    }

    /// Whether the turn is planning.
    pub fn planning(&self) -> bool {
        self.planning.load(Ordering::Acquire)
    }

    /// Refuse every later ask while `planning`; prompts already shown stay open.
    pub fn set_planning(&self, planning: bool) {
        self.planning.store(planning, Ordering::Release);
    }

    /// Start planning at the user's request and tell the client the turn now plans.
    pub async fn enter_plan_mode(&self) {
        self.set_planning(true);
        let event = Event::new(self.request_id.clone(), EventBody::PlanModeEntered);
        let _ = self.wire.send(event).await;
    }

    /// Show `plan` to the client and await its verdict, with no timeout; `None` when no
    /// client can answer, or it leaves first.
    pub async fn review_plan(&self, plan: String) -> Option<PlanVerdict> {
        let (review, verdict) = self.register_review()?;
        let event = Event::new(
            self.request_id.clone(),
            EventBody::PlanReview {
                review_id: review.id.clone(),
                plan,
            },
        );
        self.wire.send(event).await.ok()?;
        verdict.await.ok()
    }

    fn register_review(&self) -> Option<(PendingPrompt<'_>, oneshot::Receiver<PlanVerdict>)> {
        let mut pending = self.pending.lock();
        if pending.closed || pending.in_flight() >= MAX_PENDING_CONFIRMS {
            warn!(
                target: "assistd::policy",
                "plan not reviewed: the client cannot answer"
            );
            return None;
        }
        let review_id = next_prompt_id();
        let (tx, rx) = oneshot::channel();
        pending.reviews.insert(review_id.clone(), tx);
        let review = PendingPrompt {
            router: self,
            id: review_id,
        };
        Some((review, rx))
    }

    /// Deliver a client's verdict to the matching plan review.
    pub fn route_verdict(
        &self,
        review_id: &str,
        verdict: PlanVerdict,
    ) -> Result<(), NoPendingReview> {
        let tx = self
            .pending
            .lock()
            .reviews
            .remove(review_id)
            .ok_or(NoPendingReview)?;
        let _ = tx.send(verdict);
        Ok(())
    }

    /// Watch whether a prompt is parked on an answer that may take any time, during which a
    /// tool's deadline should not run.
    pub fn parked(&self) -> watch::Receiver<bool> {
        self.parked.subscribe()
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

    /// Approve `req` once when a headless turn's grant or the mode lets it run unasked;
    /// otherwise prompt the client and await the answer, for a task run settle it by the
    /// task's hosts or park it, or for a headless turn ask it out loud. Every failure is [`Approval::Deny`].
    pub(crate) async fn ask(&self, mut req: ConfirmationRequest) -> Answer {
        self.label(&mut req.source);
        if self.planning() {
            return refuse_while_planning(&self.refused_while_planning, &req).into();
        }
        let mode = *self.mode.lock();
        if self.granted(&req) || self.approves_unasked(mode, &mut req).await {
            return Approval::Once.into();
        }
        match &self.answerer {
            Answerer::TaskRun(run) => self.ask_for_task_run(run, req).await,
            Answerer::Voice(asks) => asks.ask(&req).await,
            Answerer::Client => self.ask_client(req).await,
        }
    }

    /// Whether a headless turn's grant covers `req`, which then needs no verdict from the judge.
    fn granted(&self, req: &ConfirmationRequest) -> bool {
        matches!(&self.answerer, Answerer::Voice(asks) if asks.granted(req))
    }

    /// Put this router's own source on a main agent's prompt, and name the task run on a
    /// relay's.
    fn label(&self, source: &mut ConfirmSource) {
        let Some(main) = &self.main_source else {
            return;
        };
        match (main, source) {
            (_, asker @ ConfirmSource::Main) => *asker = main.clone(),
            (
                ConfirmSource::Task { run, .. },
                ConfirmSource::Relay {
                    run: relayed_in, ..
                },
            ) => {
                *relayed_in = Some(*run);
            }
            _ => {}
        }
    }

    /// Approve `req` once when it fetches from a host the task allows; otherwise park it. An
    /// answer holds for this prompt alone, except that "always" allows its host for the rest
    /// of the run.
    async fn ask_for_task_run(&self, run: &TaskRunAsks, req: ConfirmationRequest) -> Answer {
        let host = match &req.unattended {
            Unattended::WebHost(host) => HostName::try_from(host.clone()).ok(),
            _ => None,
        };
        if let Some(host) = &host
            && run.allowed_hosts.lock().contains(host)
        {
            info!(
                target: "assistd::policy",
                tool = %req.tool,
                source = ?req.source,
                host = %host,
                "approved: the task allows this host"
            );
            return Approval::Once.into();
        }
        let answer = {
            let _parked = ParkedFlag::raise(&self.parked);
            run.parking.park(req).await
        };
        match answer {
            Answer::Choice(Approval::Always) => {
                run.allowed_hosts.lock().extend(host);
                Approval::Once.into()
            }
            answer => answer,
        }
    }

    /// Whether `mode` lets `req` run unasked, logging each approval. Why a judge did not
    /// allow it is added to the prompt a client shows.
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
                    Verdict::Allow => {
                        info!(
                            target: "assistd::policy",
                            tool = %req.tool,
                            ?mode,
                            class = %context.class,
                            "approved without asking"
                        );
                        true
                    }
                    Verdict::Ask(why) => {
                        req.matched_pattern = format!("{}; {why}", req.matched_pattern);
                        false
                    }
                }
            }
            _ => false,
        }
    }

    async fn ask_client(&self, req: ConfirmationRequest) -> Answer {
        let Some((prompt, answer)) = self.register(&req) else {
            return Approval::Deny.into();
        };
        let event = Event::new(
            self.request_id.clone(),
            EventBody::ConfirmRequest {
                confirm_id: prompt.id.clone(),
                tool: req.tool.clone(),
                script: req.script.clone(),
                matched_pattern: req.matched_pattern.clone(),
                always_allow: req.always_allow.clone(),
                source: req.source.clone(),
            },
        );
        if self.wire.send(event).await.is_err() {
            warn!(
                target: "assistd::policy",
                tool = %req.tool,
                "command denied: client disconnected before confirm"
            );
            return Approval::Deny.into();
        }
        self.await_answer(&req, answer).await
    }

    /// Record a pending prompt under a fresh id, or `None` when the
    /// router is closed or full.
    fn register(
        &self,
        req: &ConfirmationRequest,
    ) -> Option<(PendingPrompt<'_>, oneshot::Receiver<Answer>)> {
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
        if pending.in_flight() >= MAX_PENDING_CONFIRMS {
            warn!(
                target: "assistd::policy",
                tool = %req.tool,
                in_flight = pending.in_flight(),
                cap = MAX_PENDING_CONFIRMS,
                "command denied: pending-confirm cap reached"
            );
            return None;
        }
        let confirm_id = next_prompt_id();
        let (tx, rx) = oneshot::channel();
        pending.prompts.insert(confirm_id.clone(), tx);
        let prompt = PendingPrompt {
            router: self,
            id: confirm_id,
        };
        Some((prompt, rx))
    }

    async fn await_answer(
        &self,
        req: &ConfirmationRequest,
        answer: oneshot::Receiver<Answer>,
    ) -> Answer {
        match tokio::time::timeout(self.timeout, answer).await {
            Ok(Ok(answer)) => answer,
            Ok(Err(_)) => {
                warn!(
                    target: "assistd::policy",
                    tool = %req.tool,
                    "command denied: confirmation channel dropped"
                );
                Approval::Deny.into()
            }
            Err(_) => {
                warn!(
                    target: "assistd::policy",
                    tool = %req.tool,
                    timeout_secs = self.timeout.as_secs(),
                    "command denied: no answer before timeout"
                );
                Approval::Deny.into()
            }
        }
    }

    /// Deliver a client's answer to the matching pending prompt.
    /// Fails with [`NoPendingConfirm`] when no prompt with that id is in flight.
    pub fn route_response(&self, confirm_id: &str, answer: Answer) -> Result<(), NoPendingConfirm> {
        let tx = self
            .pending
            .lock()
            .prompts
            .remove(confirm_id)
            .ok_or(NoPendingConfirm)?;
        let _ = tx.send(answer);
        Ok(())
    }

    /// Deny every prompt in flight and every later ask, and drop every plan review.
    pub fn close(&self) {
        let drained = {
            let mut pending = self.pending.lock();
            pending.closed = true;
            pending.reviews.clear();
            std::mem::take(&mut pending.prompts)
        };
        for (_, tx) in drained {
            let _ = tx.send(Approval::Deny.into());
        }
    }
}

/// A router's parked flag, raised until dropped.
struct ParkedFlag<'router>(&'router watch::Sender<bool>);

impl<'router> ParkedFlag<'router> {
    fn raise(flag: &'router watch::Sender<bool>) -> Self {
        flag.send_replace(true);
        Self(flag)
    }
}

impl Drop for ParkedFlag<'_> {
    fn drop(&mut self) {
        self.0.send_replace(false);
    }
}

/// A prompt or plan review registered with a [`ConfirmRouter`], removed from its routing
/// table when dropped.
struct PendingPrompt<'router> {
    router: &'router ConfirmRouter,
    id: String,
}

impl Drop for PendingPrompt<'_> {
    fn drop(&mut self) {
        let mut pending = self.router.pending.lock();
        pending.prompts.remove(&self.id);
        pending.reviews.remove(&self.id);
    }
}

/// `path` as a sentence can name it: its last component and the folder holding that.
pub(crate) fn spoken_path(path: &Path) -> String {
    let file = path.file_name().map(OsStr::to_string_lossy);
    let folder = path
        .parent()
        .and_then(Path::file_name)
        .map(OsStr::to_string_lossy);
    match (file, folder) {
        (Some(file), Some(folder)) => format!("{file} in {folder}"),
        (Some(file), None) => file.into_owned(),
        (None, _) => path.display().to_string(),
    }
}

/// `items` as a sentence lists them, those past [`SPOKEN_LIST_ITEMS`] only counted.
pub(crate) fn spoken_list(items: impl IntoIterator<Item = impl Into<String>>) -> String {
    let mut items: Vec<String> = items.into_iter().map(Into::into).collect();
    let more = items.len().saturating_sub(SPOKEN_LIST_ITEMS);
    items.truncate(SPOKEN_LIST_ITEMS);
    let last = match more {
        0 => items.pop(),
        more => Some(format!("{more} more")),
    };
    match (items.is_empty(), last) {
        (_, None) => String::new(),
        (true, Some(last)) => last,
        (false, Some(last)) => format!("{} and {last}", items.join(", ")),
    }
}

/// Count and log `req`'s refusal by a planning turn.
/// An id no other prompt or plan review of this process has had, so a late answer to a
/// finished prompt never matches a newer one.
fn next_prompt_id() -> String {
    static NEXT_PROMPT_ID: AtomicU64 = AtomicU64::new(1);
    NEXT_PROMPT_ID.fetch_add(1, Ordering::Relaxed).to_string()
}

fn refuse_while_planning(refused: &AtomicUsize, req: &ConfirmationRequest) -> Approval {
    refused.fetch_add(1, Ordering::AcqRel);
    info!(
        target: "assistd::policy",
        tool = %req.tool,
        pattern = %req.matched_pattern,
        source = ?req.source,
        "command refused: a planning turn runs nothing that asks"
    );
    Approval::Deny
}

/// Gate that asks whoever answers the turn's prompts, denying when nobody does.
#[derive(Debug, Default)]
pub struct IpcConfirmationGate;

#[async_trait]
impl ConfirmationGate for IpcConfirmationGate {
    async fn confirm(&self, turn: &TurnContext, req: ConfirmationRequest) -> Approval {
        turn.ask(req).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::test_support::{FixedJudge, RecordingParking};
    use PermissionMode::{AcceptEdits, Auto, Manual};

    fn sample_request() -> ConfirmationRequest {
        ConfirmationRequest {
            tool: "bash".into(),
            script: "rm -rf foo".into(),
            matched_pattern: "rm -rf".into(),
            summary: Summary::of("a shell command"),
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
            grantable: Vec::new(),
        })
    }

    async fn recv_confirm_id(rx: &mut mpsc::Receiver<Event>) -> String {
        match rx.recv().await.expect("prompt on the wire").body {
            EventBody::ConfirmRequest { confirm_id, .. } => confirm_id,
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
                    assert_eq!(approval, expected.into(), "{case}");
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
        assert_eq!(approval, Approval::Deny.into());
        assert_eq!(router.pending.lock().prompts.len(), 0);
        router
            .route_response(&confirm_id, Approval::Once.into())
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
            .route_response(&confirm_id, Approval::Once.into())
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
        assert_eq!(in_flight.await.expect("ask task"), Approval::Deny.into());
        assert_eq!(router.pending.lock().prompts.len(), 0);

        assert_eq!(router.ask(sample_request()).await, Approval::Deny.into());
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

        assert_eq!(router.ask(sample_request()).await, Approval::Deny.into());
        assert_eq!(router.pending.lock().prompts.len(), MAX_PENDING_CONFIRMS);
        rx.try_recv()
            .expect_err("a denied ask must not put a prompt on the wire");
    }

    #[tokio::test]
    async fn gate_asks_the_turns_router_and_denies_a_turn_without_one() {
        let (tx, mut rx) = mpsc::channel(4);
        let router = ConfirmRouter::new("r".into(), tx, Duration::from_secs(60), None);
        let turn = TurnContext::answered_by(Arc::clone(&router));
        let answer =
            tokio::spawn(async move { IpcConfirmationGate.confirm(&turn, sample_request()).await });
        let confirm_id = recv_confirm_id(&mut rx).await;
        router
            .route_response(&confirm_id, Approval::Always.into())
            .expect("routed");
        assert_eq!(answer.await.expect("spawned gate"), Approval::Always);

        let unanswered = IpcConfirmationGate
            .confirm(&TurnContext::default(), sample_request())
            .await;
        assert_eq!(unanswered, Approval::Deny);
        rx.try_recv()
            .expect_err("a gate with no router must not reach the wire");
    }

    #[tokio::test]
    async fn a_message_denies_and_is_kept_for_the_agent_whose_prompt_it_answered() {
        let (tx, mut rx) = mpsc::channel(4);
        let router = ConfirmRouter::new("r".into(), tx, Duration::from_secs(60), None);
        let main = TurnContext::answered_by(Arc::clone(&router));
        let sub_agent = main.clone().for_agent();
        let asker = sub_agent.clone();
        let asked =
            tokio::spawn(
                async move { IpcConfirmationGate.confirm(&asker, sample_request()).await },
            );
        let confirm_id = recv_confirm_id(&mut rx).await;
        let answer = Answer::from_wire(true, true, Some(" yes, but in /tmp "));
        router.route_response(&confirm_id, answer).expect("routed");

        assert_eq!(asked.await.expect("spawned gate"), Approval::Deny);
        assert!(main.take_decline_messages().is_empty());
        assert_eq!(sub_agent.take_decline_messages(), ["yes, but in /tmp"]);
        assert_eq!(
            Answer::from_wire(true, false, Some("  ")),
            Approval::Once.into()
        );
    }

    #[tokio::test]
    async fn a_task_run_parks_all_but_its_allowed_hosts_and_what_its_mode_approves() {
        let host = |host: &str| ConfirmationRequest {
            script: format!("fetch {host}"),
            unattended: Unattended::WebHost(host.into()),
            ..sample_request()
        };
        let edit = ConfirmationRequest {
            script: "edit".into(),
            unattended: Unattended::ScopedEdit,
            ..sample_request()
        };
        for (mode, edit_parked) in [(Manual, true), (AcceptEdits, false)] {
            let parking = Arc::new(RecordingParking {
                parked: Mutex::new(Vec::new()),
                answer: Approval::Always,
            });
            let router = ConfirmRouter::for_task_run(
                "r".into(),
                ConfirmSource::Main,
                mode,
                None,
                vec!["allowed.example".to_string().try_into().unwrap()],
                Arc::clone(&parking) as Arc<dyn PromptParking>,
            );
            for req in [host("Allowed.example"), host("other.example"), edit.clone()] {
                assert_eq!(router.ask(req).await, Approval::Once.into(), "{mode:?}");
            }
            let mut expected = vec!["fetch other.example".to_string()];
            if edit_parked {
                expected.push("edit".into());
            }
            assert_eq!(*parking.parked.lock(), expected, "{mode:?}");
        }
    }
}
