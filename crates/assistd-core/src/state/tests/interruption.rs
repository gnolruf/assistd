use std::collections::VecDeque;
use std::time::Instant;

use tokio::task::JoinHandle;

use assistd_config::TaskInterruption;
use assistd_ipc::{ConversationHold, EventBody, SuspendReason};

use super::*;

/// Quiet time after a reply before held runs resume, in seconds.
const QUIET_SECS: u32 = 1;

/// The order in which turns reached the model, each user step as `user:<text>` and
/// each task step as the task conversation's name.
type LaneLog = Arc<StdMutex<Vec<String>>>;

/// The user's conversation, logging each step under the prompt it answers and handing
/// out the queued task conversations in order.
#[derive(Debug)]
struct LoggedMain {
    log: LaneLog,
    last_prompt: StdMutex<String>,
    task_conversations: StdMutex<VecDeque<Arc<dyn LlmBackend>>>,
}

#[async_trait::async_trait]
impl TurnConversation for LoggedMain {
    async fn push_user(
        &self,
        text: String,
        _attachments: Vec<assistd_tools::Attachment>,
    ) -> assistd_llm::LlmResult<()> {
        *self.last_prompt.lock() = text;
        Ok(())
    }

    async fn push_tool_results(
        &self,
        _results: Vec<ToolResultPayload>,
    ) -> assistd_llm::LlmResult<()> {
        Ok(())
    }

    async fn step(
        &self,
        _tools: Vec<serde_json::Value>,
        _tx: mpsc::Sender<LlmEvent>,
    ) -> assistd_llm::LlmResult<StepOutcome> {
        let prompt = self.last_prompt.lock().clone();
        self.log.lock().push(format!("user:{prompt}"));
        Ok(StepOutcome::Final)
    }

    async fn push_interrupted_reply(&self, _reply: String) -> assistd_llm::LlmResult<()> {
        Ok(())
    }

    async fn set_transient_note(&self, _text: String) -> assistd_llm::LlmResult<()> {
        Ok(())
    }
}

impl LlmBackend for LoggedMain {
    fn isolated_conversation(
        &self,
        _system_prompt: String,
        _slot: SlotAssignment,
    ) -> assistd_llm::LlmResult<Arc<dyn LlmBackend>> {
        self.task_conversations
            .lock()
            .pop_front()
            .ok_or_else(|| LlmError::Unavailable("no task conversation left".into()))
    }
}

/// A task run's conversation that logs each step as it starts; its first step waits for
/// `release`, when set, before it answers.
#[derive(Debug)]
struct LoggedTask {
    name: &'static str,
    log: LaneLog,
    entered: Arc<Notify>,
    release: Option<Arc<Notify>>,
    steps: AtomicUsize,
}

#[async_trait::async_trait]
impl TurnConversation for LoggedTask {
    async fn push_user(
        &self,
        _text: String,
        _attachments: Vec<assistd_tools::Attachment>,
    ) -> assistd_llm::LlmResult<()> {
        Ok(())
    }

    async fn push_tool_results(
        &self,
        _results: Vec<ToolResultPayload>,
    ) -> assistd_llm::LlmResult<()> {
        Ok(())
    }

    async fn step(
        &self,
        _tools: Vec<serde_json::Value>,
        tx: mpsc::Sender<LlmEvent>,
    ) -> assistd_llm::LlmResult<StepOutcome> {
        self.log.lock().push(self.name.to_string());
        let first = self.steps.fetch_add(1, Ordering::SeqCst) == 0;
        if first && let Some(release) = &self.release {
            self.entered.notify_one();
            release.notified().await;
        }
        let _ = tx
            .send(LlmEvent::Delta {
                text: format!("{} done", self.name),
            })
            .await;
        Ok(StepOutcome::Final)
    }

    async fn push_interrupted_reply(&self, _reply: String) -> assistd_llm::LlmResult<()> {
        Ok(())
    }

    async fn set_transient_note(&self, _text: String) -> assistd_llm::LlmResult<()> {
        Ok(())
    }
}

impl LlmBackend for LoggedTask {}

/// A one-lane daemon under `interruption` whose runs get, in order, a task conversation
/// per entry of `tasks`: its name, and the gate its first step waits on, if any.
struct Harness {
    state: Arc<AppState>,
    conv: Arc<dyn TranscriptStore>,
    task: TaskId,
    log: LaneLog,
    entered: Vec<Arc<Notify>>,
}

impl Harness {
    async fn new(
        interruption: TaskInterruption,
        tasks: &[(&'static str, Option<Arc<Notify>>)],
    ) -> Self {
        let log = LaneLog::default();
        let mut entered = Vec::new();
        let mut task_conversations = VecDeque::new();
        for (name, release) in tasks {
            let notify = Arc::new(Notify::new());
            entered.push(Arc::clone(&notify));
            task_conversations.push_back(Arc::new(LoggedTask {
                name,
                log: Arc::clone(&log),
                entered: notify,
                release: release.clone(),
                steps: AtomicUsize::new(0),
            }) as Arc<dyn LlmBackend>);
        }
        let backend = Arc::new(LoggedMain {
            log: Arc::clone(&log),
            last_prompt: StdMutex::default(),
            task_conversations: StdMutex::new(task_conversations),
        });
        let mut config = Config::default();
        config.tasks.interruption = interruption;
        config.tasks.resume_after_idle_secs = assistd_config::defaults::bounded(QUIET_SECS);
        let (state, conv, _branch, task) =
            task_state_with(config, backend, Arc::new(ToolRegistry::default())).await;
        Self {
            state,
            conv,
            task,
            log,
            entered,
        }
    }

    fn logged(&self) -> Vec<String> {
        self.log.lock().clone()
    }

    /// Wait until `count` steps have reached the model.
    async fn await_logged(&self, count: usize) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while self.log.lock().len() < count {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("only {:?} reached the model", self.logged()));
    }

    /// Send query `text` in the background, returning its events as they arrive.
    fn start_query(&self, text: &str) -> (JoinHandle<()>, mpsc::Receiver<Event>) {
        let (tx, rx) = mpsc::channel(16);
        let state = Arc::clone(&self.state);
        let request = query(text, text);
        let handle = tokio::spawn(async move {
            state
                .dispatch(request, Requester::default(), tx)
                .await
                .unwrap();
        });
        (handle, rx)
    }

    /// Run query `text` to its end, returning when its reply finished.
    async fn ask(&self, text: &str) -> Instant {
        let (res, events) = tokio::time::timeout(
            Duration::from_secs(5),
            dispatch(&self.state, query(text, text)),
        )
        .await
        .unwrap_or_else(|_| panic!("query {text:?} never reached the model"));
        res.unwrap();
        assert_eq!(events, [done(text)]);
        Instant::now()
    }

    fn held(&self) -> Option<ConversationHold> {
        self.state.turns.deps.admission.queue_snapshot().held
    }

    /// Wait for every run to succeed.
    async fn settle(&self, runs: usize) {
        settle_task_runs(&self.state).await;
        assert_eq!(
            listed_run_states(&self.conv).await,
            vec![TaskRunState::Succeeded; runs]
        );
    }
}

async fn first_event(events: &mut mpsc::Receiver<Event>) -> Event {
    tokio::time::timeout(Duration::from_secs(5), events.recv())
        .await
        .expect("no event arrived")
        .expect("the query ended without an event")
}

fn assert_resumed_after_quiet(reply_finished: Instant) {
    assert!(
        reply_finished.elapsed() >= Duration::from_secs(QUIET_SECS.into()),
        "the queue resumed before the conversation was quiet"
    );
}

#[tokio::test]
async fn interrupt_suspends_the_run_mid_step_and_reruns_it_once_the_conversation_is_quiet() {
    let never = Arc::new(Notify::new());
    let harness = Harness::new(TaskInterruption::Interrupt, &[("task", Some(never))]).await;
    let run = queue_task_run(&harness.state, harness.task).await;
    harness.entered[0].notified().await;

    harness.ask("first").await;
    assert_eq!(
        queued_runs(&harness.state),
        [(
            run,
            TaskRunState::Suspended {
                reason: SuspendReason::UserConversation
            }
        )]
    );
    assert!(harness.held().is_some_and(|hold| hold.resumes_at.is_some()));
    let replied = harness.ask("second").await;
    assert_eq!(harness.logged(), ["task", "user:first", "user:second"]);

    harness.await_logged(4).await;
    assert_resumed_after_quiet(replied);
    assert_eq!(
        harness.logged(),
        ["task", "user:first", "user:second", "task"],
        "the discarded step reruns on resume"
    );
    harness.settle(1).await;
    assert_eq!(harness.held(), None);
}

#[tokio::test]
async fn after_task_waits_for_the_running_task_then_goes_before_the_queue() {
    let release = Arc::new(Notify::new());
    let harness = Harness::new(
        TaskInterruption::AfterTask,
        &[
            ("first task", Some(Arc::clone(&release))),
            ("second task", None),
        ],
    )
    .await;
    queue_task_run(&harness.state, harness.task).await;
    harness.entered[0].notified().await;
    queue_task_run(&harness.state, harness.task).await;

    let (reply, mut events) = harness.start_query("hi");
    assert_eq!(
        first_event(&mut events).await,
        Event::new("hi", EventBody::Queued { behind: 1 })
    );
    release.notify_one();
    reply.await.unwrap();
    let replied = harness.ask("again").await;
    assert_eq!(harness.logged(), ["first task", "user:hi", "user:again"]);

    harness.await_logged(4).await;
    assert_resumed_after_quiet(replied);
    assert_eq!(harness.logged()[3], "second task");
    harness.settle(2).await;
}

#[tokio::test]
async fn after_queue_turns_wait_behind_queued_runs_and_hold_the_runs_queued_after_them() {
    let release = Arc::new(Notify::new());
    let harness = Harness::new(
        TaskInterruption::AfterQueue,
        &[
            ("first task", Some(Arc::clone(&release))),
            ("second task", None),
            ("third task", None),
        ],
    )
    .await;
    queue_task_run(&harness.state, harness.task).await;
    harness.entered[0].notified().await;
    queue_task_run(&harness.state, harness.task).await;

    let (hi, mut hi_events) = harness.start_query("hi");
    let queued = |id: &str, behind| Event::new(id, EventBody::Queued { behind });
    assert_eq!(first_event(&mut hi_events).await, queued("hi", 2));
    let (also, mut also_events) = harness.start_query("also");
    assert_eq!(first_event(&mut also_events).await, queued("also", 3));
    queue_task_run(&harness.state, harness.task).await;
    release.notify_one();
    hi.await.unwrap();
    also.await.unwrap();
    let replied = harness.ask("again").await;
    assert_eq!(
        harness.logged(),
        [
            "first task",
            "second task",
            "user:hi",
            "user:also",
            "user:again"
        ]
    );

    harness.await_logged(6).await;
    assert_resumed_after_quiet(replied);
    assert_eq!(harness.logged()[5], "third task");
    harness.settle(3).await;
}
