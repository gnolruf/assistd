use std::time::Instant;

use tokio::task::JoinHandle;

use assistd_config::TaskInterruption;
use assistd_ipc::{EventBody, InteractionHold, SuspendReason};

use super::*;

/// Quiet time after a reply before held runs resume, in seconds.
const QUIET_SECS: u32 = 1;

/// The name the main backend journals its steps under.
const MAIN: &str = "user";

/// A one-lane daemon under `interruption` whose runs get, in order, a task backend
/// per entry of `tasks`: its name, and the gate its first step waits at, if any.
struct Harness {
    state: TestState,
    conv: Arc<dyn TranscriptStore>,
    task: TaskId,
    journal: StepJournal,
}

impl Harness {
    async fn new(interruption: TaskInterruption, tasks: &[(&str, Option<&Arc<Gate>>)]) -> Self {
        let journal = StepJournal::default();
        let task_backends = tasks.iter().map(|(name, gate)| {
            let first_step = gate.map(|gate| Step::default().waiting_at(gate));
            let backend = ScriptedBackend::new(first_step)
                .then_replying(format!("{name} done"))
                .journaling(*name, &journal);
            Arc::new(backend) as Arc<dyn LlmBackend>
        });
        let backend = ScriptedBackend::default()
            .handing_out(task_backends)
            .journaling(MAIN, &journal);
        let mut config = Config::default();
        config.tasks.interruption = interruption;
        config.tasks.resume_after_idle_secs = assistd_config::defaults::bounded(QUIET_SECS);
        let (state, conv, _branch, task) =
            task_state_with(config, Arc::new(backend), Arc::new(ToolRegistry::default())).await;
        Self {
            state,
            conv,
            task,
            journal,
        }
    }

    /// The order in which steps reached the model: each of a user turn as `user:<text>`
    /// and each of a task run as its backend's name.
    fn logged(&self) -> Vec<String> {
        self.journal
            .steps()
            .into_iter()
            .map(|JournaledStep { backend, prompt }| match backend.as_str() {
                MAIN => format!("{MAIN}:{prompt}"),
                _ => backend,
            })
            .collect()
    }

    /// Wait until `count` steps have reached the model.
    async fn await_logged(&self, count: usize) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while self.journal.steps().len() < count {
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

    fn held(&self) -> Option<InteractionHold> {
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
        "the queue resumed before the interaction was quiet"
    );
}

#[tokio::test]
async fn interrupt_suspends_the_run_mid_step_and_reruns_it_once_the_interaction_is_quiet() {
    let never_released = Gate::new();
    let tasks = [("task", Some(&never_released))];
    let harness = Harness::new(TaskInterruption::Interrupt, &tasks).await;
    let run = queue_task_run(&harness.state, harness.task).await;
    never_released.reached().await;

    harness.ask("first").await;
    assert_eq!(
        queued_runs(&harness.state),
        [(
            run,
            TaskRunState::Suspended {
                reason: SuspendReason::UserInteraction
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
    let first_step = Gate::new();
    let harness = Harness::new(
        TaskInterruption::AfterTask,
        &[("first task", Some(&first_step)), ("second task", None)],
    )
    .await;
    queue_task_run(&harness.state, harness.task).await;
    first_step.reached().await;
    queue_task_run(&harness.state, harness.task).await;

    let (reply, mut events) = harness.start_query("hi");
    assert_eq!(
        first_event(&mut events).await,
        Event::new("hi", EventBody::Queued { behind: 1 })
    );
    first_step.release();
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
    let first_step = Gate::new();
    let harness = Harness::new(
        TaskInterruption::AfterQueue,
        &[
            ("first task", Some(&first_step)),
            ("second task", None),
            ("third task", None),
        ],
    )
    .await;
    queue_task_run(&harness.state, harness.task).await;
    first_step.reached().await;
    queue_task_run(&harness.state, harness.task).await;

    let (hi, mut hi_events) = harness.start_query("hi");
    let queued = |id: &str, behind| Event::new(id, EventBody::Queued { behind });
    assert_eq!(first_event(&mut hi_events).await, queued("hi", 2));
    let (also, mut also_events) = harness.start_query("also");
    assert_eq!(first_event(&mut also_events).await, queued("also", 3));
    queue_task_run(&harness.state, harness.task).await;
    first_step.release();
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

#[tokio::test]
async fn a_waiting_query_whose_client_left_ends_without_taking_the_lane() {
    let first_step = Gate::new();
    let tasks = [("task", Some(&first_step))];
    let harness = Harness::new(TaskInterruption::AfterTask, &tasks).await;
    queue_task_run(&harness.state, harness.task).await;
    first_step.reached().await;

    let (reply, mut events) = harness.start_query("hi");
    assert_eq!(
        first_event(&mut events).await,
        Event::new("hi", EventBody::Queued { behind: 1 })
    );
    drop(events);
    tokio::time::timeout(Duration::from_secs(5), reply)
        .await
        .expect("the query kept waiting for the lane")
        .unwrap();

    first_step.release();
    await_task_queue(&harness.state, |queue| queue.items.is_empty().then_some(())).await;
    harness.settle(1).await;
    assert_eq!(harness.logged(), ["task"]);
}
