//! Notification driver: decides when the turn's notification is shown, updated,
//! hidden, or turned into an interrupt; nothing shows while the chat has focus.

use std::time::Duration;

use assistd_config::NotificationsConfig;
use assistd_ipc::{Event, EventBody, IpcClient, Request, RequestBody};
use assistd_utils::time::saturating_millis;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::task::JoinSet;
use tokio::time::{Instant, MissedTickBehavior, interval};
use uuid::Uuid;

use super::activity::ActivityTracker;
use super::render::Notification;
use super::task_notices::{TaskNotice, TaskNoticeTracker};

/// Idle-check period and update throttle: changes flush at most once a tick.
const TICK_INTERVAL: Duration = Duration::from_millis(250);

/// Daemon events, user actions and service signals the driver reacts to.
#[derive(Debug)]
pub(crate) enum DriverInput {
    Event(Box<Event>),
    Disconnected,
    /// An event matched a configured wake rule.
    Wake,
    /// The tray icon was clicked: hide the notification, or bring it back.
    TrayActivated,
    /// The user dismissed the notification; interrupts a busy turn.
    Dismissed,
    /// The notification's Hide action ran.
    HideInvoked,
    /// The service closed the notification on its own, such as on expiry.
    Expired,
    Shutdown,
}

/// What the notification service should do next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum NotifierCommand {
    /// Show or update the single assistd notification.
    Show(Notification),
    Close,
    /// Show a notification of its own about a task run.
    Announce(TaskNotice),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Display {
    Closed,
    /// `stale` means the tracker changed since the last `Show`.
    Shown {
        stale: bool,
    },
}

/// Run until `Shutdown` arrives or every sender drops, closing the
/// notification on the way out.
pub(super) async fn drive_notifications(
    mut rx: UnboundedReceiver<DriverInput>,
    commands: UnboundedSender<NotifierCommand>,
    cfg: NotificationsConfig,
    ipc: IpcClient,
) {
    let mut driver = Driver::new(commands, &cfg);
    let mut ticker = interval(TICK_INTERVAL);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut interrupts = JoinSet::new();

    loop {
        tokio::select! {
            biased;
            msg = rx.recv() => {
                let Some(msg) = msg else { break };
                if matches!(msg, DriverInput::Shutdown) {
                    break;
                }
                if driver.apply(msg) {
                    interrupts.spawn(interrupt_turn(ipc.clone()));
                }
            }
            Some(res) = interrupts.join_next() => {
                if let Err(e) = res {
                    tracing::warn!(target: "tray", "notification dismiss: interrupt task failed: {e}");
                }
            }
            _ = ticker.tick() => driver.tick(),
        }
    }
    driver.close();
}

struct Driver {
    commands: UnboundedSender<NotifierCommand>,
    tracker: ActivityTracker,
    task_notices: TaskNoticeTracker,
    /// `None` until the daemon has answered, which holds notifications.
    chat_focused: Option<bool>,
    display: Display,
    /// The turn the user hid; wake rules leave it hidden.
    hidden_turn: Option<String>,
    last_activity: Instant,
    was_busy: bool,
    was_speaking: bool,
    auto_hide: Duration,
    listen_auto_hide: Duration,
}

impl Driver {
    fn new(commands: UnboundedSender<NotifierCommand>, cfg: &NotificationsConfig) -> Self {
        Self {
            commands,
            tracker: ActivityTracker::default(),
            task_notices: TaskNoticeTracker::default(),
            chat_focused: None,
            display: Display::Closed,
            hidden_turn: None,
            last_activity: Instant::now(),
            was_busy: false,
            was_speaking: false,
            auto_hide: Duration::from_millis(cfg.auto_hide_ms.get().into()),
            listen_auto_hide: Duration::from_millis(cfg.listen_auto_hide_ms()),
        }
    }

    /// Apply one input; `true` means the in-flight turn must be interrupted.
    fn apply(&mut self, msg: DriverInput) -> bool {
        match msg {
            DriverInput::Event(ev) => self.ingest(&ev),
            DriverInput::Disconnected => self.disconnect(),
            DriverInput::Wake => self.wake(),
            DriverInput::TrayActivated => self.toggle(),
            DriverInput::Dismissed => return self.dismissed(),
            DriverInput::HideInvoked => self.hide_for_turn(),
            DriverInput::Expired => self.display = Display::Closed,
            DriverInput::Shutdown => self.close(),
        }
        false
    }

    /// Any event while shown, or a turn or speech finishing, restarts the
    /// auto-hide timer.
    fn ingest(&mut self, ev: &Event) {
        match &ev.body {
            EventBody::ChatFocus { focused } => {
                self.set_chat_focus(*focused);
                return;
            }
            EventBody::TaskQueue { .. } | EventBody::TaskReply { .. } => {
                self.announce_task_notices(ev);
                return;
            }
            _ => {}
        }
        self.tracker.ingest(ev);
        if let Display::Shown { stale } = &mut self.display {
            *stale = true;
            self.last_activity = Instant::now();
        }
        let is_busy = self.tracker.is_busy();
        let is_speaking = self.tracker.is_speaking();
        if (self.was_busy && !is_busy) || (self.was_speaking && !is_speaking) {
            self.last_activity = Instant::now();
        }
        self.was_busy = is_busy;
        self.was_speaking = is_speaking;
    }

    /// A run needing input shows whether or not the chat has focus; a reply waits on the
    /// chat being known to be unfocused, as a turn does.
    fn announce_task_notices(&mut self, ev: &Event) {
        for notice in self.task_notices.notices_for(ev) {
            if notice.ignores_focus() || self.chat_focused == Some(false) {
                let _ = self.commands.send(NotifierCommand::Announce(notice));
            }
        }
    }

    /// Gaining focus clears the screen; losing it mid-turn shows the
    /// turn's current state.
    fn set_chat_focus(&mut self, focused: bool) {
        self.chat_focused = Some(focused);
        if focused {
            self.close();
        } else if self.tracker.is_busy() {
            self.wake();
        }
    }

    fn disconnect(&mut self) {
        self.tracker.set_disconnected();
        self.chat_focused = None;
        self.was_busy = false;
        self.was_speaking = false;
        self.close();
    }

    fn wake(&mut self) {
        self.last_activity = Instant::now();
        let turn_hidden = self.hidden_turn.is_some()
            && self.hidden_turn.as_deref() == self.tracker.current_turn();
        if self.display == Display::Closed && !turn_hidden {
            self.open();
        }
    }

    fn toggle(&mut self) {
        match self.display {
            Display::Shown { .. } => self.hide_for_turn(),
            Display::Closed => {
                self.hidden_turn = None;
                self.open();
            }
        }
    }

    /// The notification is already gone; interrupt if a turn or its
    /// speech is still running.
    fn dismissed(&mut self) -> bool {
        self.display = Display::Closed;
        let running = self.tracker.is_busy() || self.tracker.is_speaking();
        if running {
            self.hidden_turn = self.tracker.current_turn().map(str::to_string);
        }
        running
    }

    fn hide_for_turn(&mut self) {
        self.hidden_turn = self.tracker.current_turn().map(str::to_string);
        self.close();
    }

    /// Show the notification unless the chat is focused or its focus is
    /// still unknown.
    fn open(&mut self) {
        if self.chat_focused != Some(false) {
            return;
        }
        self.last_activity = Instant::now();
        self.display = Display::Shown { stale: false };
        self.send_show();
    }

    fn close(&mut self) {
        if self.display != Display::Closed {
            self.display = Display::Closed;
            let _ = self.commands.send(NotifierCommand::Close);
        }
    }

    fn tick(&mut self) {
        if self.display == (Display::Shown { stale: true }) {
            self.display = Display::Shown { stale: false };
            self.send_show();
        }
        self.close_if_idle();
    }

    fn send_show(&self) {
        let pinned =
            self.tracker.is_busy() || self.tracker.is_speaking() || self.tracker.is_listening();
        let expire_ms = if pinned {
            0
        } else {
            saturating_millis(self.idle_timeout())
        };
        let _ = self.commands.send(NotifierCommand::Show(Notification {
            view: self.tracker.snapshot(),
            hideable: self.tracker.is_busy(),
            expire_ms,
        }));
    }

    fn idle_timeout(&self) -> Duration {
        if self.tracker.is_listening() {
            self.listen_auto_hide
        } else {
            self.auto_hide
        }
    }

    fn close_if_idle(&mut self) {
        if self.display == Display::Closed || self.tracker.is_busy() || self.tracker.is_speaking() {
            return;
        }
        if self.last_activity.elapsed() >= self.idle_timeout() {
            self.close();
        }
    }
}

async fn interrupt_turn(ipc: IpcClient) {
    let req = Request::new(Uuid::new_v4().to_string(), RequestBody::InterruptTurn);
    if let Err(e) = ipc.send(req).await {
        tracing::warn!(target: "tray", "notification dismiss: interrupt_turn failed: {e}");
    }
}

#[cfg(test)]
mod tests {
    use assistd_ipc::{
        EventBody, PausedPrompt, QueueItem, TaskId, TaskQueueSnapshot, TaskRunId, TaskRunState,
    };
    use serde_json::json;
    use tokio::sync::mpsc;

    use super::super::task_notices::NoticeKind;
    use super::*;

    struct Harness {
        driver: Driver,
        commands: UnboundedReceiver<NotifierCommand>,
    }

    impl Harness {
        fn new() -> Self {
            let (tx, commands) = mpsc::unbounded_channel();
            Self {
                driver: Driver::new(tx, &NotificationsConfig::default()),
                commands,
            }
        }

        /// A harness whose chat is known to be unfocused, so shows go out.
        fn away() -> Self {
            let mut h = Self::new();
            h.event(chat_focus(false));
            h
        }

        fn apply(&mut self, input: DriverInput) -> bool {
            self.driver.apply(input)
        }

        fn event(&mut self, ev: Event) {
            self.apply(DriverInput::Event(Box::new(ev)));
        }

        fn next_command(&mut self) -> Option<NotifierCommand> {
            self.commands.try_recv().ok()
        }

        fn expect_show(&mut self) -> Notification {
            match self.next_command() {
                Some(NotifierCommand::Show(n)) => n,
                other => panic!("expected Show, got {other:?}"),
            }
        }

        fn expect_close(&mut self) {
            assert_eq!(self.next_command(), Some(NotifierCommand::Close));
        }

        fn expect_nothing(&mut self) {
            assert_eq!(self.next_command(), None);
        }

        fn make_idle_for(&mut self, idle: Duration) {
            self.driver.last_activity = Instant::now()
                .checked_sub(idle)
                .expect("test clock has run long enough");
        }
    }

    fn chat_focus(focused: bool) -> Event {
        Event::new("c", EventBody::ChatFocus { focused })
    }

    fn last_delta(id: &str, text: &str) -> Event {
        Event::new(id, EventBody::LastDelta { text: text.into() })
    }

    fn done(id: &str) -> Event {
        Event::new(id, EventBody::Done)
    }

    fn queue_with_paused_run(confirm_id: &str) -> Event {
        Event::new(
            "q",
            EventBody::TaskQueue {
                snapshot: TaskQueueSnapshot {
                    items: vec![QueueItem::Run {
                        run: TaskRunId(3),
                        task: TaskId(1),
                        task_name: "digest".to_string().try_into().unwrap(),
                        state: TaskRunState::NeedsInput,
                        prompt: Some(PausedPrompt {
                            confirm_id: confirm_id.into(),
                            tool: "bash".into(),
                            script: "rm x".into(),
                            matched_pattern: "rm".into(),
                            host: None,
                            source: assistd_ipc::ConfirmSource::Main,
                        }),
                        started_at: None,
                    }],
                    lanes: Vec::new(),
                    paused: false,
                    held: None,
                },
            },
        )
    }

    fn task_reply(reply: &str) -> Event {
        Event::new(
            "t",
            EventBody::TaskReply {
                run: TaskRunId(3),
                task_name: "digest".to_string().try_into().unwrap(),
                reply: reply.into(),
            },
        )
    }

    fn expect_announce(h: &mut Harness) -> NoticeKind {
        match h.next_command() {
            Some(NotifierCommand::Announce(notice)) => notice.kind,
            other => panic!("expected Announce, got {other:?}"),
        }
    }

    #[test]
    fn task_notices_wait_for_focus_to_be_lost_except_a_run_needing_input() {
        let needs_input = NoticeKind::NeedsInput {
            reason: "rm".into(),
        };
        let mut h = Harness::new();
        h.event(chat_focus(true));
        h.event(queue_with_paused_run("c1"));
        assert_eq!(expect_announce(&mut h), needs_input);
        h.event(queue_with_paused_run("c1"));
        h.expect_nothing();
        h.event(queue_with_paused_run("c2"));
        assert_eq!(expect_announce(&mut h), needs_input);
        h.event(task_reply("3 new messages"));
        h.expect_nothing();

        h.event(chat_focus(false));
        h.event(task_reply("3 new messages"));
        assert_eq!(
            expect_announce(&mut h),
            NoticeKind::Reply {
                text: "3 new messages".into()
            }
        );
    }

    #[test]
    fn nothing_shows_until_the_chat_focus_is_known() {
        let mut h = Harness::new();
        h.apply(DriverInput::Wake);
        h.apply(DriverInput::TrayActivated);
        h.expect_nothing();

        h.event(chat_focus(false));
        h.event(last_delta("a", "hi"));
        h.apply(DriverInput::Wake);
        h.expect_show();
    }

    #[test]
    fn gaining_focus_closes_the_notification_and_drops_pending_updates() {
        let mut h = Harness::away();
        h.event(last_delta("a", "first"));
        h.apply(DriverInput::Wake);
        h.expect_show();
        h.event(last_delta("a", "pending"));
        h.event(chat_focus(true));
        h.expect_close();
        h.driver.tick();
        h.expect_nothing();
    }

    #[test]
    fn updates_are_flushed_once_per_tick_with_the_latest_state() {
        let mut h = Harness::away();
        h.event(last_delta("a", "start"));
        h.apply(DriverInput::Wake);
        h.expect_show();
        for n in 0..10 {
            h.event(last_delta("a", &format!("chunk {n}")));
        }
        h.expect_nothing();
        h.driver.tick();
        let shown = h.expect_show();
        assert_eq!(shown.view.body, "chunk 9");
        assert!(shown.hideable, "a busy turn offers Hide");
        assert_eq!(shown.expire_ms, 0, "a busy turn never expires");
        h.driver.tick();
        h.expect_nothing();
    }

    #[test]
    fn dismissing_a_busy_turn_interrupts_and_keeps_it_hidden() {
        let mut h = Harness::away();
        h.event(last_delta("a", "working"));
        h.apply(DriverInput::Wake);
        h.expect_show();
        assert!(h.apply(DriverInput::Dismissed));
        h.event(Event::new(
            "a",
            EventBody::ToolCall {
                name: "bash".into(),
                args: json!({"command": "ls"}),
            },
        ));
        h.apply(DriverInput::Wake);
        h.driver.tick();
        h.expect_nothing();
    }

    #[test]
    fn dismissing_after_the_turn_ends_does_not_interrupt() {
        let mut h = Harness::away();
        h.event(last_delta("a", "reply"));
        h.apply(DriverInput::Wake);
        h.event(done("a"));
        assert!(!h.apply(DriverInput::Dismissed));
    }

    #[test]
    fn hide_closes_without_interrupting_until_the_next_turn() {
        let mut h = Harness::away();
        h.event(last_delta("a", "working"));
        h.apply(DriverInput::Wake);
        h.expect_show();
        assert!(!h.apply(DriverInput::HideInvoked));
        h.expect_close();
        h.apply(DriverInput::Wake);
        h.expect_nothing();

        h.event(done("a"));
        h.event(last_delta("b", "next turn"));
        h.apply(DriverInput::Wake);
        assert_eq!(h.expect_show().view.body, "next turn");
    }

    #[test]
    fn closes_once_idle_but_never_while_busy() {
        let mut h = Harness::away();
        h.event(last_delta("a", "working"));
        h.apply(DriverInput::Wake);
        h.expect_show();
        h.make_idle_for(Duration::from_secs(60));
        h.driver.tick();
        h.expect_nothing();

        h.event(done("a"));
        h.driver.tick();
        let finished = h.expect_show();
        assert!(!finished.hideable);
        assert_eq!(finished.expire_ms, 3000);
        h.make_idle_for(Duration::from_secs(60));
        h.driver.tick();
        h.expect_close();
    }
}
