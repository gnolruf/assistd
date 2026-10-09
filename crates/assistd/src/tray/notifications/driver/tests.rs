use serde_json::json;
use tokio::sync::mpsc;

use super::*;

struct Harness {
    driver: Driver,
    commands: UnboundedReceiver<NotifierCommand>,
}

impl Harness {
    fn new() -> Self {
        let (tx, commands) = mpsc::unbounded_channel();
        Self {
            driver: Driver::new(tx, &TrayNotificationsConfig::default()),
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
    Event::ChatFocus {
        id: "c".into(),
        focused,
    }
}

fn last_delta(id: &str, text: &str) -> Event {
    Event::LastDelta {
        id: id.into(),
        text: text.into(),
    }
}

fn done(id: &str) -> Event {
    Event::Done { id: id.into() }
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
    h.event(Event::ToolCall {
        id: "a".into(),
        name: "bash".into(),
        args: json!({"command": "ls"}),
    });
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
