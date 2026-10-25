//! `/wakeword`: the wake-word panel drawn in place of the transcript, with
//! the configured phrase and the test session the daemon runs on the mic.

use std::collections::VecDeque;

use assistd_ipc::{Event, Request, WakeWordOp};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use uuid::Uuid;

use super::{App, WireStream};

mod scene;

pub(in crate::chat) use scene::{ComputerScene, LANE_COLUMNS, Mood};

/// Test results the session view keeps, newest last.
pub(in crate::chat) const RECENT_RESULTS: usize = 4;

/// Which `/wakeword` request a [`WireStream::WakeWord`] event answers.
#[derive(Debug, Clone, Copy)]
pub(in crate::chat) enum WakeWordStream {
    Status,
    /// The test session, shown while its view is open.
    Session,
    /// A request stopping the open session.
    Control,
}

/// The `/wakeword` panel: the configured phrase and the view on top of it.
pub(in crate::chat) struct WakeWordPanel {
    pub status: StatusState,
    pub view: WakeWordView,
}

pub(in crate::chat) enum StatusState {
    Loading,
    Loaded(WakeWordStatus),
    Failed(String),
}

pub(in crate::chat) struct WakeWordStatus {
    pub phrase: String,
    pub listening: bool,
}

pub(in crate::chat) enum WakeWordView {
    Status,
    Session(MicSession),
}

/// A test session the daemon runs for this panel.
pub(in crate::chat) struct MicSession {
    /// Request id of the session; its `Done` or `Error` ends it.
    id: String,
    pub scene: ComputerScene,
    pub progress: TestProgress,
}

pub(in crate::chat) struct TestProgress {
    /// `None` when the configured phrase is not known yet.
    pub phrase: Option<String>,
    /// Whether the phrase was spotted, per utterance.
    pub results: VecDeque<bool>,
}

impl App {
    /// Open the panel on the configured phrase and ask the daemon for it.
    pub(super) fn open_wake_word_panel(&mut self) {
        self.wake_word_panel = Some(WakeWordPanel {
            status: StatusState::Loading,
            view: WakeWordView::Status,
        });
        self.send_wake_word_op(WakeWordOp::Status, WakeWordStream::Status);
    }

    /// The panel when it is on screen; an open confirmation prompt hides it.
    pub(in crate::chat) fn shown_wake_word_panel(&self) -> Option<&WakeWordPanel> {
        self.wake_word_panel
            .as_ref()
            .filter(|_| self.confirmation.is_none())
    }

    /// Whether the open session's computer is animating.
    pub(crate) fn wake_word_scene_live(&self) -> bool {
        self.wake_word_panel
            .as_ref()
            .is_some_and(|panel| match &panel.view {
                WakeWordView::Session(session) => session.scene.is_live(),
                WakeWordView::Status => false,
            })
    }

    pub(crate) fn advance_wake_word_scene(&mut self) {
        if let Some(WakeWordView::Session(session)) =
            self.wake_word_panel.as_mut().map(|panel| &mut panel.view)
        {
            session.scene.advance();
        }
    }

    /// Esc or Ctrl+C backs out of the test, then closes the panel; Enter or
    /// `t` starts a test.
    pub(super) fn on_wake_word_key(&mut self, ev: KeyEvent) {
        let Some(panel) = self.wake_word_panel.as_mut() else {
            return;
        };
        let control = ev.modifiers.contains(KeyModifiers::CONTROL);
        let back = ev.code == KeyCode::Esc || (ev.code == KeyCode::Char('c') && control);
        match &panel.view {
            WakeWordView::Status if back => self.wake_word_panel = None,
            WakeWordView::Status => {
                if matches!(ev.code, KeyCode::Enter | KeyCode::Char('t')) {
                    self.start_wake_word_test();
                }
            }
            WakeWordView::Session(_) if back => {
                panel.view = WakeWordView::Status;
                self.send_wake_word_op(WakeWordOp::TestStop, WakeWordStream::Control);
            }
            WakeWordView::Session(_) => {}
        }
    }

    /// Route an event answering a `/wakeword` request. Copies of these
    /// events on the bus never reach here.
    pub(super) fn on_wake_word_event(&mut self, stream: WakeWordStream, ev: Event) {
        match ev {
            Event::WakeWordStatus {
                phrase, listening, ..
            } => {
                let status = WakeWordStatus { phrase, listening };
                self.set_wake_word_status(StatusState::Loaded(status));
            }
            Event::WakeWordTest { id, matched } => {
                if let Some(session) = self.open_session(&id) {
                    session.scene.react(matched);
                    let results = &mut session.progress.results;
                    if results.len() == RECENT_RESULTS {
                        results.pop_front();
                    }
                    results.push_back(matched);
                }
            }
            Event::WakeWordSpeech { id, speaking } => {
                if let Some(session) = self.open_session(&id) {
                    session.scene.set_hearing(speaking);
                }
            }
            Event::Done { id } => self.on_wake_word_done(stream, &id),
            Event::Error { id, message } => self.on_wake_word_error(stream, &id, &message),
            _ => {}
        }
    }

    /// The session connection failed; leave its view.
    pub(super) fn fail_wake_word_session(&mut self) {
        self.set_wake_word_view(WakeWordView::Status);
    }

    fn set_wake_word_view(&mut self, view: WakeWordView) {
        if let Some(panel) = self.wake_word_panel.as_mut() {
            panel.view = view;
        }
    }

    fn set_wake_word_status(&mut self, status: StatusState) {
        if let Some(panel) = self.wake_word_panel.as_mut() {
            panel.status = status;
        }
    }

    fn configured_wake_phrase(&self) -> Option<String> {
        match &self.wake_word_panel.as_ref()?.status {
            StatusState::Loaded(status) => Some(status.phrase.clone()),
            StatusState::Loading | StatusState::Failed(_) => None,
        }
    }

    fn start_wake_word_test(&mut self) {
        let id = Uuid::new_v4().to_string();
        let progress = TestProgress {
            phrase: self.configured_wake_phrase(),
            results: VecDeque::with_capacity(RECENT_RESULTS),
        };
        self.set_wake_word_view(WakeWordView::Session(MicSession {
            id: id.clone(),
            scene: ComputerScene::new(),
            progress,
        }));
        self.spawn_one_shot(
            Request::WakeWord {
                id,
                op: WakeWordOp::TestStart,
            },
            WireStream::WakeWord(WakeWordStream::Session),
            "wakeword",
        );
    }

    fn send_wake_word_op(&mut self, op: WakeWordOp, stream: WakeWordStream) {
        let req = Request::WakeWord {
            id: Uuid::new_v4().to_string(),
            op,
        };
        self.spawn_one_shot(req, WireStream::WakeWord(stream), "wakeword");
    }

    /// The panel's session when it is session `id`.
    fn open_session(&mut self, id: &str) -> Option<&mut MicSession> {
        match &mut self.wake_word_panel.as_mut()?.view {
            WakeWordView::Session(session) if session.id == id => Some(session),
            _ => None,
        }
    }

    fn on_wake_word_done(&mut self, stream: WakeWordStream, id: &str) {
        if matches!(stream, WakeWordStream::Session) {
            self.end_wake_word_session(id);
            self.send_wake_word_op(WakeWordOp::Status, WakeWordStream::Status);
        }
    }

    fn on_wake_word_error(&mut self, stream: WakeWordStream, id: &str, message: &str) {
        self.output.push_error(message);
        match stream {
            WakeWordStream::Status => {
                self.set_wake_word_status(StatusState::Failed(message.to_string()));
            }
            WakeWordStream::Session => {
                self.set_notice(message);
                self.end_wake_word_session(id);
            }
            WakeWordStream::Control => self.set_notice(message),
        }
    }

    /// The session ended on the daemon; go back to the phrase.
    fn end_wake_word_session(&mut self, id: &str) {
        if self.open_session(id).is_some() {
            self.set_wake_word_view(WakeWordView::Status);
        }
    }
}
