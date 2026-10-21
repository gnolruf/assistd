//! `/wakeword`: the wake-word panel drawn in place of the transcript, with
//! its menu, the new-phrase input, and the enrollment and test sessions the
//! daemon runs on the mic.

use std::collections::VecDeque;

use assistd_ipc::{EnrollEvent, Event, Request, SampleRejection, WakePhraseText, WakeWordOp};
use assistd_tools::{ConfirmationRequest, Risk};
use assistd_voice::listen::enrollment::CONFIDENT_SAMPLES;
use assistd_voice::listen::wake_phrase::DEFAULT_WAKE_PHRASE;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use uuid::Uuid;

use super::{App, ConfirmTarget, ConfirmationPrompt, WireStream};

mod scene;

pub(in crate::chat) use scene::{ComputerScene, LANE_COLUMNS};

/// Sample or test results a session view keeps, newest last.
pub(in crate::chat) const RECENT_RESULTS: usize = 4;

/// Which `/wakeword` request a [`WireStream::WakeWord`] event answers.
#[derive(Debug, Clone, Copy)]
pub(in crate::chat) enum WakeWordStream {
    Status,
    /// An enrollment or test session, shown while its view is open.
    Session,
    /// A request stopping the open session.
    Control,
    Reset,
}

/// The `/wakeword` panel: the saved phrase and the view on top of it.
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
    pub variants: Vec<String>,
    /// Zero for the built-in phrase.
    pub samples: u32,
    pub listening: bool,
}

pub(in crate::chat) enum WakeWordView {
    Menu(MenuItem),
    /// The new phrase as typed so far.
    Update(String),
    Session(MicSession),
    Saved {
        phrase: String,
        variants: Vec<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::chat) enum MenuItem {
    Update,
    Test,
    Reset,
}

impl MenuItem {
    pub(in crate::chat) const ALL: [Self; 3] = [Self::Update, Self::Test, Self::Reset];

    fn previous(self) -> Self {
        match self {
            Self::Update | Self::Test => Self::Update,
            Self::Reset => Self::Test,
        }
    }

    fn next(self) -> Self {
        match self {
            Self::Update => Self::Test,
            Self::Test | Self::Reset => Self::Reset,
        }
    }
}

/// An enrollment or test session the daemon runs for this panel.
pub(in crate::chat) struct MicSession {
    /// Request id of the session; its `Done` or `Error` ends it.
    id: String,
    pub scene: ComputerScene,
    pub purpose: SessionPurpose,
}

pub(in crate::chat) enum SessionPurpose {
    Enroll(EnrollProgress),
    Test(TestProgress),
}

pub(in crate::chat) struct EnrollProgress {
    pub phrase: String,
    pub confident: u8,
    pub required: u8,
    pub results: VecDeque<SampleResult>,
}

pub(in crate::chat) enum SampleResult {
    Confident {
        heard: String,
    },
    /// Kept, but only its spelling is learned, as a variant.
    Variant {
        heard: String,
    },
    Rejected {
        reason: SampleRejection,
        heard: Option<String>,
    },
}

pub(in crate::chat) struct TestProgress {
    /// `None` when the saved phrase is not known yet.
    pub phrase: Option<String>,
    pub results: VecDeque<TestResult>,
}

pub(in crate::chat) struct TestResult {
    pub heard: String,
    pub matched: bool,
}

impl App {
    /// Open the panel on its menu and ask the daemon for the saved phrase.
    pub(super) fn open_wake_word_panel(&mut self) {
        self.wake_word_panel = Some(WakeWordPanel {
            status: StatusState::Loading,
            view: WakeWordView::Menu(MenuItem::Update),
        });
        self.refresh_wake_word_status();
    }

    /// The panel when it is on screen; an open confirmation prompt hides it.
    pub(in crate::chat) fn shown_wake_word_panel(&self) -> Option<&WakeWordPanel> {
        self.wake_word_panel
            .as_ref()
            .filter(|_| self.confirmation.is_none())
    }

    pub(super) fn spawn_wake_word_reset(&mut self) {
        self.send_wake_word_op(WakeWordOp::Reset, WakeWordStream::Reset);
    }

    /// Whether the open session's computer is animating.
    pub(crate) fn wake_word_scene_live(&self) -> bool {
        self.wake_word_panel
            .as_ref()
            .is_some_and(|panel| match &panel.view {
                WakeWordView::Session(session) => session.scene.is_live(),
                _ => false,
            })
    }

    pub(crate) fn advance_wake_word_scene(&mut self) {
        if let Some(WakeWordView::Session(session)) =
            self.wake_word_panel.as_mut().map(|panel| &mut panel.view)
        {
            session.scene.advance();
        }
    }

    /// Esc or Ctrl+C backs out of the current view, closing the panel from
    /// the menu.
    pub(super) fn on_wake_word_key(&mut self, ev: KeyEvent) {
        let Some(panel) = self.wake_word_panel.as_mut() else {
            return;
        };
        let control = ev.modifiers.contains(KeyModifiers::CONTROL);
        let back = ev.code == KeyCode::Esc || (ev.code == KeyCode::Char('c') && control);
        match &mut panel.view {
            WakeWordView::Menu(_) if back => self.wake_word_panel = None,
            WakeWordView::Menu(selected) => {
                let selected = *selected;
                self.on_menu_key(ev.code, selected);
            }
            WakeWordView::Update(_) if back => panel.view = WakeWordView::Menu(MenuItem::Update),
            WakeWordView::Update(phrase) => match ev.code {
                KeyCode::Enter => {
                    let phrase = phrase.clone();
                    self.start_enrollment(phrase);
                }
                KeyCode::Char('u') if control => phrase.clear(),
                KeyCode::Char(c) if !control => phrase.push(c),
                KeyCode::Backspace => {
                    phrase.pop();
                }
                _ => {}
            },
            WakeWordView::Session(session) if back => {
                let stop = match session.purpose {
                    SessionPurpose::Enroll(_) => WakeWordOp::EnrollCancel,
                    SessionPurpose::Test(_) => WakeWordOp::TestStop,
                };
                panel.view = WakeWordView::Menu(MenuItem::Update);
                self.send_wake_word_op(stop, WakeWordStream::Control);
            }
            WakeWordView::Session(_) => {}
            WakeWordView::Saved { phrase, .. } if ev.code == KeyCode::Char('t') => {
                let phrase = Some(phrase.clone());
                self.start_wake_word_test(phrase);
            }
            WakeWordView::Saved { .. } if back || ev.code == KeyCode::Enter => {
                panel.view = WakeWordView::Menu(MenuItem::Test);
            }
            WakeWordView::Saved { .. } => {}
        }
    }

    /// Route an event answering a `/wakeword` request. Copies of these
    /// events on the bus never reach here.
    pub(super) fn on_wake_word_event(&mut self, stream: WakeWordStream, ev: Event) {
        match ev {
            Event::WakeWordStatus {
                phrase,
                variants,
                samples,
                listening,
                ..
            } => self.set_wake_word_status(StatusState::Loaded(WakeWordStatus {
                phrase,
                variants,
                samples,
                listening,
            })),
            Event::WakeWordEnrollment { id, event } => self.on_enroll_event(&id, event),
            Event::WakeWordTest {
                id, heard, matched, ..
            } => {
                if let Some(SessionPurpose::Test(progress)) = self.session_purpose(&id) {
                    push_recent(&mut progress.results, TestResult { heard, matched });
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
        if let Some(panel) = self.wake_word_panel.as_mut()
            && matches!(panel.view, WakeWordView::Session(_))
        {
            panel.view = WakeWordView::Menu(MenuItem::Update);
        }
    }

    /// Up and Down move the selection; Enter or an item's initial chooses.
    fn on_menu_key(&mut self, key: KeyCode, selected: MenuItem) {
        match key {
            KeyCode::Up => self.set_wake_word_view(WakeWordView::Menu(selected.previous())),
            KeyCode::Down => self.set_wake_word_view(WakeWordView::Menu(selected.next())),
            KeyCode::Enter => self.choose_menu_item(selected),
            KeyCode::Char('u') => self.choose_menu_item(MenuItem::Update),
            KeyCode::Char('t') => self.choose_menu_item(MenuItem::Test),
            KeyCode::Char('r') => self.choose_menu_item(MenuItem::Reset),
            _ => {}
        }
    }

    fn choose_menu_item(&mut self, item: MenuItem) {
        match item {
            MenuItem::Update => self.set_wake_word_view(WakeWordView::Update(String::new())),
            MenuItem::Test => self.start_wake_word_test(self.saved_wake_phrase()),
            MenuItem::Reset => self.confirm_wake_word_reset(),
        }
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

    fn saved_wake_phrase(&self) -> Option<String> {
        match &self.wake_word_panel.as_ref()?.status {
            StatusState::Loaded(status) => Some(status.phrase.clone()),
            StatusState::Loading | StatusState::Failed(_) => None,
        }
    }

    fn refresh_wake_word_status(&mut self) {
        self.send_wake_word_op(WakeWordOp::Status, WakeWordStream::Status);
    }

    fn start_enrollment(&mut self, phrase: String) {
        let phrase = match WakePhraseText::try_from(phrase) {
            Ok(phrase) => phrase,
            Err(e) => {
                self.set_notice(&e.to_string());
                return;
            }
        };
        let purpose = SessionPurpose::Enroll(EnrollProgress {
            phrase: phrase.as_str().to_string(),
            confident: 0,
            required: CONFIDENT_SAMPLES,
            results: VecDeque::with_capacity(RECENT_RESULTS),
        });
        self.open_wake_word_session(WakeWordOp::EnrollStart { phrase }, purpose);
    }

    fn start_wake_word_test(&mut self, phrase: Option<String>) {
        let purpose = SessionPurpose::Test(TestProgress {
            phrase,
            results: VecDeque::with_capacity(RECENT_RESULTS),
        });
        self.open_wake_word_session(WakeWordOp::TestStart, purpose);
    }

    /// Refused mid-reply, when a command's confirmation prompt may need the dock.
    fn confirm_wake_word_reset(&mut self) {
        if self.generating() {
            self.set_notice("still generating, please wait");
            return;
        }
        let request = ConfirmationRequest {
            tool: "wakeword".into(),
            script: "/wakeword reset".into(),
            matched_pattern: format!(
                "Deletes the saved wake phrase, its learned variants and its recordings, \
                 and goes back to \"{DEFAULT_WAKE_PHRASE}\"."
            ),
            always_allow: Vec::new(),
            risk: Risk::Elevated,
        };
        self.confirmation = Some(ConfirmationPrompt::new(
            ConfirmTarget::WakeWordReset,
            request,
        ));
    }

    fn open_wake_word_session(&mut self, op: WakeWordOp, purpose: SessionPurpose) {
        let id = Uuid::new_v4().to_string();
        self.set_wake_word_view(WakeWordView::Session(MicSession {
            id: id.clone(),
            scene: ComputerScene::new(),
            purpose,
        }));
        self.spawn_one_shot(
            Request::WakeWord { id, op },
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

    fn session_purpose(&mut self, id: &str) -> Option<&mut SessionPurpose> {
        self.open_session(id).map(|session| &mut session.purpose)
    }

    fn on_enroll_event(&mut self, id: &str, event: EnrollEvent) {
        if let EnrollEvent::Saved { phrase, variants } = event {
            if self.open_session(id).is_some() {
                self.set_wake_word_view(WakeWordView::Saved { phrase, variants });
            }
            return;
        }
        let Some(SessionPurpose::Enroll(progress)) = self.session_purpose(id) else {
            return;
        };
        let result = match event {
            EnrollEvent::Prompt {
                confident,
                required,
            } => {
                progress.confident = confident;
                progress.required = required;
                return;
            }
            EnrollEvent::SampleAccepted {
                heard, confident, ..
            } => {
                if confident {
                    SampleResult::Confident { heard }
                } else {
                    SampleResult::Variant { heard }
                }
            }
            EnrollEvent::SampleRejected { reason, heard } => {
                SampleResult::Rejected { reason, heard }
            }
            EnrollEvent::Saved { .. } => return,
        };
        push_recent(&mut progress.results, result);
    }

    fn on_wake_word_done(&mut self, stream: WakeWordStream, id: &str) {
        match stream {
            WakeWordStream::Session => {
                self.end_wake_word_session(id);
                self.refresh_wake_word_status();
            }
            WakeWordStream::Reset => {
                self.output
                    .push_info(&format!("[wake phrase reset to \"{DEFAULT_WAKE_PHRASE}\"]"));
                self.set_notice(&format!("wake word reset to \"{DEFAULT_WAKE_PHRASE}\""));
                self.refresh_wake_word_status();
            }
            WakeWordStream::Status | WakeWordStream::Control => {}
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
            WakeWordStream::Control | WakeWordStream::Reset => self.set_notice(message),
        }
    }

    /// The session ended on the daemon; a saved phrase stays on screen.
    fn end_wake_word_session(&mut self, id: &str) {
        if self.open_session(id).is_some() {
            self.set_wake_word_view(WakeWordView::Menu(MenuItem::Update));
        }
    }
}

fn push_recent<T>(results: &mut VecDeque<T>, result: T) {
    if results.len() == RECENT_RESULTS {
        results.pop_front();
    }
    results.push_back(result);
}
