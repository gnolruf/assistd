//! `/wakeword`: the status line, the reset prompt, and the docked panel that
//! shows an enrollment or test session the daemon runs on the mic.

use std::collections::VecDeque;

use assistd_ipc::{EnrollEvent, Event, Request, SampleRejection, WakePhraseText, WakeWordOp};
use assistd_tools::{ConfirmationRequest, Risk};
use assistd_voice::listen::wake_phrase::DEFAULT_WAKE_PHRASE;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use uuid::Uuid;

use super::commands::WakeWordCommand;
use super::{App, ConfirmTarget, ConfirmationPrompt, WireStream};

/// Sample or test results the dock keeps, newest last.
const RECENT_RESULTS: usize = 5;

/// Which `/wakeword` request a [`WireStream::WakeWord`] event answers.
#[derive(Debug, Clone, Copy)]
pub(in crate::chat) enum WakeWordStream {
    /// An enrollment or test session, which owns the dock.
    Session,
    /// Status, or a request steering the open session.
    Control,
    Reset,
}

/// The docked wake-word panel and the daemon session it shows.
pub(in crate::chat) struct WakeWordDock {
    /// Request id of the session; its `Done` or `Error` ends it.
    session_id: String,
    pub view: WakeWordView,
}

pub(in crate::chat) enum WakeWordView {
    Enroll(EnrollProgress),
    /// The enrollment was saved; `t` starts a test.
    Saved {
        phrase: String,
        variants: Vec<String>,
    },
    Test(TestProgress),
}

pub(in crate::chat) struct EnrollProgress {
    pub phrase: String,
    /// `None` until the daemon's first prompt.
    pub prompt: Option<EnrollPrompt>,
    pub results: VecDeque<SampleResult>,
}

#[derive(Debug, Clone, Copy)]
pub(in crate::chat) struct EnrollPrompt {
    pub accepted: u8,
    pub minimum: u8,
    pub consistent: bool,
}

impl EnrollPrompt {
    pub(in crate::chat) fn can_save(self) -> bool {
        self.accepted >= self.minimum
    }
}

pub(in crate::chat) enum SampleResult {
    Accepted {
        heard: String,
    },
    Rejected {
        reason: SampleRejection,
        heard: Option<String>,
    },
}

pub(in crate::chat) struct TestProgress {
    /// The phrase being tested, when this chat just saved it.
    pub phrase: Option<String>,
    pub results: VecDeque<TestResult>,
}

pub(in crate::chat) struct TestResult {
    pub heard: String,
    pub matched: bool,
}

impl App {
    pub(super) fn run_wake_word_command(&mut self, command: WakeWordCommand) {
        match command {
            WakeWordCommand::Status => {
                self.send_wake_word_op(WakeWordOp::Status, WakeWordStream::Control);
            }
            WakeWordCommand::Enroll(phrase) => self.start_enrollment(phrase),
            WakeWordCommand::Test => self.start_wake_word_test(None),
            WakeWordCommand::Reset => self.confirm_wake_word_reset(),
        }
    }

    pub(super) fn spawn_wake_word_reset(&mut self) {
        self.send_wake_word_op(WakeWordOp::Reset, WakeWordStream::Reset);
    }

    /// Keys while the dock is open: Esc or Ctrl+C cancels or closes it,
    /// Enter saves an enrollment, `t` tests a saved phrase, and PgUp/PgDn
    /// still scroll the transcript.
    pub(super) fn on_wake_word_key(&mut self, ev: KeyEvent) {
        let Some(dock) = self.wake_word_dock.as_ref() else {
            return;
        };
        let cancel = ev.code == KeyCode::Esc
            || (ev.code == KeyCode::Char('c') && ev.modifiers.contains(KeyModifiers::CONTROL));
        match (&dock.view, ev.code) {
            (_, KeyCode::PageUp) => self.output.scroll_page_up(self.last_output_height),
            (_, KeyCode::PageDown) => self.output.scroll_page_down(self.last_output_height),
            (WakeWordView::Enroll(_), _) if cancel => {
                self.close_wake_word_session(WakeWordOp::EnrollCancel);
            }
            (WakeWordView::Enroll(progress), KeyCode::Enter) => {
                let prompt = progress.prompt;
                self.finish_enrollment(prompt);
            }
            (WakeWordView::Test(_), _) if cancel => {
                self.close_wake_word_session(WakeWordOp::TestStop);
            }
            (WakeWordView::Saved { phrase, .. }, KeyCode::Char('t')) => {
                let phrase = phrase.clone();
                self.start_wake_word_test(Some(phrase));
            }
            (WakeWordView::Saved { .. }, KeyCode::Enter) => self.wake_word_dock = None,
            (WakeWordView::Saved { .. }, _) if cancel => self.wake_word_dock = None,
            _ => {}
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
            } => self.show_wake_word_status(&phrase, &variants, samples, listening),
            Event::WakeWordEnrollment { id, event } => self.on_enroll_event(&id, event),
            Event::WakeWordTest {
                id, heard, matched, ..
            } => {
                if let Some(WakeWordView::Test(progress)) = self.session_view(&id) {
                    push_recent(&mut progress.results, TestResult { heard, matched });
                }
            }
            Event::Done { id } => match stream {
                WakeWordStream::Session => self.end_wake_word_session(&id),
                WakeWordStream::Reset => self
                    .output
                    .push_info(&format!("[wake phrase reset to \"{DEFAULT_WAKE_PHRASE}\"]")),
                WakeWordStream::Control => {}
            },
            Event::Error { id, message } => {
                self.output.push_error(&message);
                if matches!(stream, WakeWordStream::Session) {
                    self.end_wake_word_session(&id);
                }
            }
            _ => {}
        }
    }

    /// The session connection failed; close the dock unless it shows a
    /// saved phrase.
    pub(super) fn fail_wake_word_session(&mut self) {
        if let Some(dock) = &self.wake_word_dock
            && !matches!(dock.view, WakeWordView::Saved { .. })
        {
            self.wake_word_dock = None;
        }
    }

    fn start_enrollment(&mut self, phrase: String) {
        let phrase = match WakePhraseText::try_from(phrase) {
            Ok(phrase) => phrase,
            Err(e) => {
                self.output.push_error(&format!(
                    "/wakeword: {e}. Usage: /wakeword <phrase>, quoted if it is \"test\" or \"reset\""
                ));
                self.set_notice("/wakeword: invalid phrase");
                return;
            }
        };
        let view = WakeWordView::Enroll(EnrollProgress {
            phrase: phrase.as_str().to_string(),
            prompt: None,
            results: VecDeque::with_capacity(RECENT_RESULTS),
        });
        self.open_wake_word_session(WakeWordOp::EnrollStart { phrase }, view);
    }

    fn start_wake_word_test(&mut self, phrase: Option<String>) {
        let view = WakeWordView::Test(TestProgress {
            phrase,
            results: VecDeque::with_capacity(RECENT_RESULTS),
        });
        self.open_wake_word_session(WakeWordOp::TestStart, view);
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

    fn open_wake_word_session(&mut self, op: WakeWordOp, view: WakeWordView) {
        let id = Uuid::new_v4().to_string();
        self.wake_word_dock = Some(WakeWordDock {
            session_id: id.clone(),
            view,
        });
        self.spawn_one_shot(
            Request::WakeWord { id, op },
            WireStream::WakeWord(WakeWordStream::Session),
            "wakeword",
        );
    }

    /// Close the dock now rather than when the daemon's `Done` arrives.
    fn close_wake_word_session(&mut self, stop: WakeWordOp) {
        self.wake_word_dock = None;
        self.send_wake_word_op(stop, WakeWordStream::Control);
    }

    fn finish_enrollment(&mut self, prompt: Option<EnrollPrompt>) {
        match prompt {
            Some(prompt) if prompt.can_save() => {
                self.set_notice("saving…");
                self.send_wake_word_op(WakeWordOp::EnrollFinish, WakeWordStream::Control);
            }
            Some(prompt) => {
                let more = prompt.minimum - prompt.accepted;
                self.set_notice(&format!("say it {more} more time(s) before saving"));
            }
            None => self.set_notice("say the phrase first"),
        }
    }

    fn send_wake_word_op(&mut self, op: WakeWordOp, stream: WakeWordStream) {
        let req = Request::WakeWord {
            id: Uuid::new_v4().to_string(),
            op,
        };
        self.spawn_one_shot(req, WireStream::WakeWord(stream), "wakeword");
    }

    /// The dock's view when it shows session `id`.
    fn session_view(&mut self, id: &str) -> Option<&mut WakeWordView> {
        self.wake_word_dock
            .as_mut()
            .filter(|dock| dock.session_id == id)
            .map(|dock| &mut dock.view)
    }

    fn on_enroll_event(&mut self, id: &str, event: EnrollEvent) {
        let Some(view) = self.session_view(id) else {
            return;
        };
        if let EnrollEvent::Saved { phrase, variants } = event {
            *view = WakeWordView::Saved { phrase, variants };
            return;
        }
        let WakeWordView::Enroll(progress) = view else {
            return;
        };
        match event {
            EnrollEvent::Prompt {
                accepted,
                minimum,
                consistent,
            } => {
                progress.prompt = Some(EnrollPrompt {
                    accepted,
                    minimum,
                    consistent,
                });
            }
            EnrollEvent::SampleAccepted { heard, .. } => {
                push_recent(&mut progress.results, SampleResult::Accepted { heard });
            }
            EnrollEvent::SampleRejected { reason, heard } => {
                push_recent(
                    &mut progress.results,
                    SampleResult::Rejected { reason, heard },
                );
            }
            EnrollEvent::Saved { .. } => {}
        }
    }

    /// The session ended on the daemon; a saved phrase stays on screen.
    fn end_wake_word_session(&mut self, id: &str) {
        if self
            .session_view(id)
            .is_some_and(|view| !matches!(view, WakeWordView::Saved { .. }))
        {
            self.wake_word_dock = None;
        }
    }

    fn show_wake_word_status(
        &mut self,
        phrase: &str,
        variants: &[String],
        samples: u32,
        listening: bool,
    ) {
        let enrolled = if samples == 0 {
            "built in, not enrolled".to_string()
        } else {
            format!("{samples} samples")
        };
        let variants = if variants.is_empty() {
            "none".to_string()
        } else {
            variants.join(", ")
        };
        let listening = if listening { "on" } else { "off" };
        self.output.push_info(&format!(
            "[wake phrase \"{phrase}\" · {enrolled} · learned variants: {variants} · \
             wake-word listening {listening}]"
        ));
    }
}

fn push_recent<T>(results: &mut VecDeque<T>, result: T) {
    if results.len() == RECENT_RESULTS {
        results.pop_front();
    }
    results.push_back(result);
}
