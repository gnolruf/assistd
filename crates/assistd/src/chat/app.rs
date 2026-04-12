//! Top-level chat application state and reducer.
//!
//! `App` owns the output pane, input line, throughput meter, VRAM state,
//! and a handle to an `LlmBackend` for spawning generation tasks. The
//! `on_*` methods are pure reducers; I/O is confined to `spawn_generation`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use assistd_llm::{LlmBackend, LlmEvent};
use crossterm::event::{KeyCode, KeyEvent};
use tokio::sync::mpsc;

use super::input::{InputAction, InputLine};
use super::output::OutputPane;
use super::throughput::ThroughputMeter;
use super::vram::ResourceState;

const SPINNER_CHARS: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const NOTICE_HOLD: Duration = Duration::from_secs(3);

#[derive(Debug)]
pub enum ChatEvent {
    Llm(LlmEvent),
    LlmError(String),
}

pub struct App {
    pub output: OutputPane,
    pub input: InputLine,
    pub throughput: ThroughputMeter,
    pub resources: ResourceState,
    pub model_name: String,
    pub generating: bool,
    pub quitting: bool,
    pub spinner: usize,
    pub notice: Option<(String, Instant)>,
    pub last_output_height: u16,
    client: Arc<dyn LlmBackend>,
    chat_tx: mpsc::Sender<ChatEvent>,
}

impl App {
    pub fn new(
        client: Arc<dyn LlmBackend>,
        chat_tx: mpsc::Sender<ChatEvent>,
        model_name: String,
    ) -> Self {
        Self {
            output: OutputPane::new(),
            input: InputLine::new(),
            throughput: ThroughputMeter::new(),
            resources: ResourceState::default(),
            model_name,
            generating: false,
            quitting: false,
            spinner: 0,
            notice: None,
            last_output_height: 10,
            client,
            chat_tx,
        }
    }

    pub fn should_quit(&self) -> bool {
        self.quitting
    }

    pub fn spinner_char(&self) -> char {
        SPINNER_CHARS[self.spinner % SPINNER_CHARS.len()]
    }

    pub fn notice(&self) -> Option<&str> {
        self.notice.as_ref().map(|(s, _)| s.as_str())
    }

    pub fn set_output_height(&mut self, h: u16) {
        self.last_output_height = h;
    }

    pub fn on_key(&mut self, ev: KeyEvent) {
        match ev.code {
            KeyCode::PageUp => {
                self.output.scroll_page_up(self.last_output_height);
                return;
            }
            KeyCode::PageDown => {
                self.output.scroll_page_down(self.last_output_height);
                return;
            }
            _ => {}
        }
        match self.input.on_key(ev) {
            InputAction::None => {}
            InputAction::Submit(text) => {
                if self.generating {
                    self.set_notice("still generating — please wait");
                } else {
                    self.submit(text);
                }
            }
            InputAction::Quit => {
                self.quitting = true;
            }
        }
    }

    pub fn on_llm_event(&mut self, ev: ChatEvent) {
        let now = Instant::now();
        match ev {
            ChatEvent::Llm(LlmEvent::Delta { text }) => {
                self.throughput.on_delta(now);
                self.output.append_assistant(&text);
            }
            ChatEvent::Llm(LlmEvent::Done) => {
                self.throughput.on_done(now);
                self.output.finish_assistant();
                self.generating = false;
            }
            ChatEvent::LlmError(msg) => {
                self.throughput.on_done(now);
                self.output.finish_assistant();
                self.output.push_error(&msg);
                self.generating = false;
            }
        }
    }

    pub fn on_tick(&mut self) {
        self.spinner = self.spinner.wrapping_add(1);
        if let Some((_, at)) = &self.notice {
            if at.elapsed() > NOTICE_HOLD {
                self.notice = None;
            }
        }
    }

    pub fn on_resources(&mut self, v: ResourceState) {
        self.resources = v;
    }

    fn set_notice(&mut self, text: &str) {
        self.notice = Some((text.to_string(), Instant::now()));
    }

    fn submit(&mut self, text: String) {
        self.begin_submit(&text);
        self.spawn_generation(text);
    }

    fn begin_submit(&mut self, text: &str) {
        self.output.push_user(text);
        self.output.reset_scroll();
        self.output.begin_assistant();
        self.throughput.reset();
        self.generating = true;
    }

    fn spawn_generation(&self, text: String) {
        let client = self.client.clone();
        let tx = self.chat_tx.clone();
        tokio::spawn(async move {
            let (llm_tx, mut llm_rx) = mpsc::channel::<LlmEvent>(64);
            let tx_fwd = tx.clone();
            let forwarder = tokio::spawn(async move {
                while let Some(ev) = llm_rx.recv().await {
                    if tx_fwd.send(ChatEvent::Llm(ev)).await.is_err() {
                        return;
                    }
                }
            });
            let result = client.generate(text, llm_tx).await;
            let _ = forwarder.await;
            if let Err(e) = result {
                let _ = tx.send(ChatEvent::LlmError(e.to_string())).await;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assistd_llm::{EchoBackend, FailedBackend};
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn test_app() -> (App, mpsc::Receiver<ChatEvent>) {
        let (tx, rx) = mpsc::channel::<ChatEvent>(16);
        let client: Arc<dyn LlmBackend> = Arc::new(EchoBackend);
        let app = App::new(client, tx, "test-model".into());
        (app, rx)
    }

    fn typed(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
    }

    #[test]
    fn delta_keeps_generating_true() {
        let (mut app, _rx) = test_app();
        app.generating = true;
        app.on_llm_event(ChatEvent::Llm(LlmEvent::Delta { text: "hi".into() }));
        assert!(app.generating);
    }

    #[test]
    fn done_clears_generating() {
        let (mut app, _rx) = test_app();
        app.generating = true;
        app.on_llm_event(ChatEvent::Llm(LlmEvent::Delta { text: "hi".into() }));
        app.on_llm_event(ChatEvent::Llm(LlmEvent::Done));
        assert!(!app.generating);
    }

    #[test]
    fn llm_error_clears_generating() {
        let (mut app, _rx) = test_app();
        app.generating = true;
        app.on_llm_event(ChatEvent::LlmError("boom".into()));
        assert!(!app.generating);
    }

    #[test]
    fn page_up_increments_scroll() {
        let (mut app, _rx) = test_app();
        app.last_output_height = 10;
        app.on_key(KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE));
        assert!(app.output.scroll_offset() > 0);
    }

    #[test]
    fn ctrl_c_on_empty_input_sets_quitting() {
        let (mut app, _rx) = test_app();
        app.on_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(app.should_quit());
    }

    #[test]
    fn enter_while_generating_sets_notice() {
        let (mut app, _rx) = test_app();
        app.generating = true;
        app.on_key(typed('h'));
        app.on_key(typed('i'));
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(app.notice().is_some());
    }

    #[test]
    fn on_tick_clears_stale_notice() {
        let (mut app, _rx) = test_app();
        app.notice = Some(("old".into(), Instant::now() - Duration::from_secs(10)));
        app.on_tick();
        assert!(app.notice().is_none());
    }

    #[test]
    fn spinner_char_cycles() {
        let (mut app, _rx) = test_app();
        let c0 = app.spinner_char();
        app.on_tick();
        let c1 = app.spinner_char();
        assert_ne!(c0, c1);
    }

    #[tokio::test]
    async fn echo_backend_spawn_yields_delta_then_done() {
        let (app, mut rx) = test_app();
        app.spawn_generation("hello".into());
        let first = rx.recv().await.expect("delta");
        let second = rx.recv().await.expect("done");
        match first {
            ChatEvent::Llm(LlmEvent::Delta { text }) => assert_eq!(text, "hello"),
            other => panic!("expected Delta, got {other:?}"),
        }
        assert!(matches!(second, ChatEvent::Llm(LlmEvent::Done)));
    }

    #[tokio::test]
    async fn failed_backend_spawn_yields_llm_error() {
        let (tx, mut rx) = mpsc::channel::<ChatEvent>(16);
        let client: Arc<dyn LlmBackend> = Arc::new(FailedBackend::new("server down".into()));
        let app = App::new(client, tx, "test-model".into());
        app.spawn_generation("hello".into());
        let ev = rx.recv().await.expect("error event");
        match ev {
            ChatEvent::LlmError(msg) => assert!(msg.contains("server down")),
            other => panic!("expected LlmError, got {other:?}"),
        }
    }
}
