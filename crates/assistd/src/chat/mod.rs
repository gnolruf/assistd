//! Interactive ratatui-based chat TUI.
//!
//! `assistd chat` loads the existing config, starts its own llama-server
//! via `LlamaService`, creates a `LlamaChatClient`, and drives a
//! three-region terminal UI (output / status / input) with live streaming
//! tokens, readline-style input, and a VRAM/throughput status bar.

mod app;
mod input;
mod output;
mod throughput;
mod ui;
mod vram;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use assistd_core::Config;
use assistd_llm::{ChatSpec, LlamaChatClient, LlamaService, LlmBackend, ModelSpec, ServerSpec};
use clap::Args;
use crossterm::event::{self, Event, EventStream};
use crossterm::{cursor, execute, terminal};
use futures_util::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::{mpsc, watch};
use tracing::info;

use self::app::{App, ChatEvent};

#[derive(Args)]
pub struct ChatArgs {
    /// Path to config file [default: ~/.config/assistd/config.toml]
    #[arg(long, short)]
    pub config: Option<PathBuf>,
}

pub async fn run(args: ChatArgs) -> Result<()> {
    let config_path = match args.config {
        Some(p) => p,
        None => Config::default_path()?,
    };
    let config = Config::load_from_file(&config_path)?;
    config.validate()?;

    let _log_guard = init_file_tracing()?;

    info!("assistd chat v{}", assistd_core::version());
    info!("loaded config from {}", config_path.display());

    let (shutdown_tx, _) = watch::channel(false);
    install_signal_handler(shutdown_tx.clone());

    println!("loading model from {} ...", config.model.path);

    let llama = LlamaService::start(
        ServerSpec {
            binary_path: config.llama_server.binary_path.clone(),
            host: config.llama_server.host.clone(),
            port: config.llama_server.port,
            gpu_layers: config.llama_server.gpu_layers,
        },
        ModelSpec {
            path: config.model.path.clone(),
            vram_budget_mb: config.model.vram_budget_mb,
            context_length: config.model.context_length,
        },
        shutdown_tx.subscribe(),
    )
    .await?;
    info!(
        "llama-server ready on {}:{}",
        config.llama_server.host, config.llama_server.port
    );

    let chat_spec = ChatSpec {
        host: config.llama_server.host.clone(),
        port: config.llama_server.port,
        system_prompt: config.chat.system_prompt.clone(),
        max_history_tokens: config.chat.max_history_tokens,
        summary_target_tokens: config.chat.summary_target_tokens,
        preserve_recent_turns: config.chat.preserve_recent_turns,
        temperature: config.chat.temperature,
        max_response_tokens: config.chat.max_response_tokens,
        max_summary_tokens: config.chat.max_summary_tokens,
        request_timeout_secs: config.chat.request_timeout_secs,
        model_context_length: config.model.context_length,
    };
    let client: Arc<dyn LlmBackend> = Arc::new(LlamaChatClient::new(chat_spec)?);

    let mut vram_rx = vram::spawn_probe(shutdown_tx.subscribe());

    let model_name = Path::new(&config.model.path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "?".to_string());

    let run_result = run_tui(client, model_name, &mut vram_rx, shutdown_tx.clone()).await;

    let _ = shutdown_tx.send(true);
    if let Err(e) = llama.shutdown().await {
        tracing::error!("llama-server shutdown error: {e}");
    }
    info!("assistd chat stopped");
    run_result
}

async fn run_tui(
    client: Arc<dyn LlmBackend>,
    model_name: String,
    vram_rx: &mut watch::Receiver<vram::VramState>,
    shutdown_tx: watch::Sender<bool>,
) -> Result<()> {
    terminal::enable_raw_mode().context("enable_raw_mode")?;
    if let Err(e) = execute!(
        std::io::stdout(),
        terminal::EnterAlternateScreen,
        event::EnableMouseCapture
    ) {
        let _ = terminal::disable_raw_mode();
        return Err(e).context("EnterAlternateScreen");
    }

    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = TerminalGuard::cleanup();
        previous_hook(info);
    }));
    let _guard = TerminalGuard;

    let backend = CrosstermBackend::new(std::io::stdout());
    let mut terminal = Terminal::new(backend).context("Terminal::new")?;

    let (chat_tx, mut chat_rx) = mpsc::channel::<ChatEvent>(64);
    let mut app = App::new(client, chat_tx, model_name);

    let mut events = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(250));
    let mut shutdown_rx = shutdown_tx.subscribe();

    terminal.draw(|f| ui::render(f, &mut app))?;

    loop {
        if app.should_quit() {
            break;
        }
        tokio::select! {
            maybe_ev = events.next() => {
                match maybe_ev {
                    Some(Ok(Event::Key(k))) => app.on_key(k),
                    Some(Ok(Event::Resize(_, _))) => {}
                    Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        tracing::error!("terminal event error: {e}");
                        break;
                    }
                    None => break,
                }
            }
            Some(ev) = chat_rx.recv() => {
                app.on_llm_event(ev);
            }
            _ = tick.tick() => {
                app.on_tick();
            }
            Ok(_) = vram_rx.changed() => {
                let v = vram_rx.borrow_and_update().clone();
                app.on_vram(v);
            }
            _ = shutdown_rx.changed() => {
                break;
            }
        }
        terminal.draw(|f| ui::render(f, &mut app))?;
    }

    drop(terminal);
    Ok(())
}

fn install_signal_handler(shutdown_tx: watch::Sender<bool>) {
    tokio::spawn(async move {
        let mut term = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("failed to install SIGTERM handler: {e}");
                return;
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => info!("received SIGINT"),
            _ = term.recv() => info!("received SIGTERM"),
        }
        let _ = shutdown_tx.send(true);
    });
}

fn init_file_tracing() -> Result<tracing_appender::non_blocking::WorkerGuard> {
    use tracing_subscriber::{EnvFilter, fmt};

    let log_dir = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .map(|p| p.join("assistd"))
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/state/assistd")))
        .unwrap_or_else(std::env::temp_dir);
    std::fs::create_dir_all(&log_dir)
        .with_context(|| format!("creating log dir {}", log_dir.display()))?;

    let file_appender = tracing_appender::rolling::daily(&log_dir, "chat.log");
    let (writer, guard) = tracing_appender::non_blocking(file_appender);

    fmt()
        .with_writer(writer)
        .with_ansi(false)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    Ok(guard)
}

struct TerminalGuard;

impl TerminalGuard {
    fn cleanup() -> std::io::Result<()> {
        terminal::disable_raw_mode()?;
        execute!(
            std::io::stdout(),
            event::DisableMouseCapture,
            terminal::LeaveAlternateScreen,
            cursor::Show,
        )?;
        Ok(())
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = Self::cleanup();
    }
}
