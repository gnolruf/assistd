//! `assistd` binary: CLI parsing and subcommand dispatch.

use std::io::{self, Write};
use std::process::ExitCode;
use std::time::Duration;

use anyhow::{Error, Result};
use assistd_ipc::{IpcClientError, PresenceTarget, Request, RequestBody};
use assistd_utils::terminal_text::escape_controls;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use client::{documents, listen, mcp, memory, presence, ptt, search, task, voice_ctl};

mod agents;
mod client;
mod daemon;
mod extract_worker;
#[cfg(feature = "tray")]
mod tray;
mod voice_worker;

/// How long exit waits on `spawn_blocking` work (e.g. a speech model
/// load abandoned by a shutdown during startup) before leaving it behind.
const BLOCKING_SHUTDOWN_GRACE: Duration = Duration::from_secs(2);

/// The workspace version, shared by every crate.
const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser)]
#[command(
    name = "assistd",
    version,
    about = "Local model agent OS assistant daemon"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the assistd daemon
    Daemon(daemon::DaemonArgs),

    /// Write the annotated sample config to ~/.config/assistd/config.toml
    InitConfig,

    /// Drive a running daemon to Sleeping (stop llama-server, free all VRAM)
    Sleep,

    /// Drive a running daemon to Drowsy (unload model weights, keep server alive)
    Drowse,

    /// Drive a running daemon to Active (block until wake completes)
    Wake,

    /// Begin a push-to-talk recording on the running daemon; bind to a
    /// key's press, and `ptt-stop` to its release.
    PttStart,

    /// End the push-to-talk recording, transcribe, and dispatch as a query
    PttStop,

    /// Enable hands-free continuous listening on the daemon
    ListenStart,

    /// Disable hands-free continuous listening
    ListenStop,

    /// Toggle hands-free continuous listening on/off
    ListenToggle,

    /// Report whether continuous listening is currently active
    ListenState,

    /// Toggle TTS playback on/off mid-session (off cancels current
    /// utterance; on resumes for the next sentence delivered).
    VoiceToggle,

    /// Abort the current TTS response: stops playback, drops any
    /// pending sentences for the active query. Does not start
    /// recording.
    VoiceSkip,

    /// Report whether TTS is currently enabled at runtime.
    VoiceState,

    /// Open an interactive chat TUI
    #[cfg(feature = "chat")]
    Chat(assistd_chat::ChatArgs),

    /// Edit the config file in a terminal settings panel; restart the
    /// daemon to apply saved changes
    #[cfg(feature = "chat")]
    Settings(assistd_chat::settings::SettingsArgs),

    /// Run the system-tray icon (StatusNotifierItem over DBus). Long-
    /// lived; suitable for compositor autostart (`exec assistd tray`).
    #[cfg(feature = "tray")]
    Tray(tray::TrayArgs),

    /// Search past sessions, saved memories and indexed documents at once, by meaning
    /// and by exact words. Prints one tab-separated line per hit, best first, as the
    /// model's `search` command lists them: kind, where the passage is, similarity,
    /// passage.
    Search(search::SearchArgs),

    /// Inspect or mutate persistent memory: search session
    /// transcripts, save / load / list / forget / delete key-value
    /// memories.
    Memory(memory::MemoryArgs),

    /// Report the document index, walk the indexed folders now, or search the
    /// indexed documents.
    Documents(documents::DocumentsArgs),

    /// List saved tasks with their next fire times, run one now, or cancel a run.
    Task(task::TaskArgs),

    /// Sign OAuth MCP servers in or out, or list every MCP server's state.
    Mcp(mcp::McpArgs),

    /// List the sub-agents `relay` would offer: the built-ins and the files in the
    /// `agents` directory beside the config, with each file that is refused and why.
    Agents(agents::AgentsArgs),

    /// Run the daemon's voice stack over stdin and stdout; the daemon
    /// spawns this itself.
    #[command(hide = true)]
    VoiceWorker,

    /// Extract document text over stdin and stdout; the daemon spawns this
    /// itself, in a sandbox.
    #[command(hide = true)]
    ExtractWorker,
}

/// The `RUST_LOG` filter when it is set and parses, else `default_directives`.
fn env_filter_or(default_directives: &str) -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_directives))
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(failure) => {
            print_failure(&failure);
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<()> {
    let runtime = tokio::runtime::Runtime::new()?;
    let result = runtime.block_on(dispatch(cli));
    runtime.shutdown_timeout(BLOCKING_SHUTDOWN_GRACE);
    result
}

/// Write `failure` to stderr: a daemon's error message on one line with its control
/// characters escaped, anything else with its chain of causes.
fn print_failure(failure: &Error) {
    let mut stderr = io::stderr().lock();
    let _ = match failure.downcast_ref() {
        Some(IpcClientError::Daemon(message)) => {
            writeln!(stderr, "daemon error: {}", escape_controls(message))
        }
        _ => writeln!(stderr, "Error: {failure:?}"),
    };
}

async fn dispatch(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Daemon(args) => daemon::run(args).await,
        Commands::InitConfig => daemon::init_config(),
        Commands::Sleep => presence::run(PresenceTarget::Sleeping).await,
        Commands::Drowse => presence::run(PresenceTarget::Drowsy).await,
        Commands::Wake => presence::run(PresenceTarget::Active).await,
        Commands::PttStart => ptt::run(|id| Request::new(id, RequestBody::PttStart)).await,
        Commands::PttStop => ptt::run(|id| Request::new(id, RequestBody::PttStop)).await,
        Commands::ListenStart => listen::run(|id| Request::new(id, RequestBody::ListenStart)).await,
        Commands::ListenStop => listen::run(|id| Request::new(id, RequestBody::ListenStop)).await,
        Commands::ListenToggle => {
            listen::run(|id| Request::new(id, RequestBody::ListenToggle)).await
        }
        Commands::ListenState => {
            listen::run(|id| Request::new(id, RequestBody::GetListenState)).await
        }
        Commands::VoiceToggle => {
            voice_ctl::run(|id| Request::new(id, RequestBody::VoiceToggle)).await
        }
        Commands::VoiceSkip => voice_ctl::run(|id| Request::new(id, RequestBody::VoiceSkip)).await,
        Commands::VoiceState => {
            voice_ctl::run(|id| Request::new(id, RequestBody::GetVoiceState)).await
        }
        #[cfg(feature = "chat")]
        Commands::Chat(args) => assistd_chat::run(args).await,
        #[cfg(feature = "chat")]
        Commands::Settings(args) => assistd_chat::settings::run(args).await,
        #[cfg(feature = "tray")]
        Commands::Tray(args) => tray::run(args).await,
        Commands::Search(args) => search::run(args).await,
        Commands::Memory(args) => memory::run(args).await,
        Commands::Documents(args) => documents::run(args).await,
        Commands::Task(args) => task::run(args).await,
        Commands::Mcp(args) => mcp::run(args).await,
        Commands::Agents(args) => agents::run(args),
        Commands::VoiceWorker => voice_worker::run().await,
        Commands::ExtractWorker => extract_worker::run().await,
    }
}
