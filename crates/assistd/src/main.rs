//! `assistd` binary: CLI parsing and subcommand dispatch.

use std::time::Duration;

use anyhow::Result;
use assistd_ipc::{PresenceTarget, Request};
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use client::{listen, mcp, memory, presence, ptt, query, task, voice_ctl};

#[cfg(feature = "chat")]
mod chat;
mod client;
mod daemon;
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

    /// Write a default config file to ~/.config/assistd/config.toml
    InitConfig,

    /// Send a one-shot query to a running assistd daemon
    Query(query::QueryArgs),

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
    Chat(chat::ChatArgs),

    /// Edit the config file in a terminal settings panel; restart the
    /// daemon to apply saved changes
    #[cfg(feature = "chat")]
    Settings(chat::settings::SettingsArgs),

    /// Run the system-tray icon (StatusNotifierItem over DBus). Long-
    /// lived; suitable for compositor autostart (`exec assistd tray`).
    #[cfg(feature = "tray")]
    Tray(tray::TrayArgs),

    /// Inspect or mutate persistent memory: search session
    /// transcripts, save / load / list / forget / delete key-value
    /// memories.
    Memory(memory::MemoryArgs),

    /// List saved tasks with their next fire times, run one now, or cancel a run.
    Task(task::TaskArgs),

    /// Sign OAuth MCP servers in or out, or list every MCP server's state.
    Mcp(mcp::McpArgs),

    /// Run the daemon's voice stack over stdin and stdout; the daemon
    /// spawns this itself.
    #[command(hide = true)]
    VoiceWorker,
}

/// The `RUST_LOG` filter when it is set and parses, else `default_directives`.
fn env_filter_or(default_directives: &str) -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_directives))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let runtime = tokio::runtime::Runtime::new()?;
    let result = runtime.block_on(dispatch(cli));
    runtime.shutdown_timeout(BLOCKING_SHUTDOWN_GRACE);
    result
}

async fn dispatch(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Daemon(args) => daemon::run(args).await,
        Commands::InitConfig => daemon::init_config(),
        Commands::Query(args) => query::run(args).await,
        Commands::Sleep => presence::run(PresenceTarget::Sleeping).await,
        Commands::Drowse => presence::run(PresenceTarget::Drowsy).await,
        Commands::Wake => presence::run(PresenceTarget::Active).await,
        Commands::PttStart => ptt::run(|id| Request::PttStart { id }).await,
        Commands::PttStop => ptt::run(|id| Request::PttStop { id }).await,
        Commands::ListenStart => listen::run(|id| Request::ListenStart { id }).await,
        Commands::ListenStop => listen::run(|id| Request::ListenStop { id }).await,
        Commands::ListenToggle => listen::run(|id| Request::ListenToggle { id }).await,
        Commands::ListenState => listen::run(|id| Request::GetListenState { id }).await,
        Commands::VoiceToggle => voice_ctl::run(|id| Request::VoiceToggle { id }).await,
        Commands::VoiceSkip => voice_ctl::run(|id| Request::VoiceSkip { id }).await,
        Commands::VoiceState => voice_ctl::run(|id| Request::GetVoiceState { id }).await,
        #[cfg(feature = "chat")]
        Commands::Chat(args) => chat::run(args).await,
        #[cfg(feature = "chat")]
        Commands::Settings(args) => chat::settings::run(args).await,
        #[cfg(feature = "tray")]
        Commands::Tray(args) => tray::run(args).await,
        Commands::Memory(args) => memory::run(args).await,
        Commands::Task(args) => task::run(args).await,
        Commands::Mcp(args) => mcp::run(args).await,
        Commands::VoiceWorker => voice_worker::run().await,
    }
}
