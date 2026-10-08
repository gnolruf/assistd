//! `assistd` binary: CLI parsing and subcommand dispatch.

use std::time::Duration;

use anyhow::Result;
use clap::{Parser, Subcommand};

use client::{listen, memory, presence, ptt, query, voice_ctl};

#[cfg(feature = "chat")]
mod chat;
mod client;
mod daemon;
mod hotkey;
mod ipc_voice_proxy;
#[cfg(feature = "tray")]
mod tray;
mod wm_backend;

/// How long exit waits on `spawn_blocking` work (e.g. a Whisper model
/// load abandoned by a shutdown during startup) before leaving it behind.
const BLOCKING_SHUTDOWN_GRACE: Duration = Duration::from_secs(2);

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

    /// Advance the daemon one step along Active → Drowsy → Sleeping → Active
    Cycle,

    /// Begin a push-to-talk recording on the running daemon (for i3
    /// `bindsym`; release handled by `ptt-stop` on the matching
    /// `bindsym --release` line).
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

    /// Run the system-tray icon (StatusNotifierItem over DBus). Long-
    /// lived; suitable for compositor autostart (`exec assistd tray`).
    #[cfg(feature = "tray")]
    Tray(tray::TrayArgs),

    /// Inspect or mutate persistent memory: search conversation
    /// history, save / load / list / forget / delete key-value
    /// memories.
    Memory(memory::MemoryArgs),
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
        Commands::Sleep => presence::run(presence::PresenceAction::Sleep).await,
        Commands::Drowse => presence::run(presence::PresenceAction::Drowse).await,
        Commands::Wake => presence::run(presence::PresenceAction::Wake).await,
        Commands::Cycle => presence::run(presence::PresenceAction::Cycle).await,
        Commands::PttStart => ptt::run(ptt::PttAction::Start).await,
        Commands::PttStop => ptt::run(ptt::PttAction::Stop).await,
        Commands::ListenStart => listen::run(listen::ListenAction::Start).await,
        Commands::ListenStop => listen::run(listen::ListenAction::Stop).await,
        Commands::ListenToggle => listen::run(listen::ListenAction::Toggle).await,
        Commands::ListenState => listen::run(listen::ListenAction::State).await,
        Commands::VoiceToggle => voice_ctl::run(voice_ctl::VoiceCtlAction::Toggle).await,
        Commands::VoiceSkip => voice_ctl::run(voice_ctl::VoiceCtlAction::Skip).await,
        Commands::VoiceState => voice_ctl::run(voice_ctl::VoiceCtlAction::State).await,
        #[cfg(feature = "chat")]
        Commands::Chat(args) => chat::run(args).await,
        #[cfg(feature = "tray")]
        Commands::Tray(args) => tray::run(args).await,
        Commands::Memory(args) => memory::run(args).await,
    }
}
