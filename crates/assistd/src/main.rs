use anyhow::Result;
use clap::{Parser, Subcommand};

#[cfg(feature = "chat")]
mod chat;
#[cfg(feature = "daemon")]
mod daemon;
#[cfg(feature = "daemon")]
mod gpu_monitor;
#[cfg(any(feature = "daemon", feature = "chat"))]
mod hotkey;
#[cfg(any(feature = "daemon", feature = "chat"))]
mod idle_monitor;
#[cfg(feature = "client")]
mod listen;
#[cfg(feature = "daemon")]
mod listen_dispatcher;
#[cfg(feature = "client")]
mod presence;
#[cfg(feature = "client")]
mod ptt;
#[cfg(feature = "client")]
mod query;

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
    #[cfg(feature = "daemon")]
    Daemon(daemon::DaemonArgs),

    /// Write a default config file to ~/.config/assistd/config.toml
    #[cfg(feature = "daemon")]
    InitConfig,

    /// Send a one-shot query to a running assistd daemon
    #[cfg(feature = "client")]
    Query(query::QueryArgs),

    /// Drive a running daemon to Sleeping (stop llama-server, free all VRAM)
    #[cfg(feature = "client")]
    Sleep,

    /// Drive a running daemon to Drowsy (unload model weights, keep server alive)
    #[cfg(feature = "client")]
    Drowse,

    /// Drive a running daemon to Active (block until wake completes)
    #[cfg(feature = "client")]
    Wake,

    /// Advance the daemon one step along Active → Drowsy → Sleeping → Active
    #[cfg(feature = "client")]
    Cycle,

    /// Begin a push-to-talk recording on the running daemon (for i3
    /// `bindsym` — release handled by `ptt-stop` on the matching
    /// `bindsym --release` line).
    #[cfg(feature = "client")]
    PttStart,

    /// End the push-to-talk recording, transcribe, and dispatch as a query
    #[cfg(feature = "client")]
    PttStop,

    /// Enable hands-free continuous listening on the daemon
    #[cfg(feature = "client")]
    ListenStart,

    /// Disable hands-free continuous listening
    #[cfg(feature = "client")]
    ListenStop,

    /// Toggle hands-free continuous listening on/off
    #[cfg(feature = "client")]
    ListenToggle,

    /// Report whether continuous listening is currently active
    #[cfg(feature = "client")]
    ListenState,

    /// Open an interactive chat TUI
    #[cfg(feature = "chat")]
    Chat(chat::ChatArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        #[cfg(feature = "daemon")]
        Commands::Daemon(args) => daemon::run(args).await,
        #[cfg(feature = "daemon")]
        Commands::InitConfig => daemon::init_config(),
        #[cfg(feature = "client")]
        Commands::Query(args) => query::run(args).await,
        #[cfg(feature = "client")]
        Commands::Sleep => presence::run(presence::PresenceAction::Sleep).await,
        #[cfg(feature = "client")]
        Commands::Drowse => presence::run(presence::PresenceAction::Drowse).await,
        #[cfg(feature = "client")]
        Commands::Wake => presence::run(presence::PresenceAction::Wake).await,
        #[cfg(feature = "client")]
        Commands::Cycle => presence::run(presence::PresenceAction::Cycle).await,
        #[cfg(feature = "client")]
        Commands::PttStart => ptt::run(ptt::PttAction::Start).await,
        #[cfg(feature = "client")]
        Commands::PttStop => ptt::run(ptt::PttAction::Stop).await,
        #[cfg(feature = "client")]
        Commands::ListenStart => listen::run(listen::ListenAction::Start).await,
        #[cfg(feature = "client")]
        Commands::ListenStop => listen::run(listen::ListenAction::Stop).await,
        #[cfg(feature = "client")]
        Commands::ListenToggle => listen::run(listen::ListenAction::Toggle).await,
        #[cfg(feature = "client")]
        Commands::ListenState => listen::run(listen::ListenAction::State).await,
        #[cfg(feature = "chat")]
        Commands::Chat(args) => chat::run(args).await,
    }
}
