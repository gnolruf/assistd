use anyhow::Result;
use clap::{Parser, Subcommand};

#[cfg(feature = "daemon")]
mod daemon;
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
    }
}
