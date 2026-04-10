use anyhow::Result;
use clap::Parser;
use tracing::info;

use assistd_core::Config;

#[derive(Parser)]
#[command(
    name = "assistd",
    version,
    about = "Local model agent OS assistant daemon"
)]
struct Cli {
    /// Generate a default config file at ~/.config/assistd/config.toml
    #[arg(long)]
    init_config: bool,

    /// Path to config file [default: ~/.config/assistd/config.toml]
    #[arg(long, short)]
    config: Option<std::path::PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    if cli.init_config {
        let path = Config::default_path()?;
        Config::write_default(&path)?;
        info!("wrote default config to {}", path.display());
        return Ok(());
    }

    let config_path = match cli.config {
        Some(p) => p,
        None => Config::default_path()?,
    };
    let config = Config::load_from_file(&config_path)?;
    config.validate()?;

    info!(
        "assistd v{} — local model agent OS assistant daemon",
        assistd_core::version()
    );
    info!("  core  v{}", assistd_core::version());
    info!("  llm   v{}", assistd_llm::version());
    info!("  voice v{}", assistd_voice::version());
    info!("  tools v{}", assistd_tools::version());
    info!("  wm    v{}", assistd_wm::version());
    info!("  tui   v{}", assistd_tui::version());
    info!("loaded config from {}", config_path.display());

    info!("all subsystems registered — exiting (no runtime loop yet)");
    Ok(())
}
