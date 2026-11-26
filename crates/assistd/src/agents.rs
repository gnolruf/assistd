//! `assistd agents`: the sub-agents a daemon started now would offer, and the agent
//! files it would refuse.

use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::Result;
use assistd_config::{AgentCatalog, AgentOrigin, AgentSpec, Config};
use assistd_utils::terminal_text::escape_controls_single_line;
use clap::Args;

#[derive(Debug, Args)]
pub(crate) struct AgentsArgs {
    /// Path to config file [default: ~/.config/assistd/config.toml]
    #[arg(long, short)]
    pub config: Option<PathBuf>,
}

/// Print one tab-separated line per sub-agent (name, origin, what it reaches,
/// description), then one line per refused agent file on stderr. Fails when the config
/// does not load.
pub(crate) fn run(args: AgentsArgs) -> Result<()> {
    let config_path = Config::path_or_default(args.config)?;
    let config = Config::load(&config_path)?;
    let catalog = AgentCatalog::load(&AgentCatalog::dir_beside(&config_path), &config);
    let mut stdout = io::stdout().lock();
    for agent in &catalog.agents {
        writeln!(
            stdout,
            "{}\t{}\t{}\t{}",
            agent.name.as_str(),
            origin(agent),
            reach(agent),
            escape_controls_single_line(&agent.description)
        )?;
    }
    let mut stderr = io::stderr().lock();
    for refused in &catalog.refused {
        writeln!(
            stderr,
            "refused: {}",
            escape_controls_single_line(&refused.to_string())
        )?;
    }
    Ok(())
}

fn origin(agent: &AgentSpec) -> String {
    match &agent.origin {
        AgentOrigin::BuiltIn => "built-in".to_owned(),
        AgentOrigin::File(path) => path.display().to_string(),
    }
}

/// The agent's capabilities and MCP servers, comma-separated; `-` when it has none.
fn reach(agent: &AgentSpec) -> String {
    let capabilities = &agent.capabilities;
    let servers = capabilities
        .mcp_servers
        .iter()
        .map(|server| format!("mcp:{}", server.as_str()));
    let reach: Vec<String> = capabilities
        .names()
        .into_iter()
        .map(str::to_owned)
        .chain(servers)
        .collect();
    if reach.is_empty() {
        "-".to_owned()
    } else {
        reach.join(",")
    }
}
