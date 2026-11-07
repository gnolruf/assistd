//! `mcp` subcommands: sign an OAuth MCP server in through the browser, sign it out, or list
//! every configured server's state, all through the daemon.

use std::env;
use std::io::{self, Write};
use std::process::{Child, Command, Stdio};

use anyhow::Result;
use assistd_ipc::{Event, McpServerState, McpTransportKind, Request, TokenRevocation};
use clap::{Args, Subcommand};
use uuid::Uuid;

use super::run_one_shot;
use super::terminal_text::escape_controls_single_line;

#[derive(Args)]
pub(crate) struct McpArgs {
    #[command(subcommand)]
    pub action: McpAction,
}

#[derive(Subcommand)]
pub(crate) enum McpAction {
    /// Sign the `auth = "oauth"` server `server` in: opens the sign-in page in a browser (and
    /// prints its URL), then waits until the daemon has the token and has restarted the server.
    Login { server: String },
    /// Stop server `server` and delete its token, revoking it where the server supports that.
    Logout { server: String },
    /// One tab-separated row per configured server: name, transport, and state.
    Status,
}

pub(crate) async fn run(args: McpArgs) -> Result<()> {
    match args.action {
        McpAction::Login { server } => login(server).await,
        McpAction::Logout { server } => logout(server).await,
        McpAction::Status => status().await,
    }
}

async fn login(server: String) -> Result<()> {
    let request = Request::McpLogin {
        id: Uuid::new_v4().to_string(),
        server,
    };
    let mut browser = None;
    let outcome = run_one_shot(request, |event| {
        match event {
            Event::McpAuthorizeUrl { server, url, .. } => {
                writeln!(
                    io::stdout(),
                    "Open this URL to sign in to MCP server '{}':\n\n  {}\n\nWaiting for the \
                     browser to finish…",
                    escape_controls_single_line(server),
                    escape_controls_single_line(url),
                )?;
                browser = open_browser(url);
            }
            Event::McpServer { server, state, .. } => writeln!(
                io::stdout(),
                "Signed in to '{}'; the server is {}.",
                escape_controls_single_line(server),
                describe_state(server, state),
            )?,
            _ => {}
        }
        Ok(())
    })
    .await;
    if let Some(mut browser) = browser {
        let _ = browser.wait();
    }
    outcome
}

async fn logout(server: String) -> Result<()> {
    let request = Request::McpLogout {
        id: Uuid::new_v4().to_string(),
        server,
    };
    run_one_shot(request, |event| {
        if let Event::McpLoggedOut {
            server, revocation, ..
        } = event
        {
            let server = escape_controls_single_line(server);
            let revoked = match revocation {
                TokenRevocation::Revoked => "revoked its token and deleted it".to_owned(),
                TokenRevocation::NotOffered => {
                    "deleted its token (the server offers no revocation)".to_owned()
                }
                TokenRevocation::NothingStored => "no token was stored".to_owned(),
                TokenRevocation::Failed { reason } => format!(
                    "deleted its token, but revoking it failed: {}",
                    escape_controls_single_line(reason)
                ),
            };
            writeln!(io::stdout(), "Signed out of '{server}': {revoked}.")?;
        }
        Ok(())
    })
    .await
}

async fn status() -> Result<()> {
    let request = Request::McpStatus {
        id: Uuid::new_v4().to_string(),
    };
    run_one_shot(request, |event| {
        if let Event::McpServer {
            server,
            transport,
            state,
            ..
        } = event
        {
            let transport = match transport {
                McpTransportKind::Stdio => "stdio",
                McpTransportKind::Http => "http",
            };
            writeln!(
                io::stdout(),
                "{}\t{transport}\t{}",
                escape_controls_single_line(server),
                describe_state(server, state),
            )?;
        }
        Ok(())
    })
    .await
}

/// `state` in a few words, with the command that signs `server` in when it needs that.
fn describe_state(server: &str, state: &McpServerState) -> String {
    match state {
        McpServerState::Starting => "starting".to_owned(),
        McpServerState::Ready => "ready".to_owned(),
        McpServerState::NeedsAuthorization { scopes } => {
            let scopes = match scopes.as_slice() {
                [] => String::new(),
                scopes => format!(", scopes: {}", scopes.join(" ")),
            };
            format!(
                "needs authorization{} (run `assistd mcp login {}`)",
                escape_controls_single_line(&scopes),
                escape_controls_single_line(server),
            )
        }
        McpServerState::Unavailable { reason } => {
            format!("unavailable: {}", escape_controls_single_line(reason))
        }
    }
}

/// Open `url` with `xdg-open` when there is a graphical session to open it in.
fn open_browser(url: &str) -> Option<Child> {
    let graphical = ["WAYLAND_DISPLAY", "DISPLAY"]
        .iter()
        .any(|var| env::var_os(var).is_some_and(|value| !value.is_empty()));
    if !graphical {
        return None;
    }
    Command::new("xdg-open")
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .ok()
}
