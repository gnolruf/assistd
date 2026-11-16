//! `mcp` subcommands: sign an OAuth MCP server in through the browser, sign it out, or list
//! every configured server's state, all through the daemon.

use std::env;
use std::io::{self, Write};
use std::process::{Child, Command, Stdio};

use anyhow::Result;
use assistd_ipc::{
    EventBody, McpServerState, McpTransportKind, Request, RequestBody, TokenRevocation,
};
use assistd_utils::terminal_text::escape_controls_single_line;
use clap::{Args, Subcommand};
use tempfile::NamedTempFile;
use uuid::Uuid;

use super::run_one_shot;

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

/// A sign-in page handed to the browser as an owner-only file that forwards there, so the URL
/// and the `state` it carries stay out of every command line. Dropping it deletes the file.
struct OpenedSignIn {
    opener: Child,
    _forwarding_page: NamedTempFile,
}

pub(crate) async fn run(args: McpArgs) -> Result<()> {
    match args.action {
        McpAction::Login { server } => login(server).await,
        McpAction::Logout { server } => logout(server).await,
        McpAction::Status => status().await,
    }
}

async fn login(server: String) -> Result<()> {
    let request = Request::new(Uuid::new_v4().to_string(), RequestBody::McpLogin { server });
    let mut browser = None;
    let outcome = run_one_shot(request, |event| {
        match &event.body {
            EventBody::McpAuthorizeUrl { server, url } => {
                writeln!(
                    io::stdout(),
                    "Open this URL to sign in to MCP server '{}':\n\n  {}\n\nWaiting for the \
                     browser to finish…",
                    escape_controls_single_line(server),
                    escape_controls_single_line(url),
                )?;
                browser = open_browser(url);
            }
            EventBody::McpServer { server, state, .. } => writeln!(
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
    if let Some(mut opened) = browser {
        let _ = opened.opener.wait();
    }
    outcome
}

async fn logout(server: String) -> Result<()> {
    let request = Request::new(
        Uuid::new_v4().to_string(),
        RequestBody::McpLogout { server },
    );
    run_one_shot(request, |event| {
        if let EventBody::McpLoggedOut { server, revocation } = &event.body {
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
    let request = Request::new(Uuid::new_v4().to_string(), RequestBody::McpStatus);
    run_one_shot(request, |event| {
        if let EventBody::McpServer {
            server,
            transport,
            state,
        } = &event.body
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
fn open_browser(url: &str) -> Option<OpenedSignIn> {
    let graphical = ["WAYLAND_DISPLAY", "DISPLAY"]
        .iter()
        .any(|var| env::var_os(var).is_some_and(|value| !value.is_empty()));
    if !graphical {
        return None;
    }
    let forwarding_page = write_forwarding_page(url).ok()?;
    let opener = Command::new("xdg-open")
        .arg(forwarding_page.path())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    Some(OpenedSignIn {
        opener,
        _forwarding_page: forwarding_page,
    })
}

fn write_forwarding_page(url: &str) -> io::Result<NamedTempFile> {
    let mut page = tempfile::Builder::new()
        .prefix("assistd-mcp-login-")
        .suffix(".html")
        .tempfile_in(dirs::runtime_dir().unwrap_or_else(env::temp_dir))?;
    page.write_all(forwarding_page(url).as_bytes())?;
    Ok(page)
}

/// An HTML page that sends the browser on to `url` at once, with a link in case it does not.
fn forwarding_page(url: &str) -> String {
    let url = url
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;");
    format!(
        "<!doctype html>\n<meta charset=\"utf-8\">\n\
         <meta http-equiv=\"refresh\" content=\"0; url={url}\">\n\
         <title>assistd sign-in</title>\n\
         <a href=\"{url}\">Continue to sign in</a>\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forwarding_page_keeps_the_url_inside_its_attributes() {
        let page =
            forwarding_page("https://auth.example/authorize?a=1&state=\"><script>x</script>");
        assert!(
            page.contains("authorize?a=1&amp;state=&quot;&gt;&lt;script&gt;"),
            "{page}"
        );
        assert!(!page.contains("<script>"), "{page}");
    }
}
