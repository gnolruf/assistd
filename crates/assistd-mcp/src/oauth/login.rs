//! Signing a server in through the browser: an authorization-code flow with PKCE whose redirect
//! lands on a one-shot listener on a loopback port.

use std::fmt;
use std::io;
use std::net::Ipv4Addr;
use std::time::Duration;

use futures_util::StreamExt;
use futures_util::stream::FuturesUnordered;
use rmcp::transport::auth::{AuthorizationRequest, AuthorizationSession};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tracing::{debug, info};
use url::Url;

use super::{OAuthConfig, auth_failure, discovered_manager, is_web_endpoint};
use crate::error::McpError;

/// How long the user has to finish signing in once the authorization URL is handed out.
pub const LOGIN_TIMEOUT: Duration = Duration::from_secs(300);

/// Path on the loopback listener the authorization server redirects the browser to.
const CALLBACK_PATH: &str = "/callback";

/// Largest request head read from one browser connection.
const MAX_REQUEST_HEAD_BYTES: u64 = 16 * 1024;

/// How long one browser connection gets to send its request head.
const REQUEST_READ_TIMEOUT: Duration = Duration::from_secs(10);

/// Most connections read at once; further ones wait in the accept queue.
const MAX_OPEN_CONNECTIONS: usize = 16;

/// Client name sent when registering dynamically.
const CLIENT_NAME: &str = "assistd";

/// Body answering a callback whose `state` is not the pending sign-in's.
const FOREIGN_STATE_REPLY: &str = "assistd: this redirect does not belong to the pending sign-in.";

/// A sign-in waiting for the browser's redirect. Dropping it closes the listener and abandons
/// the flow. `Debug` never shows the state or the authorization URL.
pub struct PendingLogin {
    label: String,
    session: AuthorizationSession,
    listener: TcpListener,
    state: String,
}

/// The query of a request to [`CALLBACK_PATH`] that echoed the sign-in's `state`.
#[derive(Default)]
struct Callback {
    code: Option<String>,
    issuer: Option<String>,
    error: Option<String>,
}

impl PendingLogin {
    /// The URL the user opens to sign in.
    pub fn authorize_url(&self) -> &str {
        self.session.get_authorization_url()
    }

    /// Wait up to [`LOGIN_TIMEOUT`] for the redirect echoing this sign-in's `state`, then trade
    /// its code for tokens and store them. Redirects with another `state` are ignored; the first
    /// matching one ends the flow, failing it when it carries an error or no code.
    pub async fn complete(self) -> Result<(), McpError> {
        let (mut browser, callback) =
            tokio::time::timeout(LOGIN_TIMEOUT, next_callback(&self.listener, &self.state))
                .await
                .map_err(|_| {
                    McpError::Login(format!(
                        "no redirect from the browser within {}s",
                        LOGIN_TIMEOUT.as_secs()
                    ))
                })?;
        drop(self.listener);
        let outcome = redeem(&self.session, &self.state, callback).await;
        reply_to_browser(&mut browser, &self.label, &outcome).await;
        if outcome.is_ok() {
            info!(target: "assistd::mcp", server = %self.label, "MCP server signed in");
        }
        outcome
    }
}

impl fmt::Debug for PendingLogin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingLogin")
            .field("label", &self.label)
            .field("listener", &self.listener.local_addr().ok())
            .finish_non_exhaustive()
    }
}

/// Start signing server `label` at `url` in as `oauth`'s client: discover its authorization
/// server, register a client when none is configured, and listen on a loopback port for the
/// redirect. Bounded by `request_timeout`.
pub async fn begin_login(
    label: &str,
    url: &Url,
    oauth: &OAuthConfig,
    request_timeout: Duration,
) -> Result<PendingLogin, McpError> {
    let listen_failed =
        |e: io::Error| McpError::Login(format!("could not listen for the redirect: {e}"));
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .map_err(listen_failed)?;
    let port = listener.local_addr().map_err(listen_failed)?.port();
    let redirect_uri = format!("http://{}:{port}{CALLBACK_PATH}", Ipv4Addr::LOCALHOST);
    let session = tokio::time::timeout(
        request_timeout,
        authorization_session(label, url, oauth, redirect_uri),
    )
    .await
    .map_err(|_| McpError::RequestTimeout(request_timeout))??;
    let state = authorization_state(session.get_authorization_url())?;
    Ok(PendingLogin {
        label: label.to_owned(),
        session,
        listener,
        state,
    })
}

async fn authorization_session(
    label: &str,
    url: &Url,
    oauth: &OAuthConfig,
    redirect_uri: String,
) -> Result<AuthorizationSession, McpError> {
    let mut manager = discovered_manager(url, oauth).await?;
    manager.set_credential_store(
        oauth
            .tokens
            .for_server(label, url, oauth.client_id.as_deref()),
    );
    let mut request = AuthorizationRequest::new(redirect_uri)
        .with_scopes(oauth.scopes.clone())
        .with_client_name(CLIENT_NAME);
    if let Some(client_id) = &oauth.client_id {
        request = request.with_preregistered_client(client_id.clone());
    }
    if let Some(secret) = &oauth.client_secret {
        request = request.with_client_secret(secret.clone());
    }
    AuthorizationSession::new(manager, request)
        .await
        .map_err(|(_, e)| auth_failure(&e))
}

/// The `state` the authorization URL carries, which the redirect must echo. Fails unless the
/// URL is one a browser may be sent to, as the server's metadata chose it.
fn authorization_state(authorize_url: &str) -> Result<String, McpError> {
    let url = Url::parse(authorize_url)
        .ok()
        .filter(is_web_endpoint)
        .ok_or_else(|| McpError::OAuth("the authorization endpoint is not https".into()))?;
    url.query_pairs()
        .find(|(key, _)| key == "state")
        .map(|(_, state)| state.into_owned())
        .ok_or_else(|| McpError::OAuth("the authorization URL carries no state".into()))
}

/// Serve up to [`MAX_OPEN_CONNECTIONS`] connections at once until one delivers the callback
/// echoing `state`, so a client that sends nothing does not hold up the browser.
async fn next_callback(listener: &TcpListener, state: &str) -> (TcpStream, Callback) {
    let mut open_connections = FuturesUnordered::new();
    loop {
        tokio::select! {
            accepted = listener.accept(), if open_connections.len() < MAX_OPEN_CONNECTIONS => {
                match accepted {
                    Ok((stream, _)) => open_connections.push(callback_on(stream, state)),
                    Err(e) => {
                        debug!(target: "assistd::mcp", error = %e, "redirect listener accept failed");
                    }
                }
            }
            Some(answered) = open_connections.next() => {
                if let Some(delivered) = answered {
                    return delivered;
                }
            }
        }
    }
}

/// The callback `stream` asks for when its request is to [`CALLBACK_PATH`] echoing `state`. A
/// callback with another state is answered 400, every other request 404, and both are closed.
async fn callback_on(mut stream: TcpStream, state: &str) -> Option<(TcpStream, Callback)> {
    match read_request_target(&mut stream).await {
        Some(target) if target.path() == CALLBACK_PATH => {
            if let Some(callback) = Callback::echoing(state, &target) {
                return Some((stream, callback));
            }
            debug!(target: "assistd::mcp", "ignoring a redirect with another sign-in's state");
            write_response(&mut stream, "400 Bad Request", FOREIGN_STATE_REPLY).await;
        }
        _ => write_response(&mut stream, "404 Not Found", "Not found.").await,
    }
    None
}

/// The target of the GET request on `stream`, read whole so closing does not reset the
/// connection; `None` for anything else or a request not sent within [`REQUEST_READ_TIMEOUT`].
async fn read_request_target(stream: &mut TcpStream) -> Option<Url> {
    let mut head = BufReader::new(stream).take(MAX_REQUEST_HEAD_BYTES);
    let read_head = async {
        let mut request_line = String::new();
        head.read_line(&mut request_line).await.ok()?;
        let mut header = String::new();
        while head.read_line(&mut header).await.ok()? > 0 && !header.trim().is_empty() {
            header.clear();
        }
        Some(request_line)
    };
    let request_line = tokio::time::timeout(REQUEST_READ_TIMEOUT, read_head)
        .await
        .ok()??;
    let mut parts = request_line.split_whitespace();
    let (Some("GET"), Some(target)) = (parts.next(), parts.next()) else {
        return None;
    };
    target
        .starts_with('/')
        .then(|| Url::parse(&format!("http://{}{target}", Ipv4Addr::LOCALHOST)).ok())
        .flatten()
}

impl Callback {
    /// The callback in `target`'s query, or `None` unless that query's `state` is `state`.
    fn echoing(state: &str, target: &Url) -> Option<Self> {
        let mut callback = Self::default();
        let mut echoed_state = None;
        for (key, value) in target.query_pairs() {
            let slot = match key.as_ref() {
                "code" => &mut callback.code,
                "state" => &mut echoed_state,
                "iss" => &mut callback.issuer,
                "error" => &mut callback.error,
                _ => continue,
            };
            *slot = Some(value.into_owned());
        }
        (echoed_state.as_deref() == Some(state)).then_some(callback)
    }
}

/// Trade `callback`'s code for tokens, which the session's store keeps.
async fn redeem(
    session: &AuthorizationSession,
    state: &str,
    callback: Callback,
) -> Result<(), McpError> {
    if let Some(error) = callback.error {
        return Err(McpError::Login(format!(
            "the authorization server refused: {}",
            error.escape_debug()
        )));
    }
    let code = callback
        .code
        .ok_or_else(|| McpError::Login("the redirect carries no code".into()))?;
    session
        .handle_callback_with_issuer(&code, state, callback.issuer.as_deref())
        .await
        .map_err(|e| auth_failure(&e))?;
    Ok(())
}

async fn reply_to_browser(browser: &mut TcpStream, label: &str, outcome: &Result<(), McpError>) {
    let body = match outcome {
        Ok(()) => format!("assistd: signed in to MCP server '{label}'. You can close this tab."),
        Err(e) => format!("assistd: signing in to MCP server '{label}' failed: {e}"),
    };
    write_response(browser, "200 OK", &body).await;
}

async fn write_response(stream: &mut TcpStream, status: &str, body: &str) {
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\n\
         Content-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\n\r\n{body}",
        body.len()
    );
    if let Err(e) = stream.write_all(response.as_bytes()).await {
        debug!(target: "assistd::mcp", error = %e, "could not answer the browser");
    }
    let _ = stream.shutdown().await;
}
