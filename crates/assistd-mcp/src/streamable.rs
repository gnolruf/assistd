//! Remote MCP servers spoken to over Streamable HTTP, with the same size and time limits stdio
//! servers get and no redirect off the configured origin.

use std::fmt;
use std::time::Duration;

use http::HeaderMap;
use reqwest::redirect;
use rmcp::ServiceExt;
use rmcp::model::ClientConfig;
use rmcp::service::{ClientInitializeError, RoleClient, RunningService};
use rmcp::transport::auth::AuthClient;
use rmcp::transport::streamable_http_client::{
    StreamableHttpClient, StreamableHttpClientTransportConfig, StreamableHttpError,
};
use rmcp::transport::{DynamicTransportError, StreamableHttpClientTransport};
use tracing::info;
use url::Url;

use assistd_utils::http::remote_client_builder;

use crate::error::McpError;
use crate::oauth::{self, OAuthConfig, auth_failure, split_scopes};
use crate::session::{MAX_MESSAGE_BYTES, McpPeer, client_config, session_end};

mod client;

use client::{CappedHttpClient, HttpClientError, HttpError};

/// Most same-origin redirects followed for one request.
const MAX_REDIRECTS: usize = 10;

/// Per-server Streamable HTTP configuration. `Debug` gives the URL's origin and the header
/// names, never the path, query, header values or secrets.
#[derive(Clone)]
pub struct HttpConfig {
    pub url: Url,
    /// Sent with every request.
    pub headers: HeaderMap,
    pub auth: HttpAuth,
    /// Bounds the initialize handshake and every request.
    pub request_timeout: Duration,
    /// Server name used in tracing logs.
    pub label: String,
}

impl HttpConfig {
    /// Config that reaches `url` with no extra headers and a 30s request timeout.
    pub fn new(label: impl Into<String>, url: Url) -> Self {
        Self {
            url,
            headers: HeaderMap::new(),
            auth: HttpAuth::None,
            request_timeout: Duration::from_secs(30),
            label: label.into(),
        }
    }

    fn origin(&self) -> String {
        self.url.origin().ascii_serialization()
    }
}

impl fmt::Debug for HttpConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpConfig")
            .field("origin", &self.origin())
            .field("header_names", &self.headers.keys().collect::<Vec<_>>())
            .field("auth", &self.auth)
            .field("request_timeout", &self.request_timeout)
            .field("label", &self.label)
            .finish()
    }
}

/// How a server authenticates the daemon beyond its static headers.
#[derive(Debug, Clone, Default)]
pub enum HttpAuth {
    #[default]
    None,
    /// A bearer token from the [`TokenStore`](crate::TokenStore), refreshed ahead of expiry
    /// and once when the server rejects it.
    OAuth(OAuthConfig),
}

/// The open HTTP session. Closing it asks the server to delete the session.
#[derive(Debug)]
pub(crate) struct HttpLifeline {
    service: Option<RunningService<RoleClient, ClientConfig>>,
}

impl HttpLifeline {
    /// Resolves when the transport gives up on the session.
    pub(crate) async fn wait_for_end(&mut self) {
        session_end(self.service.take()).await;
    }

    pub(crate) async fn shutdown(self, term_timeout: Duration) {
        if let Some(mut service) = self.service {
            let _ = service.close_with_timeout(term_timeout).await;
        }
    }
}

/// Open a session to the server and run the initialize handshake; errors if either fails.
pub(crate) async fn connect(cfg: &HttpConfig) -> Result<(McpPeer, HttpLifeline), McpError> {
    let http = remote_client_builder()
        .redirect(same_origin_redirects())
        .connect_timeout(cfg.request_timeout)
        .build()
        .map_err(|e| McpError::Connection(error_chain(&e.without_url())))?;
    let http = CappedHttpClient::new(http);
    let service = match &cfg.auth {
        HttpAuth::None => initialize(cfg, http).await?,
        HttpAuth::OAuth(oauth) => initialize(cfg, authorized(cfg, oauth, http).await?).await?,
    };
    info!(
        target: "assistd::mcp",
        server = %cfg.label,
        origin = %cfg.origin(),
        "MCP Streamable HTTP server initialized",
    );
    let peer = McpPeer::new(service.peer().clone(), cfg.request_timeout);
    let lifeline = HttpLifeline {
        service: Some(service),
    };
    Ok((peer, lifeline))
}

/// `http` with the server's stored OAuth token attached to every request.
async fn authorized(
    cfg: &HttpConfig,
    oauth: &OAuthConfig,
    http: CappedHttpClient,
) -> Result<AuthClient<CappedHttpClient>, McpError> {
    let manager = tokio::time::timeout(
        cfg.request_timeout,
        oauth::authorization_manager(&cfg.label, &cfg.url, oauth),
    )
    .await
    .map_err(|_| McpError::RequestTimeout(cfg.request_timeout))??;
    Ok(AuthClient::new(http, manager))
}

async fn initialize<C>(
    cfg: &HttpConfig,
    client: C,
) -> Result<RunningService<RoleClient, ClientConfig>, McpError>
where
    C: StreamableHttpClient<Error = HttpClientError> + Sync,
{
    let transport_config = StreamableHttpClientTransportConfig::with_uri(cfg.url.as_str())
        .custom_headers(
            cfg.headers
                .iter()
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect(),
        )
        .max_sse_event_size(MAX_MESSAGE_BYTES)
        .reinit_on_expired_session(false);
    let transport = StreamableHttpClientTransport::with_client(client, transport_config);
    tokio::time::timeout(cfg.request_timeout, client_config().serve(transport))
        .await
        .map_err(|_| McpError::RequestTimeout(cfg.request_timeout))?
        .map_err(initialize_error)
}

/// The [`McpError`] for a failed HTTP request, or `None` when `error` came from another
/// transport.
pub(crate) fn transport_failure(error: &DynamicTransportError) -> Option<McpError> {
    let error = error.error.downcast_ref::<HttpError>()?;
    Some(match error {
        StreamableHttpError::AuthRequired(_) => McpError::NeedsAuthorization { scopes: Vec::new() },
        StreamableHttpError::InsufficientScope(insufficient) => McpError::NeedsAuthorization {
            scopes: insufficient
                .required_scope
                .as_deref()
                .map(split_scopes)
                .unwrap_or_default(),
        },
        StreamableHttpError::Auth(failure) => auth_failure(failure),
        StreamableHttpError::HttpStatus(status) => McpError::HttpStatus(status.status),
        StreamableHttpError::SessionExpired => McpError::SessionExpired,
        StreamableHttpError::Client(HttpClientError::Request(source)) => {
            McpError::Connection(error_chain(source))
        }
        other => McpError::Protocol(other.to_string()),
    })
}

/// Follow 307 and 308 redirects within the configured origin; refuse any to another origin and
/// stop at the rest, which change the method.
fn same_origin_redirects() -> redirect::Policy {
    redirect::Policy::custom(|attempt| {
        let same_origin = attempt
            .previous()
            .first()
            .is_some_and(|original| original.origin() == attempt.url().origin());
        let keeps_method = matches!(attempt.status().as_u16(), 307 | 308);
        if !same_origin {
            attempt.error("redirect to another origin refused")
        } else if !keeps_method {
            attempt.stop()
        } else if attempt.previous().len() > MAX_REDIRECTS {
            attempt.error(format!("more than {MAX_REDIRECTS} redirects"))
        } else {
            attempt.follow()
        }
    })
}

fn initialize_error(error: ClientInitializeError) -> McpError {
    match &error {
        ClientInitializeError::TransportError { error, .. } => transport_failure(error),
        _ => None,
    }
    .unwrap_or_else(|| McpError::Initialize(Box::new(error)))
}

/// `error` and each of its sources, joined by `: `.
fn error_chain(error: &reqwest::Error) -> String {
    std::iter::successors(Some(error as &dyn std::error::Error), |e| e.source())
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(": ")
}
