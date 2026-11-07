//! End-to-end Streamable HTTP tests against an in-process `rmcp` server.

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use assistd_mcp::{
    Connector, HttpAuth, HttpConfig, McpClient, McpError, McpServer, OAuthConfig, TokenStore,
};
use axum::Router;
use axum::extract::Request;
use axum::http::{StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use parking_lot::Mutex;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, ListToolsResult,
    PaginatedRequestParams, ServerCapabilities, ServerConfig, Tool,
};
use rmcp::service::RequestContext;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::{ErrorData, RoleServer, ServerHandler};
use serde_json::json;
use tokio::net::TcpListener;
use tokio_util::task::AbortOnDropHandle;
use url::Url;

/// The client ID stored tokens are issued to.
const CLIENT_ID: &str = "assistd-test";

/// More text than one response may carry.
const OVERSIZED_REPLY_BYTES: usize = 65 * 1024 * 1024;

/// Answers `echo` with its `text` argument and `flood` with an oversized reply.
#[derive(Debug, Clone)]
struct EchoServer;

impl ServerHandler for EchoServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        let schema = json!({"type": "object"}).as_object().cloned().unwrap();
        Ok(ListToolsResult::with_all_items(vec![
            Tool::new("echo", "Echo text back", schema.clone()),
            Tool::new("flood", "Reply with too much text", schema),
        ]))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let text = match request.name.as_ref() {
            "flood" => "x".repeat(OVERSIZED_REPLY_BYTES),
            _ => request
                .arguments
                .and_then(|args| args.get("text")?.as_str().map(str::to_owned))
                .unwrap_or_default(),
        };
        Ok(CallToolResult::success(vec![ContentBlock::text(text)]).into())
    }
}

/// Serve `router` on a loopback port until the handle drops.
async fn serve(router: Router) -> (Url, AbortOnDropHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    let url = Url::parse(&format!("http://{addr}/mcp")).unwrap();
    (url, AbortOnDropHandle::new(task))
}

/// Serve [`EchoServer`]; with `json_response` it replies with plain JSON instead of SSE.
async fn serve_echo(json_response: bool) -> (Url, AbortOnDropHandle<()>) {
    let config = StreamableHttpServerConfig::default()
        .with_legacy_session_mode(!json_response)
        .with_json_response(json_response);
    let service: StreamableHttpService<EchoServer, LocalSessionManager> =
        StreamableHttpService::new(|| Ok(EchoServer), Arc::default(), config);
    serve(Router::new().nest_service("/mcp", service)).await
}

async fn start(url: Url) -> Result<McpServer, McpError> {
    McpServer::start(
        "remote".into(),
        Connector::Http(HttpConfig::new("remote", url)),
    )
    .await
}

fn first_text(result: &CallToolResult) -> Option<&str> {
    result.content.first().and_then(|block| match block {
        ContentBlock::Text(text) => Some(text.text.as_str()),
        _ => None,
    })
}

async fn echo(server: &McpServer, text: &str) -> Option<String> {
    let result = server.invoke("echo", json!({"text": text})).await.ok()?;
    first_text(&result).map(str::to_owned)
}

#[tokio::test(flavor = "multi_thread")]
async fn lists_and_calls_tools_and_survives_an_oversized_reply() {
    for json_response in [false, true] {
        let (url, _server) = serve_echo(json_response).await;
        let server = start(url).await.expect("server starts");
        let names: Vec<_> = server
            .list_tools()
            .await
            .unwrap()
            .into_iter()
            .map(|tool| tool.name)
            .collect();
        assert_eq!(names, ["echo", "flood"], "json_response = {json_response}");
        assert_eq!(echo(&server, "hi").await.as_deref(), Some("hi"));

        let flood = server.invoke("flood", json!({})).await;
        assert!(flood.is_err(), "json_response = {json_response}: {flood:?}");
        assert_eq!(
            echo(&server, "still here").await.as_deref(),
            Some("still here"),
            "json_response = {json_response}",
        );
        server.shutdown().await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_redirect_to_another_origin_is_refused() {
    let (target, _target_server) = serve_echo(false).await;
    let redirect = Router::new().fallback(move || {
        let location = target.to_string();
        async move {
            (
                StatusCode::TEMPORARY_REDIRECT,
                [(header::LOCATION, location)],
            )
                .into_response()
        }
    });
    let (url, _redirect_server) = serve(redirect).await;
    let err = start(url).await.expect_err("redirect is not followed");
    assert!(matches!(err, McpError::Connection(_)), "{err:?}");
}

/// An MCP endpoint that accepts only the access token `fresh`, beside an authorization server
/// that trades refresh token `r1` for it.
#[derive(Debug, Clone, Default)]
struct OAuthMock {
    bearers_seen: Arc<Mutex<Vec<String>>>,
    refreshes: Arc<AtomicUsize>,
}

impl OAuthMock {
    async fn serve(&self) -> (Url, AbortOnDropHandle<()>) {
        let config = StreamableHttpServerConfig::default().with_json_response(true);
        let service: StreamableHttpService<EchoServer, LocalSessionManager> =
            StreamableHttpService::new(|| Ok(EchoServer), Arc::default(), config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let metadata = json!({
            "issuer": origin,
            "authorization_endpoint": format!("{origin}/authorize"),
            "token_endpoint": format!("{origin}/token"),
            "response_types_supported": ["code"],
            "code_challenge_methods_supported": ["S256"],
        })
        .to_string();
        let refreshes = self.refreshes.clone();
        let bearers_seen = self.bearers_seen.clone();
        let mcp = Router::new()
            .nest_service("/mcp", service)
            .layer(middleware::from_fn(move |request, next| {
                require_fresh_token(bearers_seen.clone(), request, next)
            }));
        let router = mcp
            .route(
                "/.well-known/oauth-authorization-server",
                get(move || std::future::ready(json_reply(metadata.clone()))),
            )
            .route(
                "/token",
                post(move |body: String| std::future::ready(refresh(&refreshes, &body))),
            );
        let task = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let url = Url::parse(&format!("{origin}/mcp")).unwrap();
        (url, AbortOnDropHandle::new(task))
    }
}

async fn require_fresh_token(
    bearers_seen: Arc<Mutex<Vec<String>>>,
    request: Request,
    next: Next,
) -> Response {
    let bearer = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok()?.strip_prefix("Bearer "))
        .map(str::to_owned);
    bearers_seen.lock().extend(bearer.clone());
    if bearer.as_deref() == Some("fresh") {
        return next.run(request).await;
    }
    (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, r#"Bearer error="invalid_token""#)],
    )
        .into_response()
}

fn refresh(refreshes: &AtomicUsize, body: &str) -> Response {
    if !body.contains("grant_type=refresh_token") || !body.contains("refresh_token=r1") {
        return (StatusCode::BAD_REQUEST, r#"{"error":"invalid_grant"}"#).into_response();
    }
    refreshes.fetch_add(1, Ordering::SeqCst);
    json_reply(
        json!({
            "access_token": "fresh",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "r2",
        })
        .to_string(),
    )
}

fn json_reply(body: String) -> Response {
    ([(header::CONTENT_TYPE, "application/json")], body).into_response()
}

/// Write a token file holding access token `stale` and refresh token `r1` for server `remote`
/// at `url`, received at `received_at` seconds since the epoch and valid for an hour.
fn seed_token_file(path: &Path, url: &Url, received_at: u64) {
    let entries = json!({"entries": [{
        "server": "remote",
        "url": url.as_str(),
        "client_id": CLIENT_ID,
        "credentials": {
            "client_id": CLIENT_ID,
            "token_response": {
                "access_token": "stale",
                "token_type": "bearer",
                "expires_in": 3600,
                "refresh_token": "r1",
            },
            "granted_scopes": [],
            "token_received_at": received_at,
        },
    }]});
    std::fs::write(path, entries.to_string()).unwrap();
}

async fn start_oauth(url: Url, client_id: &str, tokens: &Path) -> Result<McpServer, McpError> {
    let mut http = HttpConfig::new("remote", url);
    http.auth = HttpAuth::OAuth(OAuthConfig {
        client_id: Some(client_id.to_owned()),
        client_secret: None,
        scopes: Vec::new(),
        tokens: TokenStore::new(tokens.to_path_buf()),
    });
    McpServer::start("remote".into(), Connector::Http(http)).await
}

#[tokio::test(flavor = "multi_thread")]
async fn an_expired_or_rejected_token_is_refreshed_and_the_request_succeeds() {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    for (label, received_at, stale_sent) in [("expired", 1, false), ("rejected", now, true)] {
        let mock = OAuthMock::default();
        let (url, _server) = mock.serve().await;
        let dir = tempfile::tempdir().unwrap();
        let tokens = dir.path().join("mcp-tokens.json");
        seed_token_file(&tokens, &url, received_at);

        let server = start_oauth(url, CLIENT_ID, &tokens)
            .await
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(echo(&server, "hi").await.as_deref(), Some("hi"), "{label}");
        assert_eq!(mock.refreshes.load(Ordering::SeqCst), 1, "{label}");
        let sent_stale = mock.bearers_seen.lock().iter().any(|b| b == "stale");
        assert_eq!(sent_stale, stale_sent, "{label}");

        let saved = std::fs::read_to_string(&tokens).unwrap();
        assert!(
            saved.contains(r#""access_token": "fresh""#),
            "{label}: {saved}"
        );
        let mode = std::fs::metadata(&tokens).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600, "{label}");
        server.shutdown().await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_token_is_not_sent_to_a_server_with_another_url_or_client_id() {
    let mock = OAuthMock::default();
    let (url, _server) = mock.serve().await;
    let dir = tempfile::tempdir().unwrap();
    let tokens = dir.path().join("mcp-tokens.json");
    seed_token_file(&tokens, &url.join("elsewhere").unwrap(), 1);
    let other_url = start_oauth(url.clone(), CLIENT_ID, &tokens).await;
    seed_token_file(&tokens, &url, 1);
    let other_client = start_oauth(url, "another-client", &tokens).await;

    for outcome in [other_url, other_client] {
        assert!(
            matches!(outcome, Err(McpError::NeedsAuthorization { .. })),
            "{outcome:?}"
        );
    }
    assert!(mock.bearers_seen.lock().is_empty());
    assert_eq!(mock.refreshes.load(Ordering::SeqCst), 0);
}
