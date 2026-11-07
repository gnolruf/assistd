//! End-to-end Streamable HTTP tests against an in-process `rmcp` server.

use std::sync::Arc;

use assistd_mcp::{Connector, HttpConfig, McpClient, McpError, McpServer};
use axum::Router;
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
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
