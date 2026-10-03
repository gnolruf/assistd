use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use assistd_tools::{AlwaysAllowGate, Approval, Approvals, ConfirmationGate, DenyAllGate};
use parking_lot::Mutex;

use super::*;

/// [`adapt_client_as_tools`] for server `web` with every call allowed.
async fn adapt_allowing(
    client: Arc<dyn McpClient>,
    output: PresentSpec,
) -> Result<Vec<Box<dyn Tool>>, McpError> {
    let approvals = ApprovalGate::new(Arc::new(AlwaysAllowGate), Arc::new(Approvals::unsaved()));
    adapt_client_as_tools(client, "web", output, &approvals).await
}

fn mcp_tool(name: &str, description: &str) -> rmcp::model::Tool {
    serde_json::from_value(json!({
        "name": name,
        "description": description,
        "inputSchema": {"type": "object", "properties": {}}
    }))
    .unwrap()
}

fn call_result(result: Value) -> CallToolResult {
    serde_json::from_value(result).unwrap()
}

fn text_result(text: &str) -> CallToolResult {
    call_result(json!({"content": [{"type": "text", "text": text}]}))
}

/// Lists one `search` tool and echoes arguments back as text.
#[derive(Debug)]
struct FakeMcpClient;

#[async_trait]
impl McpClient for FakeMcpClient {
    async fn list_tools(&self) -> Result<Vec<rmcp::model::Tool>, McpError> {
        Ok(vec![mcp_tool("search", "search the web")])
    }

    async fn invoke(&self, name: &str, arguments: Value) -> Result<CallToolResult, McpError> {
        Ok(text_result(&format!("called {name} with {arguments}")))
    }
}

/// Fails its one `invoke` with a pre-armed error after `sleep`.
#[derive(Debug)]
struct ErrFakeClient {
    err: Mutex<Option<McpError>>,
    sleep: Duration,
}

#[async_trait]
impl McpClient for ErrFakeClient {
    async fn list_tools(&self) -> Result<Vec<rmcp::model::Tool>, McpError> {
        Ok(vec![mcp_tool("search", "search")])
    }

    async fn invoke(&self, _name: &str, _arguments: Value) -> Result<CallToolResult, McpError> {
        tokio::time::sleep(self.sleep).await;
        Err(self.err.lock().take().expect("err pre-armed"))
    }
}

async fn failing_tool(err: McpError, sleep: Duration) -> Box<dyn Tool> {
    let client = Arc::new(ErrFakeClient {
        err: Mutex::new(Some(err)),
        sleep,
    });
    let mut tools = adapt_allowing(client, PresentSpec::default())
        .await
        .unwrap();
    tools.pop().unwrap()
}

fn unlimited() -> TextTruncator {
    TextTruncator::new(PresentSpec::default(), "mcp-test")
}

#[tokio::test]
async fn adapter_forwards_tool_metadata_under_registry_name() {
    let tools = adapt_allowing(Arc::new(FakeMcpClient), PresentSpec::default())
        .await
        .unwrap();
    let [tool] = tools.as_slice() else {
        panic!("expected one tool");
    };
    assert_eq!(tool.name(), "mcp__web__search");
    assert_eq!(tool.description(), "search the web");
    assert_eq!(
        tool.parameters_schema(),
        json!({"type": "object", "properties": {}})
    );
}

#[tokio::test]
async fn adapter_invokes_the_server_native_name() {
    let tools = adapt_allowing(Arc::new(FakeMcpClient), PresentSpec::default())
        .await
        .unwrap();
    let out = tools[0].invoke(json!({"q": "rust"})).await.unwrap();
    assert_eq!(out["type"], "text");
    assert_eq!(out["output"], r#"called search with {"q":"rust"}"#);
    assert_eq!(out["truncated"], false);
}

#[tokio::test]
async fn adapter_cuts_long_results_and_spills_the_rest() {
    let dir = tempfile::tempdir().unwrap();
    let spec = PresentSpec {
        max_lines: 200,
        max_bytes: 16,
        overflow_dir: dir.path().to_path_buf(),
    };
    let tools = adapt_allowing(Arc::new(FakeMcpClient), spec).await.unwrap();
    let out = tools[0].invoke(json!({"q": "rust"})).await.unwrap();
    let spilled = dir.path().join("mcp-web-1.txt");
    assert_eq!(out["truncated"], true);
    assert_eq!(out["overflow_file"], json!(spilled.to_string_lossy()));
    let output = out["output"].as_str().unwrap();
    assert!(
        output.starts_with("called search wi\n--- output truncated"),
        "{output}"
    );
    assert_eq!(
        std::fs::read_to_string(spilled).unwrap(),
        r#"called search with {"q":"rust"}"#
    );
}

#[test]
fn text_and_empty_results_render_as_text_envelopes() {
    let text = |output: &str| {
        json!({
            "type": "text",
            "output": output,
            "exit_code": 0,
            "duration_ms": 42,
            "truncated": false,
        })
    };
    let cases = [
        (
            json!({"content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "x"}]}),
            text("hello"),
        ),
        (
            json!({"content": [{"type": "text", "text": "no such file"}], "isError": true}),
            text("[mcp tool error] no such file"),
        ),
        (json!({"content": []}), text("")),
    ];
    for (result, expected) in cases {
        assert_eq!(
            tool_result_to_json(call_result(result), 42, &unlimited()),
            expected
        );
    }
}

#[test]
fn image_and_other_results_render_as_their_own_envelopes() {
    let image = call_result(json!({
        "content": [{"type": "image", "mimeType": "image/png", "data": "3q2+7w=="}]
    }));
    assert_eq!(
        tool_result_to_json(image, 42, &unlimited()),
        json!({
            "type": "image",
            "output": "(image: image/png)",
            "exit_code": 0,
            "duration_ms": 42,
            "truncated": false,
            "attachments": [{"type": "image", "mime": "image/png", "data": "3q2+7w=="}],
        })
    );

    let link = call_result(json!({
        "content": [{"type": "resource_link", "uri": "file:///a.txt", "name": "a.txt"}]
    }));
    let envelope = tool_result_to_json(link, 42, &unlimited());
    assert_eq!(envelope["type"], "json");
    assert_eq!(envelope["value"]["uri"], "file:///a.txt");
    assert_eq!(envelope["output"], envelope["value"].to_string());
}

#[tokio::test]
async fn adapter_turns_client_errors_into_error_envelopes() {
    let err = || McpError::RpcError {
        code: -32602,
        message: "missing field 'query'".into(),
    };
    let tool = failing_tool(err(), Duration::ZERO).await;
    let mut out = tool.invoke(json!({})).await.unwrap();
    assert!(out["duration_ms"].is_u64(), "{out}");
    out.as_object_mut().unwrap().remove("duration_ms");
    assert_eq!(
        out,
        json!({
            "type": "error",
            "output": mcp_error_line("mcp__web__search", &err()),
            "exit_code": -1,
            "truncated": false,
        })
    );
}

#[tokio::test]
async fn adapter_records_real_duration_ms() {
    let tool = failing_tool(McpError::ServerDown, Duration::from_millis(20)).await;
    let out = tool.invoke(json!({})).await.unwrap();
    let dur = out["duration_ms"].as_u64().expect("duration_ms u64");
    assert!(
        dur >= 15,
        "duration must include the upstream sleep: got {dur}ms"
    );
}

/// Counts the calls that reach the server.
#[derive(Debug, Default)]
struct CountingClient {
    calls: AtomicUsize,
}

#[async_trait]
impl McpClient for CountingClient {
    async fn list_tools(&self) -> Result<Vec<rmcp::model::Tool>, McpError> {
        Ok(vec![mcp_tool("delete", "deletes a file")])
    }

    async fn invoke(&self, _name: &str, _arguments: Value) -> Result<CallToolResult, McpError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(text_result("deleted"))
    }
}

/// Answers every prompt with `answer`, recording each.
#[derive(Debug)]
struct AnsweringGate {
    answer: Approval,
    asked: Mutex<Vec<ConfirmationRequest>>,
}

#[async_trait]
impl ConfirmationGate for AnsweringGate {
    async fn confirm(&self, req: ConfirmationRequest) -> Approval {
        self.asked.lock().push(req);
        self.answer
    }
}

async fn gated_tool(
    gate: Arc<dyn ConfirmationGate>,
    approvals: Arc<Approvals>,
) -> (Box<dyn Tool>, Arc<CountingClient>) {
    let client = Arc::new(CountingClient::default());
    let mut tools = adapt_client_as_tools(
        client.clone(),
        "files",
        PresentSpec::default(),
        &ApprovalGate::new(gate, approvals),
    )
    .await
    .unwrap();
    (tools.pop().unwrap(), client)
}

#[tokio::test]
async fn declined_call_never_reaches_the_server() {
    let (tool, client) = gated_tool(Arc::new(DenyAllGate), Arc::new(Approvals::unsaved())).await;
    let out = tool.invoke(json!({"path": "/x"})).await.unwrap();
    assert_eq!(client.calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        out,
        json!({
            "type": "error",
            "output": "[error] mcp__files__delete: call cancelled by user. \
                       Try: a different approach\n",
            "exit_code": -1,
            "duration_ms": 0,
            "truncated": false,
        })
    );
}

#[tokio::test]
async fn call_asks_with_its_arguments_and_always_approves_the_tool() {
    let gate = Arc::new(AnsweringGate {
        answer: Approval::Always,
        asked: Mutex::new(Vec::new()),
    });
    let approvals = Arc::new(Approvals::unsaved());
    let (tool, client) = gated_tool(gate.clone(), Arc::clone(&approvals)).await;

    tool.invoke(json!({"path": "/x"})).await.unwrap();
    tool.invoke(json!({"path": "/y"})).await.unwrap();

    assert_eq!(client.calls.load(Ordering::SeqCst), 2);
    assert!(approvals.contains("mcp__files__delete"));
    let asked = gate.asked.lock();
    let [request] = asked.as_slice() else {
        panic!("only the first call should ask: {asked:?}");
    };
    assert_eq!(request.tool, "mcp__files__delete");
    assert_eq!(request.script, format!("{:#}", json!({"path": "/x"})));
    assert_eq!(request.always_allow, ["mcp__files__delete"]);
}

#[tokio::test]
async fn approved_tool_runs_without_asking() {
    let approvals = Arc::new(Approvals::unsaved());
    approvals
        .approve("mcp__files__delete")
        .await
        .expect("unsaved approve");
    let (tool, client) = gated_tool(Arc::new(DenyAllGate), approvals).await;
    tool.invoke(json!({})).await.unwrap();
    assert_eq!(client.calls.load(Ordering::SeqCst), 1);
}
