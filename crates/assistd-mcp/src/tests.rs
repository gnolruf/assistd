use std::sync::atomic::{AtomicUsize, Ordering};

use assistd_tools::{AlwaysAllowGate, Approval, Approvals, ConfirmationGate, DenyAllGate};
use parking_lot::Mutex;
use serde_json::json;

use super::*;

/// [`adapt_client_as_tools`] for server `web` with every call allowed.
async fn adapt_allowing(client: Arc<dyn McpClient>) -> Result<Vec<Box<dyn Tool>>, McpError> {
    let approvals = ApprovalGate::new(Arc::new(AlwaysAllowGate), Arc::new(Approvals::unsaved()));
    adapt_client_as_tools(
        client,
        &"web".parse().unwrap(),
        &approvals,
        &VisionGate::new(true),
    )
    .await
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

/// Fails its one `invoke` with a pre-armed error.
#[derive(Debug)]
struct ErrFakeClient {
    err: Mutex<Option<McpError>>,
}

#[async_trait]
impl McpClient for ErrFakeClient {
    async fn list_tools(&self) -> Result<Vec<rmcp::model::Tool>, McpError> {
        Ok(vec![mcp_tool("search", "search")])
    }

    async fn invoke(&self, _name: &str, _arguments: Value) -> Result<CallToolResult, McpError> {
        Err(self.err.lock().take().expect("err pre-armed"))
    }
}

async fn failing_tool(err: McpError) -> Box<dyn Tool> {
    let client = Arc::new(ErrFakeClient {
        err: Mutex::new(Some(err)),
    });
    let mut tools = adapt_allowing(client).await.unwrap();
    tools.pop().unwrap()
}

fn image_bytes(attachments: &[Attachment]) -> Vec<(ImageMime, &[u8])> {
    attachments
        .iter()
        .map(|Attachment::Image { mime, bytes }| (*mime, bytes.as_slice()))
        .collect()
}

#[test]
fn text_only_results_join_every_block() {
    let cases = [
        (
            json!({"content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "x"}]}),
            "hello\nx",
        ),
        (
            json!({"content": [{"type": "text", "text": "no such file"}], "isError": true}),
            "[mcp tool error] no such file",
        ),
        (json!({"content": []}), ""),
        (
            json!({"content": [], "structuredContent": {"temp": 21}}),
            r#"{"temp":21}"#,
        ),
        (
            json!({"content": [{"type": "text", "text": "21"}], "structuredContent": {"temp": 21}}),
            "21",
        ),
    ];
    for (result, expected) in cases {
        let output = tool_output(call_result(result), Duration::from_millis(42), true);
        assert_eq!(output.output, expected);
        assert_eq!(output.exit_code, 0);
        assert!(output.attachments.is_empty());
    }
}

#[test]
fn mixed_results_keep_every_block_and_collect_images() {
    let result = call_result(json!({
        "content": [
            {"type": "text", "text": "took screenshot"},
            {"type": "image", "mimeType": "image/png", "data": "3q2+7w=="},
            {"type": "resource_link", "uri": "file:///a.txt", "name": "a.txt"},
            {"type": "image", "mimeType": "image/jpeg", "data": "AAAA"},
            {"type": "image", "mimeType": "image/gif", "data": "AAAA"},
            {"type": "image", "mimeType": "image/png", "data": "not base64!"},
        ]
    }));
    let result = tool_output(result, Duration::from_millis(42), true);
    let lines: Vec<&str> = result.output.lines().collect();
    let [text, png, link, jpeg, gif, garbled] = lines.as_slice() else {
        panic!("expected six sections: {}", result.output);
    };
    assert_eq!(*text, "took screenshot");
    assert_eq!(*png, "(image: image/png)");
    assert_eq!(*link, "(resource link: a.txt, file:///a.txt)");
    assert_eq!(*jpeg, "(image: image/jpeg)");
    assert_eq!(*gif, "(image omitted: image/gif; not PNG, JPEG or WebP)");
    assert!(
        garbled.starts_with("(image omitted: image/png; invalid base64"),
        "{garbled}"
    );
    assert_eq!(
        image_bytes(&result.attachments),
        [
            (ImageMime::Png, [0xDE, 0xAD, 0xBE, 0xEF].as_slice()),
            (ImageMime::Jpeg, [0, 0, 0].as_slice()),
        ]
    );
}

#[test]
fn resource_blocks_render_as_readable_text() {
    let result = call_result(json!({
        "content": [
            {"type": "text", "text": "2 matches"},
            {
                "type": "resource_link", "uri": "drive://doc/42", "name": "Q3 plan",
                "mimeType": "text/markdown", "size": 2048, "description": "ignored"
            },
            {
                "type": "resource",
                "resource": {
                    "uri": "drive://doc/42", "mimeType": "text/markdown",
                    "text": "# Q3 plan\n\n- ship \"relay\"\n"
                }
            },
            {
                "type": "resource",
                "resource": {"uri": "drive://doc/7", "mimeType": "application/pdf", "blob": "JVBERi0="}
            },
            {
                "type": "resource",
                "resource": {"uri": "drive://img/1", "mimeType": "image/png", "blob": "3q2+7w=="}
            },
        ]
    }));
    let output = tool_output(result, Duration::from_millis(42), true);
    assert_eq!(
        output.output,
        "2 matches\n\
         (resource link: Q3 plan, drive://doc/42, text/markdown, 2KB)\n\
         (resource: drive://doc/42, text/markdown)\n\
         # Q3 plan\n\n- ship \"relay\"\n\n\
         (binary resource omitted: application/pdf, 5B)\n\
         (image: image/png)"
    );
    assert_eq!(
        image_bytes(&output.attachments),
        [(ImageMime::Png, [0xDE, 0xAD, 0xBE, 0xEF].as_slice())]
    );
}

/// Returns one PNG image for every call.
#[derive(Debug)]
struct ImageClient;

#[async_trait]
impl McpClient for ImageClient {
    async fn list_tools(&self) -> Result<Vec<rmcp::model::Tool>, McpError> {
        Ok(vec![mcp_tool("snap", "take a screenshot")])
    }

    async fn invoke(&self, _name: &str, _arguments: Value) -> Result<CallToolResult, McpError> {
        Ok(call_result(json!({
            "content": [{"type": "image", "mimeType": "image/png", "data": "AAAA"}]
        })))
    }
}

#[tokio::test]
async fn adapter_follows_the_vision_gate_on_every_call() {
    let vision = VisionGate::new(false);
    let approvals = ApprovalGate::new(Arc::new(AlwaysAllowGate), Arc::new(Approvals::unsaved()));
    let tools = adapt_client_as_tools(
        Arc::new(ImageClient),
        &"shots".parse().unwrap(),
        &approvals,
        &vision,
    )
    .await
    .unwrap();
    let without = tools[0].invoke(json!({})).await.unwrap();
    assert!(without.attachments.is_empty());
    vision.set(true);
    let with = tools[0].invoke(json!({})).await.unwrap();
    assert_eq!(
        image_bytes(&with.attachments),
        [(ImageMime::Png, [0, 0, 0].as_slice())]
    );
}

#[tokio::test]
async fn adapter_turns_client_errors_into_error_envelopes() {
    let err = || McpError::RpcError {
        code: -32602,
        message: "missing field 'query'".into(),
    };
    let tool = failing_tool(err()).await;
    let out = tool.invoke(json!({})).await.unwrap();
    assert_eq!(out.output, mcp_error_line("mcp__web__search", &err()));
    assert_eq!(out.exit_code, -1);
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
        &"files".parse().unwrap(),
        &ApprovalGate::new(gate, approvals),
        &VisionGate::new(true),
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
        out.output,
        "[error] mcp__files__delete: call cancelled by user. Try: a different approach\n"
    );
    assert_eq!(out.exit_code, -1);
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

/// Advertises a fixed list of tools.
#[derive(Debug)]
struct ListingClient(Vec<rmcp::model::Tool>);

#[async_trait]
impl McpClient for ListingClient {
    async fn list_tools(&self) -> Result<Vec<rmcp::model::Tool>, McpError> {
        Ok(self.0.clone())
    }

    async fn invoke(&self, _name: &str, _arguments: Value) -> Result<CallToolResult, McpError> {
        Ok(text_result(""))
    }
}

#[tokio::test]
async fn only_well_formed_unique_tools_are_registered() {
    let mut huge_schema = mcp_tool("huge_schema", "ok");
    huge_schema.input_schema = Arc::new(
        serde_json::from_value(json!({"type": "object", "description": "x".repeat(20_000)}))
            .unwrap(),
    );
    let client = Arc::new(ListingClient(vec![
        mcp_tool("search", "first"),
        mcp_tool("bad name\"", "quoted"),
        mcp_tool("", "empty"),
        mcp_tool(&"a".repeat(60), "too long once prefixed"),
        mcp_tool("verbose", &"x".repeat(5_000)),
        huge_schema,
        mcp_tool("search", "second"),
        mcp_tool("fetch-page", "kept"),
    ]));
    let tools = adapt_allowing(client).await.unwrap();
    let descriptions: Vec<_> = tools.iter().map(|t| t.description()).collect();
    let kept: Vec<(&str, &str)> = tools
        .iter()
        .zip(&descriptions)
        .map(|(t, description)| (t.name(), description.as_ref()))
        .collect();
    assert_eq!(
        kept,
        [
            ("mcp__web__search", "first"),
            ("mcp__web__fetch-page", "kept")
        ]
    );
}

#[tokio::test]
async fn tools_past_the_server_catalog_cap_are_skipped() {
    let client = Arc::new(ListingClient(
        (0..200)
            .map(|index| mcp_tool(&format!("tool{index}"), "short"))
            .collect(),
    ));
    let tools = adapt_allowing(client).await.unwrap();
    assert_eq!(tools.len(), 128);
    assert_eq!(tools.last().unwrap().name(), "mcp__web__tool127");

    let client = Arc::new(ListingClient(
        (0..100)
            .map(|index| mcp_tool(&format!("tool{index}"), &"x".repeat(4_000)))
            .collect(),
    ));
    let tools = adapt_allowing(client).await.unwrap();
    assert!(
        tools.len() < 40,
        "admitted {} oversized-catalog tools",
        tools.len()
    );
}
