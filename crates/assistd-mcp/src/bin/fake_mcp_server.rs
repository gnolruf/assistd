//! Tiny in-tree MCP server fixture used by the integration test.
//!
//! Reads JSON-RPC requests one-per-line from stdin, replies on stdout
//! with the corresponding response (also one line of JSON).
//!
//! Implements just enough of the protocol to exercise the daemon's
//! discovery + invocation path:
//!   * `initialize` → returns capabilities + protocol version.
//!   * `notifications/initialized` → ignored (no response).
//!   * `tools/list` → returns two tools: `echo` and `crash_me`.
//!   * `tools/call` for `echo` → returns the input under `text`.
//!   * `tools/call` for `crash_me` → exits the process (used to
//!     simulate a server crash mid-session).
//!
//! Built as a regular `[[bin]]` of `assistd-mcp` so the integration
//! test can locate it under `target/<profile>/fake_mcp_server` via the
//! `CARGO_BIN_EXE_<name>` env var that cargo sets for tests.

#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unwrap_used,
    clippy::expect_used
)]

use std::io::{BufRead, BufReader, Write};

use serde_json::{Value, json};

fn main() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut reader = BufReader::new(stdin.lock());
    let mut line = String::new();

    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                eprintln!("fake-mcp: read error: {e}");
                break;
            }
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("fake-mcp: parse error: {e}");
                continue;
            }
        };
        // Notifications have no `id` and require no response.
        let Some(id) = req.get("id").cloned() else {
            continue;
        };
        let method = req.get("method").and_then(Value::as_str).unwrap_or("");
        let response = match method {
            "initialize" => json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake-mcp", "version": "0.0.0"}
                }
            }),
            "tools/list" => json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "echoes its `msg` argument",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"msg": {"type": "string"}},
                                "required": ["msg"],
                                "additionalProperties": false
                            }
                        },
                        {
                            "name": "crash_me",
                            "description": "exits the server process",
                            "inputSchema": {"type": "object", "properties": {}}
                        }
                    ]
                }
            }),
            "tools/call" => {
                let name = req
                    .get("params")
                    .and_then(|p| p.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                match name {
                    "echo" => {
                        let msg = req
                            .get("params")
                            .and_then(|p| p.get("arguments"))
                            .and_then(|a| a.get("msg"))
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "content": [{"type": "text", "text": format!("echo:{msg}")}],
                                "isError": false
                            }
                        })
                    }
                    "crash_me" => {
                        // Goodbye, cruel daemon.
                        std::process::exit(0);
                    }
                    other => json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {
                            "code": -32601,
                            "message": format!("unknown tool `{other}`")
                        }
                    }),
                }
            }
            "ping" => json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {}
            }),
            other => json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {
                    "code": -32601,
                    "message": format!("method not found: {other}")
                }
            }),
        };
        let mut bytes = serde_json::to_vec(&response).unwrap();
        bytes.push(b'\n');
        if out.write_all(&bytes).is_err() {
            break;
        }
        if out.flush().is_err() {
            break;
        }
    }
}
