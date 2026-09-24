//! Minimal MCP server fixture: newline-delimited JSON-RPC on
//! stdin/stdout, answering `initialize`, `ping`, `tools/list` and
//! `tools/call` for the `echo`, `crash_me` and `flood_stdout` tools.

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
                let _ = writeln!(std::io::stderr(), "fake-mcp: read error: {e}");
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
                let _ = writeln!(std::io::stderr(), "fake-mcp: parse error: {e}");
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
                        },
                        {
                            "name": "flood_stdout",
                            "description": "writes one oversized line and stays alive",
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
                    // Exits without replying, simulating a crash mid-session.
                    "crash_me" => {
                        std::process::exit(0);
                    }
                    // Kills the client's read loop while this process stays
                    // alive. Overshoots the 1 MiB line cap by less than a
                    // pipe buffer, so the write completes and stdin stays
                    // readable after the client gives up.
                    "flood_stdout" => {
                        let mut junk = vec![b'x'; 1024 * 1024 + 1024];
                        junk.push(b'\n');
                        let _ = out.write_all(&junk);
                        let _ = out.flush();
                        continue;
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
