//! Minimal MCP server fixture: newline-delimited JSON-RPC on
//! stdin/stdout, answering `initialize`, `ping`, `tools/list` and
//! `tools/call` for the `echo`, `crash_me`, `flood_stdout` and
//! `spawn_orphan_and_crash` tools.

use std::io::{BufRead, BufReader, Write};
use std::process::Stdio;

use serde_json::{Value, json};

/// When set, `initialize` spawns a long-lived grandchild, writes its pid
/// to the named file, and then fails the handshake.
const FAIL_INIT_WITH_ORPHAN_ENV: &str = "FAKE_MCP_FAIL_INIT_WITH_ORPHAN_PID_FILE";

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
            Ok(req) => req,
            Err(e) => {
                let _ = writeln!(std::io::stderr(), "fake-mcp: parse error: {e}");
                continue;
            }
        };
        let Some(id) = req.get("id").cloned() else {
            continue;
        };
        let Some(response) = respond(&req, id, &mut out) else {
            continue;
        };
        if write_line(&mut out, &response).is_err() {
            break;
        }
    }
}

/// The reply to request `req`, or `None` when the tool answers by
/// writing to `out` directly.
fn respond(req: &Value, id: Value, out: &mut impl Write) -> Option<Value> {
    let method = req.get("method").and_then(Value::as_str).unwrap_or("");
    let response = match method {
        "initialize" => initialize(id),
        "tools/list" => tools_list(id),
        "tools/call" => return call_tool(req, id, out),
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
    Some(response)
}

fn initialize(id: Value) -> Value {
    if let Some(pid_file) = std::env::var_os(FAIL_INIT_WITH_ORPHAN_ENV) {
        let orphan = spawn_orphan();
        std::fs::write(pid_file, orphan.to_string()).expect("write orphan pid file");
        return json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {"code": -32000, "message": "initialize refused by fixture"}
        });
    }
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake-mcp", "version": "0.0.0"}
        }
    })
}

/// A `sleep` that inherits this process group and outlives this process.
fn spawn_orphan() -> u32 {
    std::process::Command::new("sleep")
        .arg("300")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn sleep")
        .id()
}

fn tools_list(id: Value) -> Value {
    json!({
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
                },
                {
                    "name": "spawn_orphan_and_crash",
                    "description": "spawns a grandchild, answers with its pid, then exits",
                    "inputSchema": {"type": "object", "properties": {}}
                }
            ]
        }
    })
}

fn call_tool(req: &Value, id: Value, out: &mut impl Write) -> Option<Value> {
    let params = req.get("params");
    let name = params
        .and_then(|params| params.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("");
    match name {
        "echo" => {
            let msg = params
                .and_then(|params| params.get("arguments"))
                .and_then(|arguments| arguments.get("msg"))
                .and_then(Value::as_str)
                .unwrap_or("");
            Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "content": [{"type": "text", "text": format!("echo:{msg}")}],
                    "isError": false
                }
            }))
        }
        "crash_me" => std::process::exit(0),
        "flood_stdout" => {
            flood_stdout(out);
            None
        }
        "spawn_orphan_and_crash" => {
            let orphan = spawn_orphan();
            let response = json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "content": [{"type": "text", "text": orphan.to_string()}],
                    "isError": false
                }
            });
            let _ = write_line(out, &response);
            std::process::exit(0)
        }
        other => Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {
                "code": -32601,
                "message": format!("unknown tool `{other}`")
            }
        })),
    }
}

/// Overshoot the client's 1 MiB line cap by less than a pipe buffer, so
/// the write completes and this process keeps reading stdin.
fn flood_stdout(out: &mut impl Write) {
    let mut junk = vec![b'x'; 1024 * 1024 + 1024];
    junk.push(b'\n');
    let _ = out.write_all(&junk);
    let _ = out.flush();
}

fn write_line(out: &mut impl Write, response: &Value) -> std::io::Result<()> {
    let mut bytes = serde_json::to_vec(response).unwrap();
    bytes.push(b'\n');
    out.write_all(&bytes)?;
    out.flush()
}
