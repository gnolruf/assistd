//! Minimal MCP server fixture: newline-delimited JSON-RPC on
//! stdin/stdout, answering `initialize`, `ping`, `tools/list` and
//! `tools/call` for the `echo`, `crash_me`, `never_answers`,
//! `close_stdout`, `spawn_orphan_and_crash` and `env_names` tools.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::process::CommandExt;
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
        let Some(response) = respond(&req, &id, &mut out) else {
            continue;
        };
        if write_line(&mut out, &response).is_err() {
            break;
        }
    }
}

/// The reply to request `req`, or `None` when the tool answers by
/// writing to `out` directly.
fn respond(req: &Value, id: &Value, out: &mut impl Write) -> Option<Value> {
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

fn initialize(id: &Value) -> Value {
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

fn tools_list(id: &Value) -> Value {
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
                    "name": "never_answers",
                    "description": "accepts the call and never replies",
                    "inputSchema": {"type": "object", "properties": {}}
                },
                {
                    "name": "close_stdout",
                    "description": "closes stdout without answering and stays alive",
                    "inputSchema": {"type": "object", "properties": {}}
                },
                {
                    "name": "spawn_orphan_and_crash",
                    "description": "spawns a grandchild, answers with its pid, then exits",
                    "inputSchema": {"type": "object", "properties": {}}
                },
                {
                    "name": "env_names",
                    "description": "answers with the names of its environment variables",
                    "inputSchema": {"type": "object", "properties": {}}
                }
            ]
        }
    })
}

fn call_tool(req: &Value, id: &Value, out: &mut impl Write) -> Option<Value> {
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
        "never_answers" => None,
        "close_stdout" => close_stdout(),
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
        "env_names" => Some(env_names(id)),
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

fn env_names(id: &Value) -> Value {
    let names: Vec<String> = std::env::vars_os()
        .map(|(name, _)| name.to_string_lossy().into_owned())
        .collect();
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "content": [{"type": "text", "text": names.join("\n")}],
            "isError": false
        }
    })
}

fn close_stdout() -> ! {
    let err = std::process::Command::new("sleep")
        .arg("300")
        .stdout(Stdio::null())
        .exec();
    panic!("exec sleep: {err}");
}

fn write_line(out: &mut impl Write, response: &Value) -> std::io::Result<()> {
    let mut bytes = serde_json::to_vec(response).unwrap();
    bytes.push(b'\n');
    out.write_all(&bytes)?;
    out.flush()
}
