//! The single LLM-facing tool: `run`. Parses a command-line string,
//! routes each stage through a [`CommandRegistry`], and returns a
//! structured JSON result with stdout (capped), stderr, and exit code.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde_json::{Value, json};

use crate::Tool;
use crate::chain::{execute, parse_chain};
use crate::command::CommandRegistry;

/// Cap on the stdout we surface to the LLM. A runaway command (e.g. a
/// `bash "cat /dev/urandom"`) would otherwise bury the model's context
/// window. When hit, the returned JSON gains `truncated: true` and the
/// stdout string ends with `"…<truncated>"`.
pub const STDOUT_MAX: usize = 64 * 1024;

pub struct RunTool {
    registry: Arc<CommandRegistry>,
}

impl RunTool {
    pub fn new(registry: Arc<CommandRegistry>) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for RunTool {
    fn name(&self) -> &str {
        "run"
    }

    fn description(&self) -> &str {
        "Execute a shell-style command in the daemon's working directory. \
         Supports pipelines (|), and/or (&&, ||), and sequencing (;). \
         Built-in commands: cat, ls, grep, wc, echo, write, bash. \
         Redirections (>, <), env expansion ($VAR), and backgrounding (&) \
         are NOT supported — use `bash \"…\"` for a real shell when needed."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command line to execute, \
                                    e.g. \"cat log.txt | grep ERROR | wc -l\"."
                }
            },
            "required": ["command"]
        })
    }

    async fn invoke(&self, args: Value) -> Result<Value> {
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("`command` (string) is required"))?;

        let chain = parse_chain(command).map_err(|e| anyhow!("parse error: {e}"))?;
        let out = execute(&chain, &self.registry, Vec::new()).await?;

        let (stdout, truncated) = cap_utf8(&out.stdout, STDOUT_MAX);
        Ok(json!({
            "stdout": stdout,
            "stderr": String::from_utf8_lossy(&out.stderr),
            "exit_code": out.exit_code,
            "truncated": truncated,
        }))
    }
}

/// Truncate `raw` to at most `max` UTF-8 bytes, falling back to a
/// replacement-character-preserving decode. Returns `(string, truncated)`.
fn cap_utf8(raw: &[u8], max: usize) -> (String, bool) {
    if raw.len() <= max {
        return (String::from_utf8_lossy(raw).into_owned(), false);
    }
    // Find a char boundary at or before `max` so we don't split UTF-8.
    let s = String::from_utf8_lossy(raw);
    let mut cut = max.min(s.len());
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    let mut truncated = s[..cut].to_string();
    truncated.push_str("…<truncated>");
    (truncated, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::{Command, CommandInput, CommandOutput};
    use crate::commands::{CatCommand, EchoCommand, GrepCommand, LsCommand, WcCommand};
    use tempfile::tempdir;

    fn registry() -> Arc<CommandRegistry> {
        let mut r = CommandRegistry::new();
        r.register(CatCommand);
        r.register(EchoCommand);
        r.register(GrepCommand);
        r.register(LsCommand);
        r.register(WcCommand);
        Arc::new(r)
    }

    fn invoke(tool: &RunTool, cmd: &str) -> Value {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(tool.invoke(json!({ "command": cmd }))).unwrap()
    }

    // --- acceptance criteria ----------------------------------------------

    #[test]
    fn run_cat_returns_file_contents() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("notes.md");
        std::fs::write(&path, b"hello notes\n").unwrap();
        let tool = RunTool::new(registry());
        let cmd = format!("cat {}", path.to_string_lossy());
        let result = invoke(&tool, &cmd);
        assert_eq!(result["exit_code"], 0);
        assert_eq!(result["stdout"], "hello notes\n");
    }

    #[test]
    fn run_pipeline_cat_grep_wc() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("log.txt");
        std::fs::write(&path, b"INFO start\nERROR a\nWARN b\nERROR c\nINFO done\n").unwrap();
        let tool = RunTool::new(registry());
        let cmd = format!("cat {} | grep ERROR | wc -l", path.to_string_lossy());
        let result = invoke(&tool, &cmd);
        assert_eq!(result["exit_code"], 0);
        assert_eq!(result["stdout"], "2\n");
    }

    #[test]
    fn run_or_fallback_on_missing_file() {
        let tool = RunTool::new(registry());
        let result = invoke(&tool, "cat /no/such/path || echo 'not found'");
        assert_eq!(result["exit_code"], 0);
        let stdout = result["stdout"].as_str().unwrap();
        assert!(stdout.contains("not found"), "{stdout}");
    }

    #[test]
    fn run_and_chains_only_on_success() {
        let dir = tempdir().unwrap();
        let tool = RunTool::new(registry());
        let cmd = format!("ls {} && echo done", dir.path().to_string_lossy());
        let result = invoke(&tool, &cmd);
        assert_eq!(result["exit_code"], 0);
        let stdout = result["stdout"].as_str().unwrap();
        assert!(stdout.contains("done"), "{stdout}");
    }

    #[test]
    fn run_seq_runs_both() {
        let tool = RunTool::new(registry());
        let result = invoke(&tool, "echo hello ; echo world");
        assert_eq!(result["exit_code"], 0);
        let stdout = result["stdout"].as_str().unwrap();
        assert!(stdout.contains("hello"), "{stdout}");
        assert!(stdout.contains("world"), "{stdout}");
    }

    #[test]
    fn run_unknown_command_lists_available() {
        let tool = RunTool::new(registry());
        let result = invoke(&tool, "foo");
        assert_eq!(result["exit_code"], 127);
        let stderr = result["stderr"].as_str().unwrap();
        assert!(stderr.contains("[error] unknown command: foo"), "{stderr}");
        // Registry order is preserved by `sorted_names` alphabetically.
        assert!(stderr.contains("cat"), "{stderr}");
        assert!(stderr.contains("echo"), "{stderr}");
        assert!(stderr.contains("grep"), "{stderr}");
        assert!(stderr.contains("ls"), "{stderr}");
        assert!(stderr.contains("wc"), "{stderr}");
    }

    // --- boundary / sanity ------------------------------------------------

    #[test]
    fn run_rejects_wrong_argument_key() {
        let tool = RunTool::new(registry());
        let rt = tokio::runtime::Runtime::new().unwrap();
        let err = rt
            .block_on(tool.invoke(json!({ "cmd": "ls" })))
            .unwrap_err();
        assert!(err.to_string().contains("command"), "{err}");
    }

    #[test]
    fn run_parse_error_surfaces() {
        let tool = RunTool::new(registry());
        let rt = tokio::runtime::Runtime::new().unwrap();
        let err = rt
            .block_on(tool.invoke(json!({ "command": "echo hi |" })))
            .unwrap_err();
        assert!(err.to_string().contains("parse error"), "{err}");
    }

    /// Fake command that emits 200 KiB — should trigger `STDOUT_MAX`.
    struct Mass;
    #[async_trait]
    impl Command for Mass {
        fn name(&self) -> &str {
            "mass"
        }
        async fn run(&self, _input: CommandInput) -> Result<CommandOutput> {
            Ok(CommandOutput::ok(vec![b'x'; 200 * 1024]))
        }
    }

    #[test]
    fn run_caps_stdout_and_sets_truncated() {
        let mut reg = CommandRegistry::new();
        reg.register(Mass);
        let tool = RunTool::new(Arc::new(reg));
        let result = invoke(&tool, "mass");
        assert_eq!(result["truncated"], true);
        let stdout = result["stdout"].as_str().unwrap();
        assert!(stdout.ends_with("…<truncated>"), "length={}", stdout.len());
        assert!(stdout.len() <= STDOUT_MAX + "…<truncated>".len());
    }

    // --- OpenAI schema ----------------------------------------------------

    #[test]
    fn parameters_schema_shape() {
        let tool = RunTool::new(registry());
        let schema = tool.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["required"][0], "command");
        assert_eq!(schema["properties"]["command"]["type"], "string");
    }
}
