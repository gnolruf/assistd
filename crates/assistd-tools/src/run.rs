//! The single LLM-facing tool: `run`. Parses a command-line string,
//! routes each stage through a [`CommandRegistry`] (Layer 1), then hands the
//! final [`CommandOutput`] to the [`crate::presentation`] module (Layer 2) for
//! binary guarding, overflow spill-to-file, stderr attachment, and the
//! metadata footer.
//!
//! The split between the two layers matters: pipes operate on raw bytes
//! (Layer 1), so `cat bigfile | grep foo | wc -l` feeds grep the full cat
//! output without any LLM-presentation truncation. Only the final chain
//! result is condensed into a model-friendly body.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Instant;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use serde_json::{Value, json};

use crate::Tool;
use crate::chain::{execute, parse_chain};
use crate::command::{Attachment, CommandRegistry};
use crate::presentation::{PresentSpec, present};

pub struct RunTool {
    registry: Arc<CommandRegistry>,
    spec: PresentSpec,
    counter: Arc<AtomicU64>,
    /// Level-0 description, built once in `new` from
    /// `registry.sorted_summaries()`. Returned by reference from
    /// [`Tool::description`] — owned here so the trait's `&str` signature
    /// is honored without requiring a static string.
    description: String,
}

impl RunTool {
    pub fn new(registry: Arc<CommandRegistry>, spec: PresentSpec) -> Self {
        let description = build_description(&registry);
        Self {
            registry,
            spec,
            counter: Arc::new(AtomicU64::new(0)),
            description,
        }
    }
}

/// Build the Level-0 description the LLM sees in its tool schema. The
/// header/footer are static; the middle block is a bulleted list of
/// `(name, summary)` pairs pulled from every registered command. The
/// "run a command with no arguments" footer is the discovery hook: the
/// LLM learns that calling a command bare returns Level-1 usage.
fn build_description(registry: &CommandRegistry) -> String {
    let mut s = String::with_capacity(1024);
    s.push_str(
        "Execute a shell-style command in the daemon's working directory. \
         Supports pipelines (|), and/or (&&, ||), and sequencing (;). \
         Redirections (>, <), env expansion ($VAR), and backgrounding (&) \
         are NOT supported — use `bash \"…\"` for a real shell when needed. \
         Large outputs are truncated; the truncation notice includes a \
         `Full output: /tmp/assistd-output/cmd-N.txt` path that subsequent \
         `run` calls can grep/cat to read the full content.\n\n\
         Available commands:\n",
    );
    let pairs = registry.sorted_summaries();
    let name_width = pairs.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
    for (name, summary) in pairs {
        s.push_str("  ");
        s.push_str(name);
        for _ in name.len()..name_width {
            s.push(' ');
        }
        s.push_str(" — ");
        s.push_str(summary);
        s.push('\n');
    }
    s.push_str(
        "\nCall a command with no (or insufficient) arguments to see its \
         usage (exit code 2, stdout). Errors in real calls exit with a \
         `[<name>]\\t` stderr prefix — distinct from help on stdout.",
    );
    s
}

#[async_trait]
impl Tool for RunTool {
    fn name(&self) -> &str {
        "run"
    }

    fn description(&self) -> &str {
        &self.description
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
        let start = Instant::now();
        let out = execute(&chain, &self.registry, Vec::new()).await?;
        let duration = start.elapsed();

        let r = present(out, &self.spec, &self.counter, duration);

        let mut result = json!({
            "output":      r.output,
            "stdout":      r.stdout_raw,
            "stderr":      r.stderr_raw,
            "exit_code":   r.exit_code,
            "truncated":   r.truncated,
            "duration_ms": r.duration_ms,
        });
        if let Some(p) = &r.overflow_file {
            result["overflow_file"] = json!(p.to_string_lossy());
        }
        if !r.attachments.is_empty() {
            let rendered: Vec<Value> = r.attachments.iter().map(render_attachment).collect();
            result["attachments"] = Value::Array(rendered);
        }
        Ok(result)
    }
}

fn render_attachment(a: &Attachment) -> Value {
    match a {
        Attachment::Image { mime, bytes } => json!({
            "type": "image",
            "mime": mime,
            "data": B64.encode(bytes),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::{Command, CommandInput, CommandOutput};
    use crate::commands::{
        BashCommand, CatCommand, EchoCommand, GrepCommand, LsCommand, SeeCommand, WcCommand,
        WebCommand, WriteCommand,
    };
    use crate::ToolRegistry;
    use async_trait::async_trait;
    use regex::Regex;
    use std::path::Path;
    use tempfile::{TempDir, tempdir};

    fn registry() -> Arc<CommandRegistry> {
        let mut r = CommandRegistry::new();
        r.register(CatCommand);
        r.register(EchoCommand);
        r.register(GrepCommand);
        r.register(LsCommand);
        r.register(WcCommand);
        r.register(SeeCommand);
        Arc::new(r)
    }

    /// Full 9-command registry matching the daemon's production set.
    /// Used by Level-0 description tests that need to verify every
    /// command name flows into the LLM-facing schema.
    fn full_registry() -> Arc<CommandRegistry> {
        let mut r = CommandRegistry::new();
        r.register(CatCommand);
        r.register(LsCommand);
        r.register(GrepCommand);
        r.register(WcCommand);
        r.register(EchoCommand);
        r.register(WriteCommand);
        r.register(SeeCommand);
        r.register(WebCommand::new());
        r.register(BashCommand::default());
        Arc::new(r)
    }

    fn tool_with_dir(dir: &Path) -> RunTool {
        RunTool::new(
            registry(),
            PresentSpec {
                max_lines: 200,
                max_bytes: 50 * 1024,
                overflow_dir: dir.to_path_buf(),
            },
        )
    }

    fn tool_with(dir: &Path, reg: Arc<CommandRegistry>) -> RunTool {
        RunTool::new(
            reg,
            PresentSpec {
                max_lines: 200,
                max_bytes: 50 * 1024,
                overflow_dir: dir.to_path_buf(),
            },
        )
    }

    fn fresh_dir() -> TempDir {
        tempdir().expect("tempdir")
    }

    fn invoke(tool: &RunTool, cmd: &str) -> Value {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(tool.invoke(json!({ "command": cmd }))).unwrap()
    }

    fn footer_re() -> Regex {
        Regex::new(r"\[exit:-?\d+ \| \d+ms\]$").unwrap()
    }

    fn assert_footer(output: &str, expected_exit: i32) {
        assert!(
            footer_re().is_match(output),
            "expected footer at end of: {output:?}"
        );
        let prefix = format!("[exit:{expected_exit} | ");
        assert!(
            output.contains(&prefix),
            "expected footer exit_code={expected_exit} in: {output:?}"
        );
    }

    const PNG_BYTES: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F,
        0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00,
        0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49,
        0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];

    // --- acceptance criteria ----------------------------------------------

    #[test]
    fn run_cat_returns_file_contents() {
        let dir = fresh_dir();
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("notes.md");
        std::fs::write(&path, b"hello notes\n").unwrap();
        let tool = tool_with_dir(dir.path());
        let cmd = format!("cat {}", path.to_string_lossy());
        let result = invoke(&tool, &cmd);
        assert_eq!(result["exit_code"], 0);
        assert_eq!(result["stdout"], "hello notes\n");
        let output = result["output"].as_str().unwrap();
        assert!(output.starts_with("hello notes\n"));
        assert_footer(output, 0);
    }

    #[test]
    fn run_cat_rejects_binary_image() {
        // `cat` rejects the binary file at Layer 1 — exit 1 with stderr.
        // Layer 2 surfaces stderr inline via [stderr] marker.
        let dir = fresh_dir();
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("photo.png");
        std::fs::write(&path, PNG_BYTES).unwrap();
        let tool = tool_with_dir(dir.path());
        let cmd = format!("cat {}", path.to_string_lossy());
        let result = invoke(&tool, &cmd);
        assert_eq!(result["exit_code"], 1);
        let stderr = result["stderr"].as_str().unwrap();
        assert!(stderr.contains("binary image/png"), "{stderr}");
        let output = result["output"].as_str().unwrap();
        assert!(output.contains("[stderr] "), "output={output}");
        assert!(output.contains("binary image/png"), "output={output}");
        assert_footer(output, 1);
    }

    #[test]
    fn run_see_returns_attachment_as_base64() {
        let dir = fresh_dir();
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("shot.png");
        std::fs::write(&path, PNG_BYTES).unwrap();
        let tool = tool_with_dir(dir.path());
        let cmd = format!("see {}", path.to_string_lossy());
        let result = invoke(&tool, &cmd);
        assert_eq!(result["exit_code"], 0);
        let attachments = result["attachments"].as_array().expect("attachments array");
        assert_eq!(attachments.len(), 1);
        assert_eq!(attachments[0]["type"], "image");
        assert_eq!(attachments[0]["mime"], "image/png");
        let decoded = B64
            .decode(attachments[0]["data"].as_str().unwrap())
            .unwrap();
        assert_eq!(decoded.as_slice(), PNG_BYTES);
    }

    #[test]
    fn run_attachments_flow_through_pipeline() {
        let dir = fresh_dir();
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("shot.png");
        std::fs::write(&path, PNG_BYTES).unwrap();
        let tool = tool_with_dir(dir.path());
        let cmd = format!("see {} | wc -l", path.to_string_lossy());
        let result = invoke(&tool, &cmd);
        assert_eq!(result["exit_code"], 0);
        let attachments = result["attachments"].as_array().expect("attachments");
        assert_eq!(attachments.len(), 1);
        assert_eq!(attachments[0]["mime"], "image/png");
    }

    #[test]
    fn run_grep_ic_returns_count() {
        let dir = fresh_dir();
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("log.txt");
        std::fs::write(&path, b"ERROR a\ninfo\nError b\nwarn\n").unwrap();
        let tool = tool_with_dir(dir.path());
        let cmd = format!("cat {} | grep -ic \"error\"", path.to_string_lossy());
        let result = invoke(&tool, &cmd);
        assert_eq!(result["exit_code"], 0);
        assert_eq!(result["stdout"], "2\n");
    }

    #[test]
    fn run_pipeline_cat_grep_wc() {
        let dir = fresh_dir();
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("log.txt");
        std::fs::write(&path, b"INFO start\nERROR a\nWARN b\nERROR c\nINFO done\n").unwrap();
        let tool = tool_with_dir(dir.path());
        let cmd = format!("cat {} | grep ERROR | wc -l", path.to_string_lossy());
        let result = invoke(&tool, &cmd);
        assert_eq!(result["exit_code"], 0);
        assert_eq!(result["stdout"], "2\n");
    }

    #[test]
    fn run_or_fallback_on_missing_file() {
        let dir = fresh_dir();
        let tool = tool_with_dir(dir.path());
        let result = invoke(&tool, "cat /no/such/path || echo 'not found'");
        assert_eq!(result["exit_code"], 0);
        let stdout = result["stdout"].as_str().unwrap();
        assert!(stdout.contains("not found"), "{stdout}");
    }

    #[test]
    fn run_and_chains_only_on_success() {
        let dir = fresh_dir();
        let tmp = tempdir().unwrap();
        let tool = tool_with_dir(dir.path());
        let cmd = format!("ls {} && echo done", tmp.path().to_string_lossy());
        let result = invoke(&tool, &cmd);
        assert_eq!(result["exit_code"], 0);
        let stdout = result["stdout"].as_str().unwrap();
        assert!(stdout.contains("done"), "{stdout}");
    }

    #[test]
    fn run_seq_runs_both() {
        let dir = fresh_dir();
        let tool = tool_with_dir(dir.path());
        let result = invoke(&tool, "echo hello ; echo world");
        assert_eq!(result["exit_code"], 0);
        let stdout = result["stdout"].as_str().unwrap();
        assert!(stdout.contains("hello"), "{stdout}");
        assert!(stdout.contains("world"), "{stdout}");
    }

    #[test]
    fn run_unknown_command_lists_available() {
        let dir = fresh_dir();
        let tool = tool_with_dir(dir.path());
        let result = invoke(&tool, "foo");
        assert_eq!(result["exit_code"], 127);
        let stderr = result["stderr"].as_str().unwrap();
        assert!(stderr.contains("[error] unknown command: foo"), "{stderr}");
        assert!(stderr.contains("cat"), "{stderr}");
        assert!(stderr.contains("echo"), "{stderr}");
        assert!(stderr.contains("grep"), "{stderr}");
        assert!(stderr.contains("ls"), "{stderr}");
        assert!(stderr.contains("see"), "{stderr}");
        assert!(stderr.contains("wc"), "{stderr}");
    }

    // --- boundary / sanity ------------------------------------------------

    #[test]
    fn run_rejects_wrong_argument_key() {
        let dir = fresh_dir();
        let tool = tool_with_dir(dir.path());
        let rt = tokio::runtime::Runtime::new().unwrap();
        let err = rt
            .block_on(tool.invoke(json!({ "cmd": "ls" })))
            .unwrap_err();
        assert!(err.to_string().contains("command"), "{err}");
    }

    #[test]
    fn run_parse_error_surfaces() {
        let dir = fresh_dir();
        let tool = tool_with_dir(dir.path());
        let rt = tokio::runtime::Runtime::new().unwrap();
        let err = rt
            .block_on(tool.invoke(json!({ "command": "echo hi |" })))
            .unwrap_err();
        assert!(err.to_string().contains("parse error"), "{err}");
    }

    #[test]
    fn run_omits_attachments_key_when_empty() {
        let dir = fresh_dir();
        let tool = tool_with_dir(dir.path());
        let result = invoke(&tool, "echo hi");
        assert!(
            result.get("attachments").is_none(),
            "attachments key should be absent when empty"
        );
    }

    // --- new: Layer 2 acceptance criteria --------------------------------

    /// Fake command: emits a configurable number of lines.
    struct Lines(usize);
    #[async_trait]
    impl Command for Lines {
        fn name(&self) -> &str {
            "lines"
        }
        fn summary(&self) -> &'static str {
            "test: emit N lines"
        }
        fn help(&self) -> String {
            "usage: lines".to_string()
        }
        async fn run(&self, _input: CommandInput) -> Result<CommandOutput> {
            let mut v = Vec::with_capacity(self.0 * 8);
            for i in 1..=self.0 {
                v.extend_from_slice(format!("line {i}\n").as_bytes());
            }
            Ok(CommandOutput::ok(v))
        }
    }

    /// Fake command: counts how many bytes flow in on stdin.
    struct ByteCount;
    #[async_trait]
    impl Command for ByteCount {
        fn name(&self) -> &str {
            "bytecount"
        }
        fn summary(&self) -> &'static str {
            "test: count bytes"
        }
        fn help(&self) -> String {
            "usage: bytecount".to_string()
        }
        async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
            Ok(CommandOutput::ok(
                format!("{}\n", input.stdin.len()).into_bytes(),
            ))
        }
    }

    /// Fake command: emits raw PNG bytes as stdout.
    struct EmitPng;
    #[async_trait]
    impl Command for EmitPng {
        fn name(&self) -> &str {
            "emitpng"
        }
        fn summary(&self) -> &'static str {
            "test: emit PNG"
        }
        fn help(&self) -> String {
            "usage: emitpng".to_string()
        }
        async fn run(&self, _input: CommandInput) -> Result<CommandOutput> {
            Ok(CommandOutput::ok(PNG_BYTES.to_vec()))
        }
    }

    /// Fake command: produces stdout + stderr + non-zero exit in one shot.
    struct Noisy;
    #[async_trait]
    impl Command for Noisy {
        fn name(&self) -> &str {
            "noisy"
        }
        fn summary(&self) -> &'static str {
            "test: noisy output"
        }
        fn help(&self) -> String {
            "usage: noisy".to_string()
        }
        async fn run(&self, _input: CommandInput) -> Result<CommandOutput> {
            Ok(CommandOutput {
                stdout: b"stdout content\n".to_vec(),
                stderr: b"stderr content\n".to_vec(),
                exit_code: 1,
                attachments: Vec::new(),
            })
        }
    }

    #[test]
    fn run_pipe_integrity_full_bytes_reach_final_stage() {
        // Layer 1 must NOT truncate the 5000-line stream between `lines` and
        // `bytecount`. The final `bytecount` sees the full upstream bytes.
        let mut reg = CommandRegistry::new();
        reg.register(Lines(5000));
        reg.register(ByteCount);
        let dir = fresh_dir();
        let tool = tool_with(dir.path(), Arc::new(reg));
        let result = invoke(&tool, "lines | bytecount");
        let expected_bytes: usize = (1..=5000).map(|i| format!("line {i}\n").len()).sum();
        assert_eq!(result["exit_code"], 0);
        assert_eq!(
            result["stdout"].as_str().unwrap(),
            &format!("{expected_bytes}\n")
        );
        assert_eq!(result["truncated"], false);
        assert!(result.get("overflow_file").is_none());
    }

    #[test]
    fn run_overflow_end_to_end_writes_temp_file() {
        let mut reg = CommandRegistry::new();
        reg.register(Lines(5000));
        let dir = fresh_dir();
        let tool = tool_with(dir.path(), Arc::new(reg));
        let result = invoke(&tool, "lines");
        assert_eq!(result["truncated"], true);
        let path = result["overflow_file"].as_str().expect("overflow_file");
        assert_eq!(path, dir.path().join("cmd-1.txt").to_string_lossy());

        let contents = std::fs::read_to_string(path).unwrap();
        // The full 5000-line content is on disk.
        assert_eq!(contents.lines().count(), 5000);
        assert!(contents.starts_with("line 1\n"));
        assert!(contents.ends_with("line 5000\n"));

        let output = result["output"].as_str().unwrap();
        assert!(output.contains("line 1\n"));
        assert!(output.contains("line 200\n"));
        assert!(!output.contains("line 201\n"));
        assert!(output.contains("--- output truncated (5000 lines,"));
        assert!(output.contains(&format!("Full output: {path}")));
        assert!(output.contains(&format!("Explore: cat {path} | grep")));
        assert!(output.contains(&format!("cat {path} | tail 100")));
        assert_footer(output, 0);
    }

    #[test]
    fn run_overflow_file_readable_via_followup_grep() {
        // Acceptance: after an overflow, the LLM can follow up with
        // `cat <overflow-path> | grep <pat>` and get matches from the full
        // output.
        let mut reg = CommandRegistry::new();
        reg.register(Lines(5000));
        reg.register(CatCommand);
        reg.register(GrepCommand);
        let dir = fresh_dir();
        let tool = tool_with(dir.path(), Arc::new(reg));

        let first = invoke(&tool, "lines");
        let path = first["overflow_file"].as_str().expect("path").to_string();

        // Follow-up: same RunTool instance, same overflow dir, reads the
        // spilled file.
        let followup = invoke(&tool, &format!("cat {path} | grep \"line 4242\""));
        assert_eq!(followup["exit_code"], 0);
        assert_eq!(followup["stdout"].as_str().unwrap(), "line 4242\n");
    }

    #[test]
    fn run_binary_guard_end_to_end() {
        let mut reg = CommandRegistry::new();
        reg.register(EmitPng);
        let dir = fresh_dir();
        let tool = tool_with(dir.path(), Arc::new(reg));
        let result = invoke(&tool, "emitpng");
        assert_eq!(result["exit_code"], 0);
        let output = result["output"].as_str().unwrap();
        assert!(output.starts_with("[error] binary output (image/png, "));
        assert!(output.contains(". Use: cat -b <path>"));
        assert_footer(output, 0);
        // stdout_raw is suppressed when binary guard trips.
        assert_eq!(result["stdout"].as_str().unwrap(), "");
    }

    #[test]
    fn run_metadata_footer_on_unknown_command() {
        let dir = fresh_dir();
        let tool = tool_with_dir(dir.path());
        let result = invoke(&tool, "nope");
        assert_eq!(result["exit_code"], 127);
        let output = result["output"].as_str().unwrap();
        assert_footer(output, 127);
        assert!(output.contains("[stderr] "));
        assert!(output.contains("unknown command: nope"));
    }

    #[test]
    fn run_stderr_attached_when_both_stdout_and_stderr_present() {
        let mut reg = CommandRegistry::new();
        reg.register(Noisy);
        let dir = fresh_dir();
        let tool = tool_with(dir.path(), Arc::new(reg));
        let result = invoke(&tool, "noisy");
        assert_eq!(result["exit_code"], 1);
        let output = result["output"].as_str().unwrap();
        // Stdout content survives; stderr is not silently dropped; both appear.
        assert!(output.contains("stdout content\n"));
        assert!(output.contains("[stderr] [noisy]\tstderr content\n"));
        assert_footer(output, 1);
    }

    #[test]
    fn run_metadata_footer_present_on_success() {
        let dir = fresh_dir();
        let tool = tool_with_dir(dir.path());
        let result = invoke(&tool, "echo hi");
        let output = result["output"].as_str().unwrap();
        assert_footer(output, 0);
        assert!(output.starts_with("hi\n"));
    }

    #[test]
    fn run_respects_config_overrides() {
        // Tight spec: 3 lines / 10 KB. `lines 10` overflows via line cap.
        let mut reg = CommandRegistry::new();
        reg.register(Lines(10));
        let dir = fresh_dir();
        let tool = RunTool::new(
            Arc::new(reg),
            PresentSpec {
                max_lines: 3,
                max_bytes: 10 * 1024,
                overflow_dir: dir.path().to_path_buf(),
            },
        );
        let result = invoke(&tool, "lines");
        assert_eq!(result["truncated"], true);
        let output = result["output"].as_str().unwrap();
        assert!(output.contains("line 1\n"));
        assert!(output.contains("line 3\n"));
        assert!(!output.contains("line 4\n"));
        assert!(output.contains("--- output truncated (10 lines,"));
    }

    // --- progressive help: Level 0 (tool description) --------------------

    /// Acceptance #1: the `run` tool's description lists every registered
    /// command with its one-line summary.
    #[test]
    fn run_tool_description_lists_all_commands() {
        let dir = fresh_dir();
        let tool = tool_with(dir.path(), full_registry());
        let desc = tool.description();
        for name in ["cat", "ls", "grep", "wc", "echo", "write", "see", "web", "bash"] {
            assert!(desc.contains(name), "description missing `{name}`: {desc}");
        }
        // Summary text from a representative command should appear.
        assert!(
            desc.contains("filter lines matching a pattern"),
            "description missing grep summary: {desc}"
        );
        // Level-1 discovery hint tells the LLM how to drill in.
        assert!(
            desc.to_lowercase().contains("no (or insufficient) arguments")
                || desc.to_lowercase().contains("usage"),
            "description missing drill-in hint: {desc}"
        );
    }

    /// Acceptance #4: adding a new command to the registry automatically
    /// includes it in the Level-0 summary without touching any tool-side
    /// description string.
    #[test]
    fn run_tool_description_auto_updates_when_command_added() {
        struct Frobnicate;
        #[async_trait]
        impl Command for Frobnicate {
            fn name(&self) -> &str {
                "frobnicate"
            }
            fn summary(&self) -> &'static str {
                "frobnicate the widget"
            }
            fn help(&self) -> String {
                "usage: frobnicate".to_string()
            }
            async fn run(&self, _input: CommandInput) -> Result<CommandOutput> {
                Ok(CommandOutput::ok(Vec::new()))
            }
        }
        let mut reg = CommandRegistry::new();
        reg.register(CatCommand);
        reg.register(Frobnicate);
        let dir = fresh_dir();
        let tool = tool_with(dir.path(), Arc::new(reg));
        let desc = tool.description();
        assert!(desc.contains("frobnicate"), "missing name: {desc}");
        assert!(
            desc.contains("frobnicate the widget"),
            "missing summary: {desc}"
        );
    }

    /// Acceptance #1, wire-level: the OpenAI-compatible schema (what the
    /// LLM actually consumes) exposes the dynamic description verbatim.
    #[test]
    fn openai_schemas_description_includes_all_commands() {
        let dir = fresh_dir();
        let mut tools = ToolRegistry::new();
        tools.register(RunTool::new(
            full_registry(),
            PresentSpec {
                max_lines: 200,
                max_bytes: 50 * 1024,
                overflow_dir: dir.path().to_path_buf(),
            },
        ));
        let schemas = tools.openai_schemas();
        assert_eq!(schemas.len(), 1);
        let desc = schemas[0]["function"]["description"]
            .as_str()
            .expect("description is a string");
        for name in ["cat", "ls", "grep", "wc", "echo", "write", "see", "web", "bash"] {
            assert!(desc.contains(name), "schema description missing `{name}`");
        }
    }

    // --- progressive help: Level 1 (command-level help on missing args) --

    /// Acceptance #2, #5: calling a command with no arguments returns its
    /// help text on stdout with a non-zero exit code so the LLM can tell
    /// help from successful execution.
    #[test]
    fn run_grep_no_args_returns_help_on_stdout_exit_2() {
        let dir = fresh_dir();
        let tool = tool_with_dir(dir.path());
        let result = invoke(&tool, "grep");
        assert_eq!(result["exit_code"], 2);
        let stdout = result["stdout"].as_str().unwrap();
        assert!(
            stdout.starts_with("usage: grep"),
            "stdout should start with `usage: grep`: {stdout:?}"
        );
        assert!(stdout.contains("PATTERN"), "help missing PATTERN: {stdout}");
        // Help goes to stdout; stderr stays empty (no `[grep]\t` prefix).
        assert_eq!(result["stderr"].as_str().unwrap(), "");
        let output = result["output"].as_str().unwrap();
        assert_footer(output, 2);
    }

    #[test]
    fn run_see_no_args_returns_help_on_stdout_exit_2() {
        let dir = fresh_dir();
        let mut reg = CommandRegistry::new();
        reg.register(SeeCommand);
        let tool = tool_with(dir.path(), Arc::new(reg));
        let result = invoke(&tool, "see");
        assert_eq!(result["exit_code"], 2);
        let stdout = result["stdout"].as_str().unwrap();
        assert!(stdout.starts_with("usage: see"), "{stdout:?}");
        assert_eq!(result["stderr"].as_str().unwrap(), "");
    }

    #[test]
    fn run_write_no_args_returns_help_on_stdout_exit_2() {
        let dir = fresh_dir();
        let mut reg = CommandRegistry::new();
        reg.register(WriteCommand);
        let tool = tool_with(dir.path(), Arc::new(reg));
        let result = invoke(&tool, "write");
        assert_eq!(result["exit_code"], 2);
        let stdout = result["stdout"].as_str().unwrap();
        assert!(stdout.starts_with("usage: write"), "{stdout:?}");
        assert_eq!(result["stderr"].as_str().unwrap(), "");
    }

    #[test]
    fn run_web_no_args_returns_help_on_stdout_exit_2() {
        let dir = fresh_dir();
        let mut reg = CommandRegistry::new();
        reg.register(WebCommand::new());
        let tool = tool_with(dir.path(), Arc::new(reg));
        let result = invoke(&tool, "web");
        assert_eq!(result["exit_code"], 2);
        let stdout = result["stdout"].as_str().unwrap();
        assert!(stdout.starts_with("usage: web"), "{stdout:?}");
        assert_eq!(result["stderr"].as_str().unwrap(), "");
    }

    #[test]
    fn run_bash_no_args_returns_help_on_stdout_exit_2() {
        let dir = fresh_dir();
        let mut reg = CommandRegistry::new();
        reg.register(BashCommand::default());
        let tool = tool_with(dir.path(), Arc::new(reg));
        let result = invoke(&tool, "bash");
        assert_eq!(result["exit_code"], 2);
        let stdout = result["stdout"].as_str().unwrap();
        assert!(stdout.starts_with("usage: bash"), "{stdout:?}");
        assert_eq!(result["stderr"].as_str().unwrap(), "");
    }

    /// Help output on stdout is visually distinct from a real usage error
    /// on stderr. A real grep error (bad regex) still goes to stderr with
    /// the `[grep]\t` executor prefix — exits 2 like help does, but the
    /// transport is different. This test locks the distinction.
    #[test]
    fn run_grep_real_usage_error_still_on_stderr() {
        let dir = fresh_dir();
        let tool = tool_with_dir(dir.path());
        // Unrecognized flag — triggers parse_flags error path, which
        // remains on stderr (unlike the no-args path, which emits help).
        let result = invoke(&tool, "grep -x foo");
        assert_eq!(result["exit_code"], 2);
        let stderr = result["stderr"].as_str().unwrap();
        assert!(
            stderr.contains("[grep]\t"),
            "real errors keep executor prefix: {stderr}"
        );
        // Stdout stays empty — the usage error didn't go to stdout.
        assert_eq!(result["stdout"].as_str().unwrap(), "");
    }

    // --- OpenAI schema ----------------------------------------------------

    #[test]
    fn parameters_schema_shape() {
        let dir = fresh_dir();
        let tool = tool_with_dir(dir.path());
        let schema = tool.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["required"][0], "command");
        assert_eq!(schema["properties"]["command"]["type"], "string");
    }
}
