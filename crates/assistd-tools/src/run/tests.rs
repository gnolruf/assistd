use std::path::Path;

use regex::Regex;
use tempfile::{TempDir, tempdir};

use super::*;
use crate::command::{Command, CommandInput};
use crate::commands::{
    CatCommand, EchoCommand, GrepCommand, LsCommand, SeeCommand, TailCommand, WcCommand,
};
use crate::fixtures::PNG_BYTES;

/// Fake command: emits a configurable number of lines.
#[derive(Debug)]
struct Lines(usize);
#[async_trait]
impl Command for Lines {
    fn name(&self) -> &'static str {
        "lines"
    }
    fn summary(&self) -> &'static str {
        "test: emit N lines"
    }
    fn help(&self) -> String {
        "usage: lines".to_string()
    }
    async fn run(&self, _input: CommandInput) -> CommandOutput {
        let mut stdout = Vec::with_capacity(self.0 * 8);
        for i in 1..=self.0 {
            stdout.extend_from_slice(format!("line {i}\n").as_bytes());
        }
        CommandOutput::ok(stdout)
    }
}

/// Fake command: counts how many bytes flow in on stdin.
#[derive(Debug)]
struct ByteCount;
#[async_trait]
impl Command for ByteCount {
    fn name(&self) -> &'static str {
        "bytecount"
    }
    fn summary(&self) -> &'static str {
        "test: count bytes"
    }
    fn help(&self) -> String {
        "usage: bytecount".to_string()
    }
    async fn run(&self, input: CommandInput) -> CommandOutput {
        CommandOutput::ok(format!("{}\n", input.stdin.map_or(0, |s| s.len())).into_bytes())
    }
}

fn registry() -> Arc<CommandRegistry> {
    let mut reg = CommandRegistry::new();
    reg.register(CatCommand);
    reg.register(EchoCommand);
    reg.register(GrepCommand);
    reg.register(LsCommand);
    reg.register(WcCommand);
    reg.register(SeeCommand::default());
    Arc::new(reg)
}

fn full_registry() -> Arc<CommandRegistry> {
    Arc::new(crate::commands::test_registry())
}

fn tool_with_dir(dir: &Path) -> RunTool {
    RunTool::new(registry(), &ToolsOutputConfig::default(), dir.to_path_buf())
}

fn tool_with(dir: &Path, reg: Arc<CommandRegistry>) -> RunTool {
    RunTool::new(reg, &ToolsOutputConfig::default(), dir.to_path_buf())
}

fn fresh_dir() -> TempDir {
    tempdir().expect("tempdir")
}

fn invoke(tool: &RunTool, cmd: &str) -> Value {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(tool.invoke(json!({ "command": cmd }))).unwrap()
}

fn assert_footer(output: &str, expected_exit: i32) {
    let footer = Regex::new(&format!(r"\[exit:{expected_exit} \| \d+ms\]$")).unwrap();
    assert!(
        footer.is_match(output),
        "expected an exit:{expected_exit} footer at the end of: {output:?}"
    );
}

#[test]
fn run_see_returns_attachment_as_base64() {
    let dir = fresh_dir();
    let files = tempdir().unwrap();
    let path = files.path().join("shot.png");
    std::fs::write(&path, PNG_BYTES).unwrap();
    let tool = tool_with_dir(dir.path());
    let cmd = format!("see {}", path.to_string_lossy());
    let result = invoke(&tool, &cmd);
    assert_eq!(result["exit_code"], 0);
    assert_eq!(
        result["attachments"],
        json!([{ "type": "image", "mime": "image/png", "data": B64.encode(PNG_BYTES) }])
    );
}

/// The pipelines the model actually writes: flags on the built-in
/// commands, a glob, and the stream filters composed end to end.
#[test]
fn run_composes_flags_globs_and_stream_filters() {
    let dir = fresh_dir();
    let files = tempdir().unwrap();
    std::fs::write(files.path().join("a.log"), b"ERROR one\ninfo\n").unwrap();
    std::fs::write(files.path().join("b.log"), b"ERROR two\nERROR three\n").unwrap();
    std::fs::write(files.path().join("notes.txt"), b"ERROR ignored\n").unwrap();
    let tool = tool_with(dir.path(), full_registry());
    let root = files.path().to_string_lossy().into_owned();

    let globbed = invoke(&tool, &format!("grep ERROR {root}/*.log | wc -l"));
    assert_eq!(globbed["exit_code"], 0, "{globbed}");
    assert_eq!(globbed["stdout"], "3\n");

    let recursive = invoke(&tool, &format!("grep -rn ERROR {root} | wc -l"));
    assert_eq!(recursive["exit_code"], 0, "{recursive}");
    assert_eq!(recursive["stdout"], "4\n");

    std::fs::write(files.path().join("hits.txt"), b"b\na\nb\n").unwrap();
    let freq = invoke(
        &tool,
        &format!("cat {root}/hits.txt | sort | uniq -c | sort -nr | head -1"),
    );
    assert_eq!(freq["stdout"], "2\tb\n", "{freq}");

    let numbered = invoke(&tool, &format!("cat -n {root}/b.log | tail -1"));
    assert_eq!(numbered["stdout"], "2\tERROR three\n");

    let listed = invoke(&tool, &format!("ls -la {root} | wc -l"));
    assert_eq!(listed["exit_code"], 0, "{listed}");
    assert_eq!(listed["stdout"], "4\n", "a.log, b.log, notes.txt, hits.txt");
}

/// Parse errors are presented like any other failure, not returned as `Err`.
#[test]
fn run_parse_error_surfaces_as_present_result() {
    let dir = fresh_dir();
    let tool = tool_with_dir(dir.path());
    let result = invoke(&tool, "echo hi |");
    assert_eq!(result["exit_code"], 2);
    assert_eq!(
        result["stderr"],
        "[error] parse: trailing operator '|'. Try: add a command after the operator\n"
    );
    let output = result["output"].as_str().unwrap();
    assert!(output.starts_with("[stderr] [error] parse: "), "{output}");
    assert_footer(output, 2);
}

/// Truncation applies only to the final output: a stage in the middle
/// of a pipe hands its whole stream on.
#[test]
fn run_pipe_integrity_full_bytes_reach_final_stage() {
    let mut reg = CommandRegistry::new();
    reg.register(Lines(5000));
    reg.register(ByteCount);
    let dir = fresh_dir();
    let tool = tool_with(dir.path(), Arc::new(reg));
    let result = invoke(&tool, "lines | bytecount");
    let expected_bytes: usize = (1..=5000).map(|i| format!("line {i}\n").len()).sum();
    assert_eq!(result["exit_code"], 0);
    assert_eq!(result["stdout"], format!("{expected_bytes}\n"));
    assert_eq!(result["truncated"], false);
    assert!(result.get("overflow_file").is_none());
}

/// The tail hint in the overflow banner must be runnable as printed;
/// `tail` rejects a bare count, so the banner has to spell `-n`.
#[test]
fn run_overflow_tail_hint_runs_as_printed() {
    let mut reg = CommandRegistry::new();
    reg.register(Lines(5000));
    reg.register(CatCommand);
    reg.register(TailCommand);
    let dir = fresh_dir();
    let tool = tool_with(dir.path(), Arc::new(reg));

    let first = invoke(&tool, "lines");
    let output = first["output"].as_str().unwrap();
    let hint = output
        .lines()
        .find(|l| l.contains("| tail"))
        .expect("tail hint in banner");

    let followup = invoke(&tool, hint);
    assert_eq!(followup["exit_code"], 0, "{followup}");
    let expected: String = (4901..=5000).map(|i| format!("line {i}\n")).collect();
    assert_eq!(followup["stdout"], expected);
}

/// `/dev/stdin` is the daemon's own terminal; reading it would block the
/// turn until someone typed there.
#[test]
fn run_refuses_to_read_the_daemon_terminal() {
    let dir = fresh_dir();
    let tool = tool_with(dir.path(), full_registry());
    let result = invoke(
        &tool,
        "echo \"via stdin\" | cat /dev/stdin | head -c 0 || true",
    );
    let stderr = result["stderr"].as_str().unwrap();
    assert!(stderr.contains("not a regular file"), "{result}");
}
