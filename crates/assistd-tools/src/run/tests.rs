use std::path::Path;

use assistd_config::defaults::nz32;
use regex::Regex;
use tempfile::{TempDir, tempdir};

use super::*;
use crate::command::{Command, CommandInput};
use crate::commands::{
    CatCommand, EchoCommand, GrepCommand, LsCommand, SeeCommand, TailCommand, WcCommand,
};
use crate::fixtures::PNG_BYTES;

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
    async fn run(&self, _input: CommandInput) -> CommandOutput {
        let mut stdout = Vec::with_capacity(self.0 * 8);
        for i in 1..=self.0 {
            stdout.extend_from_slice(format!("line {i}\n").as_bytes());
        }
        CommandOutput::ok(stdout)
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
fn run_cat_returns_file_contents() {
    let dir = fresh_dir();
    let files = tempdir().unwrap();
    let path = files.path().join("notes.md");
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

#[test]
fn run_attachments_flow_through_pipeline() {
    let dir = fresh_dir();
    let files = tempdir().unwrap();
    let path = files.path().join("shot.png");
    std::fs::write(&path, PNG_BYTES).unwrap();
    let tool = tool_with_dir(dir.path());
    let cmd = format!("see {} | wc -l", path.to_string_lossy());
    let result = invoke(&tool, &cmd);
    assert_eq!(result["exit_code"], 0);
    let attachments = result["attachments"].as_array().expect("attachments");
    assert_eq!(attachments.len(), 1);
    assert_eq!(attachments[0]["mime"], "image/png");
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

/// A pipeline reports its last stage's exit code, so an earlier failure is
/// visible only in stderr and must be shown.
#[test]
fn run_surfaces_a_failed_stage_behind_a_successful_one() {
    let dir = fresh_dir();
    let tool = tool_with(dir.path(), full_registry());
    let result = invoke(&tool, "find . -name Cargo.toml | head -3");
    assert_eq!(result["exit_code"], 0, "{result}");
    let output = result["output"].as_str().unwrap();
    assert!(
        output.contains("[stderr] [error] unknown command: find"),
        "{output}"
    );
}

#[test]
fn run_quoted_glob_reaches_the_command_literally() {
    let dir = fresh_dir();
    let tool = tool_with(dir.path(), registry());
    let result = invoke(&tool, "echo 'a*b' | grep 'a\\*b'");
    assert_eq!(result["exit_code"], 0, "{result}");
    assert_eq!(result["stdout"], "a*b\n");
}

/// Usage goes to stdout with exit 2 and empty stderr; bare `ls` and `echo`
/// do real work, so only `--help` reaches their usage.
#[test]
fn run_prints_usage_on_stdout_for_bare_calls_and_help_flags() {
    let dir = fresh_dir();
    let tool = tool_with(dir.path(), full_registry());
    for (line, usage) in [
        ("grep", "usage: grep"),
        ("see", "usage: see"),
        ("write", "usage: write"),
        ("web", "usage: web"),
        ("bash", "usage: bash"),
        ("ls --help", "usage: ls"),
        ("echo --help", "usage: echo"),
        ("grep --help", "usage: grep"),
    ] {
        let result = invoke(&tool, line);
        assert_eq!(result["exit_code"], 2, "{line}: {result}");
        assert_eq!(result["stderr"], "", "{line}: {result}");
        assert!(
            result["stdout"]
                .as_str()
                .unwrap_or_default()
                .starts_with(usage),
            "{line}: {result}"
        );
    }
}

#[test]
fn unquoted_bre_alternation_names_the_quoting_fix() {
    let dir = fresh_dir();
    let tool = tool_with(dir.path(), full_registry());
    let result = invoke(&tool, r"grep -r TODO\|FIXME AGENTS.md");
    assert_eq!(result["exit_code"], 2, "{result}");
    assert_eq!(
        result["stderr"],
        concat!(
            r#"[error] parse: '\|' outside quotes: the '|' opened a pipeline and the '\' "#,
            r#"stayed on the previous word. Use: a quoted ERE pattern, as in "#,
            r#"`grep "TODO|FIXME" FILE`"#,
            "\n"
        )
    );
}

#[test]
fn run_rejects_wrong_argument_key() {
    let dir = fresh_dir();
    let tool = tool_with_dir(dir.path());
    let rt = tokio::runtime::Runtime::new().unwrap();
    let err = rt
        .block_on(tool.invoke(json!({ "cmd": "ls" })))
        .unwrap_err();
    assert!(
        matches!(&err, ToolError::InvalidArgs(msg) if msg == "`command` (string) is required"),
        "{err:?}"
    );
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

#[test]
fn run_omits_attachments_key_when_empty() {
    let dir = fresh_dir();
    let tool = tool_with_dir(dir.path());
    let result = invoke(&tool, "echo hi");
    assert!(result.get("attachments").is_none(), "{result}");
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

    let expected: String = (1..=5000).map(|i| format!("line {i}\n")).collect();
    assert_eq!(std::fs::read_to_string(path).unwrap(), expected);

    let output = result["output"].as_str().unwrap();
    assert!(
        output.contains(&format!("Full output: {path}\n")),
        "{output}"
    );
    assert_footer(output, 0);
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

#[test]
fn run_respects_config_overrides() {
    let mut reg = CommandRegistry::new();
    reg.register(Lines(10));
    let dir = fresh_dir();
    let tight = ToolsOutputConfig {
        max_lines: nz32(3),
        max_kb: nz32(10),
        overflow_dir: PathBuf::new(),
    };
    let tool = RunTool::new(Arc::new(reg), &tight, dir.path().to_path_buf());
    let result = invoke(&tool, "lines");
    assert_eq!(result["truncated"], true);
    assert_eq!(result["stdout"], "line 1\nline 2\nline 3\n");
    let output = result["output"].as_str().unwrap();
    assert!(
        output.contains("--- output truncated (10 lines,"),
        "{output}"
    );
}

#[test]
fn run_tool_description_lists_every_command_with_its_summary() {
    let dir = fresh_dir();
    let reg = full_registry();
    let tool = tool_with(dir.path(), Arc::clone(&reg));
    let desc = tool.description();
    let listed: Vec<(&str, &str)> = desc
        .lines()
        .filter_map(|l| l.strip_prefix("  "))
        .filter_map(|l| l.split_once(": "))
        .map(|(name, summary)| (name.trim_end(), summary))
        .collect();
    assert_eq!(listed, reg.sorted_summaries(), "{desc}");
    assert!(
        desc.contains("Call a command with no (or insufficient) arguments to see its usage"),
        "{desc}"
    );
}

#[test]
fn run_tool_description_states_configured_truncation_limits() {
    let dir = fresh_dir();
    let tight = ToolsOutputConfig {
        max_lines: nz32(3),
        max_kb: nz32(10),
        overflow_dir: PathBuf::new(),
    };
    let tool = RunTool::new(registry(), &tight, dir.path().to_path_buf());
    let desc = tool.description();
    assert!(desc.contains("exceeds 3 lines or 10KB"), "{desc}");
    assert!(
        desc.contains(&format!("{}/cmd-N.txt", dir.path().display())),
        "{desc}"
    );
}

/// An unknown flag is an error on stderr, unlike help on stdout.
#[test]
fn run_grep_real_usage_error_still_on_stderr() {
    let dir = fresh_dir();
    let tool = tool_with_dir(dir.path());
    let result = invoke(&tool, "grep -x foo");
    assert_eq!(result["exit_code"], 2);
    let stderr = result["stderr"].as_str().unwrap();
    assert!(stderr.starts_with("[grep]\t[error] grep: "), "{stderr}");
    assert_eq!(result["stdout"], "");
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
