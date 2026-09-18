use super::*;
use crate::ToolRegistry;
use crate::command::{Command, CommandInput, CommandOutput};
use crate::commands::{
    BashCommand, CatCommand, EchoCommand, GrepCommand, LsCommand, SeeCommand, WcCommand,
    WebCommand, WriteCommand,
};
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
    r.register(SeeCommand::default());
    Arc::new(r)
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
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
    0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
    0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
    0x42, 0x60, 0x82,
];

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
    // `cat` rejects the binary file at Layer 1 (exit 1 with stderr).
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
    assert!(
        stderr.contains("[error] cat: binary image file"),
        "{stderr}"
    );
    assert!(stderr.contains("Use: see "), "{stderr}");
    let output = result["output"].as_str().unwrap();
    assert!(output.contains("[stderr] "), "output={output}");
    assert!(output.contains("binary image file"), "output={output}");
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

/// The pipelines the model actually writes: flags on the built-in
/// commands, a glob, and the stream filters composed end to end.
#[test]
fn run_composes_flags_globs_and_stream_filters() {
    let dir = fresh_dir();
    let tmp = tempdir().unwrap();
    std::fs::write(tmp.path().join("a.log"), b"ERROR one\ninfo\n").unwrap();
    std::fs::write(tmp.path().join("b.log"), b"ERROR two\nERROR three\n").unwrap();
    std::fs::write(tmp.path().join("notes.txt"), b"ERROR ignored\n").unwrap();
    let tool = tool_with(dir.path(), full_registry());
    let root = tmp.path().to_string_lossy().into_owned();

    // A glob feeds two files into grep, which labels its output.
    let globbed = invoke(&tool, &format!("grep ERROR {root}/*.log | wc -l"));
    assert_eq!(globbed["exit_code"], 0, "{globbed}");
    assert_eq!(globbed["stdout"], "3\n");

    // -r walks the directory instead, and numbers the hits.
    let recursive = invoke(&tool, &format!("grep -rn ERROR {root} | wc -l"));
    assert_eq!(recursive["exit_code"], 0, "{recursive}");
    assert_eq!(recursive["stdout"], "4\n");

    // The frequency idiom, start to finish.
    std::fs::write(tmp.path().join("hits.txt"), b"b\na\nb\n").unwrap();
    let freq = invoke(
        &tool,
        &format!("cat {root}/hits.txt | sort | uniq -c | sort -nr | head -1"),
    );
    assert_eq!(freq["stdout"], "2\tb\n", "{freq}");

    // cat -n numbers, tail slices, ls -a survives its flags.
    let numbered = invoke(&tool, &format!("cat -n {root}/b.log | tail -1"));
    assert_eq!(numbered["stdout"], "2\tERROR three\n");
    // a.log, b.log, notes.txt, hits.txt.
    let listed = invoke(&tool, &format!("ls -la {root} | wc -l"));
    assert_eq!(listed["exit_code"], 0, "{listed}");
    assert_eq!(listed["stdout"], "4\n");
}

/// A failing stage followed by a succeeding one reports the exit
/// code of the last stage, per Unix. The failure is then only
/// visible in stderr, so the model has to be shown it — otherwise
/// `find . | head` reads as "ran fine, found nothing".
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
    // Quoted, so the regex must not be mistaken for a file pattern.
    let result = invoke(&tool, "echo 'a*b' | grep 'a\\*b'");
    assert_eq!(result["exit_code"], 0, "{result}");
    assert_eq!(result["stdout"], "a*b\n");
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
fn help_flag_works_even_where_bare_calls_do_work() {
    let dir = fresh_dir();
    let tool = tool_with(dir.path(), full_registry());
    for (cmd, usage) in [
        ("ls --help", "usage: ls"),
        ("echo --help", "usage: echo"),
        ("grep --help", "usage: grep"),
    ] {
        let result = invoke(&tool, cmd);
        assert_eq!(result["exit_code"], 2, "{cmd}: {result}");
        assert!(
            result["stdout"]
                .as_str()
                .unwrap_or_default()
                .starts_with(usage),
            "{cmd}: {result}"
        );
    }
}

#[test]
fn unquoted_bre_alternation_names_the_quoting_fix() {
    let dir = fresh_dir();
    let tool = tool_with(dir.path(), full_registry());
    let result = invoke(&tool, r"grep -r TODO\|FIXME AGENTS.md");
    assert_eq!(result["exit_code"], 2, "{result}");
    let stderr = result["stderr"].as_str().unwrap_or_default();
    assert!(stderr.contains("outside quotes"), "{stderr}");
    assert!(stderr.contains(r#"grep "TODO|FIXME" FILE"#), "{stderr}");
}

#[test]
fn run_head_reads_a_named_file() {
    let dir = fresh_dir();
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("log.txt");
    std::fs::write(&path, b"one\ntwo\nthree\nfour\n").unwrap();
    let tool = tool_with(dir.path(), full_registry());

    // The spelling the model reaches for first, end to end through
    // the parser and executor.
    let result = invoke(&tool, &format!("head -n 2 {}", path.to_string_lossy()));
    assert_eq!(result["exit_code"], 0, "{result}");
    assert_eq!(result["stdout"], "one\ntwo\n");

    // The pipeline spelling keeps working.
    let piped = invoke(
        &tool,
        &format!("cat {} | head -n 2", path.to_string_lossy()),
    );
    assert_eq!(piped["stdout"], "one\ntwo\n");
}

#[test]
fn run_wc_counts_a_named_file() {
    let dir = fresh_dir();
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("log.txt");
    std::fs::write(&path, b"a\nb\nc\n").unwrap();
    let tool = tool_with_dir(dir.path());
    let result = invoke(&tool, &format!("wc -l {}", path.to_string_lossy()));
    assert_eq!(result["exit_code"], 0, "{result}");
    assert_eq!(result["stdout"], "3\n");
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
fn run_parse_error_surfaces_as_present_result() {
    // Parse errors flow through `present()` like any other failure: the
    // LLM sees the usual `[stderr] ... [exit:N | Xms]` shape with a
    // `[error] parse: ...` line, not a raw anyhow at the tool boundary.
    let dir = fresh_dir();
    let tool = tool_with_dir(dir.path());
    let result = invoke(&tool, "echo hi |");
    assert_eq!(result["exit_code"], 2);
    let stderr = result["stderr"].as_str().unwrap();
    assert!(stderr.contains("[error] parse: "), "{stderr}");
    let output = result["output"].as_str().unwrap();
    assert!(output.contains("[stderr] "), "{output}");
    assert_footer(output, 2);
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
            format!("{}\n", input.stdin.map_or(0, |s| s.len())).into_bytes(),
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
    assert!(output.contains(&format!("cat {path} | tail -n 100")));
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

/// The tail hint in the overflow banner must be runnable as printed;
/// `tail` rejects a bare count, so the banner has to spell `-n`.
#[test]
fn run_overflow_tail_hint_runs_as_printed() {
    use crate::commands::TailCommand;
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
    let stdout = followup["stdout"].as_str().unwrap();
    assert_eq!(stdout.lines().count(), 100);
    assert!(stdout.ends_with("line 5000\n"));
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
    let tight = ToolsOutputConfig {
        max_lines: nz32(3),
        max_kb: nz32(10),
        overflow_dir: PathBuf::new(),
    };
    let tool = RunTool::new(Arc::new(reg), &tight, dir.path().to_path_buf());
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
    for name in [
        "cat",
        "ls",
        "grep",
        "wc",
        "head",
        "tail",
        "sort",
        "uniq",
        "echo",
        "write",
        "see",
        "screenshot",
        "web",
        "bash",
        "wm",
    ] {
        assert!(desc.contains(name), "description missing `{name}`: {desc}");
    }
    // Summary text from a representative command should appear.
    assert!(
        desc.contains("filter lines matching an ERE regex"),
        "description missing grep summary: {desc}"
    );
    // Level-1 discovery hint tells the LLM how to drill in.
    assert!(
        desc.to_lowercase()
            .contains("no (or insufficient) arguments")
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
        &ToolsOutputConfig::default(),
        dir.path().to_path_buf(),
    ));
    let schemas = tools.openai_schemas();
    assert_eq!(schemas.len(), 1);
    let desc = schemas[0]["function"]["description"]
        .as_str()
        .expect("description is a string");
    for name in [
        "cat",
        "ls",
        "grep",
        "wc",
        "head",
        "tail",
        "sort",
        "uniq",
        "echo",
        "write",
        "see",
        "screenshot",
        "web",
        "bash",
        "wm",
    ] {
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
    reg.register(SeeCommand::default());
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
    reg.register(WriteCommand::permissive_for_tests());
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
/// the `[grep]\t` executor prefix; exits 2 like help does, but the
/// transport is different. This test locks the distinction.
#[test]
fn run_grep_real_usage_error_still_on_stderr() {
    let dir = fresh_dir();
    let tool = tool_with_dir(dir.path());
    // Unrecognized flag; triggers parse_flags error path, which
    // remains on stderr (unlike the no-args path, which emits help).
    let result = invoke(&tool, "grep -x foo");
    assert_eq!(result["exit_code"], 2);
    let stderr = result["stderr"].as_str().unwrap();
    assert!(
        stderr.contains("[grep]\t"),
        "real errors keep executor prefix: {stderr}"
    );
    // Stdout stays empty; the usage error didn't go to stdout.
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
