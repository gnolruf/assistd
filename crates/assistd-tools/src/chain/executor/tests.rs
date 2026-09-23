use super::*;
use crate::chain::parse_chain;
use crate::command::{Command, CommandInput, CommandOutput, CommandRegistry};
use async_trait::async_trait;

/// Fake command: emits fixed stdout (and optionally stderr) with a
/// fixed exit code, ignoring its input.
struct Stub {
    name: &'static str,
    stdout: &'static [u8],
    stderr: &'static [u8],
    exit_code: i32,
}

impl Stub {
    fn new(name: &'static str, stdout: &'static [u8], exit_code: i32) -> Self {
        Self {
            name,
            stdout,
            stderr: b"",
            exit_code,
        }
    }
}

#[async_trait]
impl Command for Stub {
    fn name(&self) -> &str {
        self.name
    }
    fn summary(&self) -> &'static str {
        "test stub"
    }
    fn help(&self) -> String {
        "stub help".to_string()
    }
    async fn run(&self, _input: CommandInput) -> CommandOutput {
        CommandOutput {
            stdout: self.stdout.to_vec(),
            stderr: self.stderr.to_vec(),
            exit_code: self.exit_code,
            attachments: Vec::new(),
        }
    }
}

/// Echoes stdin to stdout.
struct Echo;
#[async_trait]
impl Command for Echo {
    fn name(&self) -> &str {
        "echo_stdin"
    }
    fn summary(&self) -> &'static str {
        "test: echo stdin"
    }
    fn help(&self) -> String {
        "usage: echo_stdin".to_string()
    }
    async fn run(&self, input: CommandInput) -> CommandOutput {
        CommandOutput::ok(input.stdin.unwrap_or_default())
    }
}

/// Counts newlines in stdin.
struct LineCount;
#[async_trait]
impl Command for LineCount {
    fn name(&self) -> &str {
        "lc"
    }
    fn summary(&self) -> &'static str {
        "test: count lines"
    }
    fn help(&self) -> String {
        "usage: lc".to_string()
    }
    async fn run(&self, input: CommandInput) -> CommandOutput {
        let n = input
            .stdin
            .unwrap_or_default()
            .iter()
            .filter(|b| **b == b'\n')
            .count();
        CommandOutput::ok(format!("{n}\n").into_bytes())
    }
}

/// Reports whether it was handed a stdin at all.
struct StdinKind;
#[async_trait]
impl Command for StdinKind {
    fn name(&self) -> &str {
        "stdin_kind"
    }
    fn summary(&self) -> &'static str {
        "test: report stdin presence"
    }
    fn help(&self) -> String {
        "usage: stdin_kind".to_string()
    }
    async fn run(&self, input: CommandInput) -> CommandOutput {
        let kind = if input.stdin.is_some() {
            "piped"
        } else {
            "none"
        };
        CommandOutput::ok(kind.as_bytes().to_vec())
    }
}

/// Emits bytes of configurable length; exercises PIPE_BUF_MAX.
struct Flood(usize);
#[async_trait]
impl Command for Flood {
    fn name(&self) -> &str {
        "flood"
    }
    fn summary(&self) -> &'static str {
        "test: emit N bytes"
    }
    fn help(&self) -> String {
        "usage: flood".to_string()
    }
    async fn run(&self, _input: CommandInput) -> CommandOutput {
        CommandOutput::ok(vec![b'x'; self.0])
    }
}

fn registry_of(stubs: impl IntoIterator<Item = Stub>) -> CommandRegistry {
    let mut r = CommandRegistry::new();
    for stub in stubs {
        r.register(stub);
    }
    r
}

async fn run_line(line: &str, registry: &CommandRegistry) -> CommandOutput {
    let chain = parse_chain(line).unwrap();
    execute(&chain, registry, None).await
}

#[tokio::test]
async fn pipe_threads_stdout_through_every_stage() {
    let mut r = registry_of([Stub::new("emit", b"a\nb\nc\n", 0)]);
    r.register(Echo);
    r.register(LineCount);
    let out = run_line("emit | echo_stdin | lc", &r).await;
    assert_eq!(out.stdout, b"3\n");
    assert_eq!(out.exit_code, 0);
}

/// Only a stage on the right of a pipe gets `Some` stdin, even when the
/// upstream stage printed nothing; that is how a filter tells "nothing
/// piped" from "empty input".
#[tokio::test]
async fn only_piped_stages_receive_stdin() {
    let mut r = registry_of([Stub::new("silent", b"", 0)]);
    r.register(StdinKind);
    assert_eq!(run_line("stdin_kind", &r).await.stdout, b"none");
    assert_eq!(run_line("silent | stdin_kind", &r).await.stdout, b"piped");
}

#[tokio::test]
async fn and_runs_right_on_success() {
    let r = registry_of([
        Stub::new("ok", b"first", 0),
        Stub::new("right", b"second", 0),
    ]);
    let out = run_line("ok && right", &r).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"firstsecond");
}

#[tokio::test]
async fn and_short_circuits_on_failure() {
    let r = registry_of([Stub::new("bad", b"x", 1), Stub::new("right", b"never", 0)]);
    let out = run_line("bad && right", &r).await;
    assert_eq!(out.exit_code, 1);
    assert_eq!(out.stdout, b"x");
}

#[tokio::test]
async fn or_runs_right_on_failure() {
    let r = registry_of([Stub::new("bad", b"boom", 2), Stub::new("recover", b"ok", 0)]);
    let out = run_line("bad || recover", &r).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"boomok");
}

#[tokio::test]
async fn or_short_circuits_on_success() {
    let r = registry_of([Stub::new("good", b"g", 0), Stub::new("right", b"never", 0)]);
    let out = run_line("good || right", &r).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"g");
}

#[tokio::test]
async fn seq_runs_both_regardless_of_exit() {
    let r = registry_of([
        Stub::new("first", b"a\n", 5),
        Stub::new("second", b"b\n", 0),
    ]);
    let out = run_line("first ; second", &r).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"a\nb\n");
}

/// An unknown command is an ordinary failed stage, so `||` and `&&`
/// treat it like any other non-zero exit.
#[tokio::test]
async fn unknown_command_returns_127_with_available_list() {
    let mut r = CommandRegistry::new();
    r.register(Echo);
    r.register(LineCount);
    let out = run_line("nope", &r).await;
    assert_eq!(out.exit_code, 127);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "[error] unknown command: nope. Available: echo_stdin, lc\n"
    );
}

#[tokio::test]
async fn pipe_buf_max_stops_a_flood_before_the_next_stage() {
    let mut r = CommandRegistry::new();
    r.register(Flood(PIPE_BUF_MAX + 1024));
    r.register(LineCount);
    let out = run_line("flood | lc", &r).await;
    assert_eq!(out.exit_code, 141);
    assert!(out.stdout.is_empty(), "flooded bytes must not be forwarded");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!(
            "[error] pipe: stage output exceeded {PIPE_BUF_MAX} bytes. \
             Try: pipe through wc -l or head first to shrink the stream\n"
        )
    );
}

/// `a && b | lc || d` parses as `(a && (b | lc)) || d`; the `||` sees
/// the pipe's success and skips `d`.
#[tokio::test]
async fn combined_precedence_and_short_circuit() {
    let mut r = registry_of([
        Stub::new("a", b"", 0),
        Stub::new("b", b"hi\n", 0),
        Stub::new("d", b"never", 0),
    ]);
    r.register(LineCount);
    let out = run_line("a && b | lc || d", &r).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"1\n");
}

#[tokio::test]
async fn stderr_lines_are_each_prefixed_with_the_command_name() {
    let r = registry_of([Stub {
        name: "boom",
        stdout: b"",
        stderr: b"first\nsecond",
        exit_code: 1,
    }]);
    let out = run_line("boom", &r).await;
    assert_eq!(out.exit_code, 1);
    assert_eq!(out.stderr, b"[boom]\tfirst\n[boom]\tsecond");
}
