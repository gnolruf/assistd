use async_trait::async_trait;

use super::*;
use crate::chain::{MAX_OPERATORS, parse_chain};
use crate::command::{Command, CommandInput, CommandOutput, CommandRegistry};

/// Emits fixed stdout, stderr and exit code, ignoring its input.
#[derive(Debug)]
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
#[derive(Debug)]
struct Echo;
#[async_trait]
impl Command for Echo {
    fn name(&self) -> &'static str {
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
#[derive(Debug)]
struct LineCount;
#[async_trait]
impl Command for LineCount {
    fn name(&self) -> &'static str {
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
#[derive(Debug)]
struct StdinKind;
#[async_trait]
impl Command for StdinKind {
    fn name(&self) -> &'static str {
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

/// Emits bytes of configurable length; exercises `OUTPUT_MAX`.
#[derive(Debug)]
struct Flood(usize);
#[async_trait]
impl Command for Flood {
    fn name(&self) -> &'static str {
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

/// Attaches one image of configurable size; exercises the attachment caps.
#[derive(Debug)]
struct Picture {
    name: &'static str,
    size: usize,
}
#[async_trait]
impl Command for Picture {
    fn name(&self) -> &str {
        self.name
    }
    fn summary(&self) -> &'static str {
        "test: attach an image"
    }
    fn help(&self) -> String {
        "usage: picture".to_string()
    }
    async fn run(&self, _input: CommandInput) -> CommandOutput {
        CommandOutput {
            attachments: vec![Attachment::Image {
                mime: "image/png".to_string(),
                bytes: vec![0; self.size],
            }],
            ..CommandOutput::ok(b"attached\n".to_vec())
        }
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

/// A piped stage gets `Some` stdin even when upstream printed nothing.
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
async fn output_max_stops_a_flood_before_the_next_stage() {
    let mut r = CommandRegistry::new();
    r.register(Flood(OUTPUT_MAX + 1024));
    r.register(LineCount);
    let out = run_line("flood | lc", &r).await;
    assert_eq!(out.exit_code, 141);
    assert!(out.stdout.is_empty(), "flooded bytes must not be forwarded");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!(
            "[error] pipe: stage output exceeded {OUTPUT_MAX} bytes. \
             Try: pipe through wc -l or head first to shrink the stream\n"
        )
    );
}

const RUN_OVERFLOW: &str = "[error] run: output exceeded 10485760 bytes. \
     Try: fewer files per command, or grep -l / grep -c to find what matters first\n";

#[tokio::test]
async fn seq_joined_output_past_output_max_fails_as_a_whole() {
    let mut r = CommandRegistry::new();
    r.register(Flood(OUTPUT_MAX / 2 + 1));
    let out = run_line("flood; flood", &r).await;
    assert_eq!(out.exit_code, 141);
    assert!(out.stdout.is_empty(), "joined bytes must be dropped");
    assert_eq!(String::from_utf8_lossy(&out.stderr), RUN_OVERFLOW);
}

#[tokio::test]
async fn overflowing_stage_fails_so_or_falls_back() {
    let mut r = registry_of([Stub::new("fallback", b"ok\n", 0)]);
    r.register(Flood(OUTPUT_MAX + 1));
    let out = run_line("flood || fallback", &r).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"ok\n");
    assert_eq!(String::from_utf8_lossy(&out.stderr), RUN_OVERFLOW);
}

#[test]
fn stderr_past_output_max_is_cut_to_whole_lines() {
    let line = b"abcdef\n";
    let out = within_output_max(CommandOutput::failed(
        1,
        line.repeat(OUTPUT_MAX / line.len() + 1),
    ));
    assert_eq!(out.exit_code, 141);
    let kept = out
        .stderr
        .strip_suffix(RUN_OVERFLOW.as_bytes())
        .expect("ends with the overflow line");
    assert_eq!(kept.len(), OUTPUT_MAX / line.len() * line.len());
    assert!(kept.chunks(line.len()).all(|chunk| chunk == line));
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

#[tokio::test]
async fn a_chain_at_the_operator_limit_runs_without_exhausting_the_stack() {
    let r = registry_of([Stub::new("ok", b"x", 0)]);
    let line = ["ok"; MAX_OPERATORS + 1].join(" && ");
    let out = run_line(&line, &r).await;
    assert_eq!(out.stdout, vec![b'x'; MAX_OPERATORS + 1]);
    assert_eq!(out.exit_code, 0);
}

const ATTACHMENT_OVERFLOW: &str = "[error] run: images past 4 or 33554432 bytes in total were \
     dropped. Try: fewer images per command, such as one see per run call\n";

fn picture_registry(pictures: impl IntoIterator<Item = (&'static str, usize)>) -> CommandRegistry {
    let mut r = registry_of([Stub::new("fallback", b"ok\n", 0)]);
    r.register(Echo);
    for (name, size) in pictures {
        r.register(Picture { name, size });
    }
    r
}

fn attachment_sizes(out: &CommandOutput) -> Vec<usize> {
    out.attachments.iter().map(attachment_bytes).collect()
}

#[tokio::test]
async fn images_past_the_count_cap_are_dropped_and_reported_once() {
    let r = picture_registry([("pic", 8)]);
    let out = run_line(&["pic"; 6].join("; "), &r).await;
    assert_eq!(attachment_sizes(&out), [8; ATTACHMENTS_MAX]);
    assert_eq!(out.exit_code, 141);
    assert_eq!(String::from_utf8_lossy(&out.stderr), ATTACHMENT_OVERFLOW);
}

#[tokio::test]
async fn images_past_the_byte_cap_are_dropped_with_every_later_one() {
    let half = ATTACHMENT_BYTES_MAX / 2;
    let r = picture_registry([("half", half), ("big", ATTACHMENT_BYTES_MAX), ("tiny", 1)]);
    let out = run_line("half | echo_stdin; half; big; tiny", &r).await;
    assert_eq!(attachment_sizes(&out), [half, half]);
    assert_eq!(out.exit_code, 141);
    assert_eq!(String::from_utf8_lossy(&out.stderr), ATTACHMENT_OVERFLOW);
}

#[tokio::test]
async fn a_dropped_image_fails_its_stage_so_and_stops_and_or_falls_back() {
    let r = picture_registry([("pic", 8)]);
    let and_line = ["pic"; ATTACHMENTS_MAX + 2].join(" && ");
    let and_out = run_line(&format!("{and_line} && fallback"), &r).await;
    assert_eq!(and_out.exit_code, 141);
    assert!(!and_out.stdout.ends_with(b"ok\n"));

    let or_line = ["pic"; ATTACHMENTS_MAX + 1].join("; ");
    let or_out = run_line(&format!("{or_line} || fallback"), &r).await;
    assert_eq!(or_out.exit_code, 0);
    assert!(or_out.stdout.ends_with(b"ok\n"));
    assert_eq!(attachment_sizes(&or_out), [8; ATTACHMENTS_MAX]);
}

#[tokio::test]
async fn images_within_both_caps_pass_through_untouched() {
    let r = picture_registry([("pic", 8), ("rest", ATTACHMENT_BYTES_MAX - 24)]);
    let out = run_line("pic | echo_stdin; pic && pic || fallback; rest", &r).await;
    assert_eq!(attachment_sizes(&out), [8, 8, 8, ATTACHMENT_BYTES_MAX - 24]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stderr.is_empty());
}
