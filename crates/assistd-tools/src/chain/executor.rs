//! Walks a [`Chain`], dispatching each stage through a [`CommandRegistry`]
//! and combining results with Unix pipeline semantics.

use std::future::Future;
use std::pin::Pin;

use assistd_ipc::Attachment;

use super::expand::{ExpandError, expand_args};
use super::{Chain, Word};
use crate::TurnContext;
use crate::command::{
    CommandInput, CommandOutput, CommandRegistry, Hint, error_line, run_rendered,
};

/// Most bytes of stdout, or of stderr, that one stage or a whole chain may
/// hold. Overflow exits 141, the SIGPIPE code, so `||` fallbacks still fire.
pub const OUTPUT_MAX: usize = 10 * 1024 * 1024;

/// Most attachments one chain may carry.
const ATTACHMENTS_MAX: usize = 4;

/// Most attachment bytes, summed over the chain, that one chain may carry.
const ATTACHMENT_BYTES_MAX: usize = 32 * 1024 * 1024;

/// Attachments admitted so far in one chain. Once one is dropped, every
/// later one is too, so the kept attachments are always the earliest.
#[derive(Debug)]
enum AttachmentBudget {
    Open { images: usize, bytes: usize },
    Spent,
}

impl AttachmentBudget {
    const fn new() -> Self {
        Self::Open {
            images: 0,
            bytes: 0,
        }
    }

    /// Keep the attachments of `out` that still fit. A stage that loses any
    /// exits 141, and the first such stage also reports the drop on stderr.
    fn admit(&mut self, mut out: CommandOutput) -> CommandOutput {
        let was_open = matches!(self, Self::Open { .. });
        let fitting = self.count_fitting(&out.attachments);
        if fitting == out.attachments.len() {
            return out;
        }
        out.attachments.truncate(fitting);
        out.exit_code = 141;
        if was_open {
            out.stderr.extend(attachment_overflow_line().into_bytes());
        }
        out
    }

    fn count_fitting(&mut self, attachments: &[Attachment]) -> usize {
        let mut fitting = 0;
        for attachment in attachments {
            let Self::Open { images, bytes } = self else {
                break;
            };
            let size = attachment_bytes(attachment);
            if *images == ATTACHMENTS_MAX || *bytes + size > ATTACHMENT_BYTES_MAX {
                *self = Self::Spent;
                break;
            }
            *images += 1;
            *bytes += size;
            fitting += 1;
        }
        fitting
    }
}

/// Execute a parsed chain, piping stages in sequence with `pipefail` exit status and stderr lines
/// prefixed `[name]\t`. Output past [`OUTPUT_MAX`] fails with an overflow; attachments past the
/// first four or 32 MiB in all are dropped and their stage exits 141.
pub async fn execute(
    chain: &Chain,
    registry: &CommandRegistry,
    turn: &TurnContext,
) -> CommandOutput {
    let mut budget = AttachmentBudget::new();
    execute_within(chain, registry, turn, &mut budget).await
}

fn execute_within<'a>(
    chain: &'a Chain,
    registry: &'a CommandRegistry,
    turn: &'a TurnContext,
    budget: &'a mut AttachmentBudget,
) -> Pin<Box<dyn Future<Output = CommandOutput> + Send + 'a>> {
    Box::pin(
        async move { within_output_max(execute_uncapped(chain, registry, turn, budget).await) },
    )
}

fn execute_uncapped<'a>(
    chain: &'a Chain,
    registry: &'a CommandRegistry,
    turn: &'a TurnContext,
    budget: &'a mut AttachmentBudget,
) -> Pin<Box<dyn Future<Output = CommandOutput> + Send + 'a>> {
    Box::pin(async move {
        match chain {
            Chain::Command(argv) => budget.admit(run_command(argv, registry, turn, None).await),
            Chain::Pipe(l, argv) => {
                let mut left = execute_uncapped(l, registry, turn, budget).await;
                let piped = std::mem::take(&mut left.stdout);
                if piped.len() > OUTPUT_MAX {
                    return left.then(pipe_overflow());
                }
                let right = budget.admit(run_command(argv, registry, turn, Some(piped)).await);
                pipe_join(left, right)
            }
            Chain::And(l, r) | Chain::Or(l, r) => {
                let left = execute_within(l, registry, turn, budget).await;
                let run_right = matches!(chain, Chain::And(..)) == (left.exit_code == 0);
                if !run_right {
                    return left;
                }
                let right = execute_within(r, registry, turn, budget).await;
                left.then(right)
            }
            Chain::Seq(l, r) => {
                let left = execute_within(l, registry, turn, budget).await;
                let right = execute_within(r, registry, turn, budget).await;
                left.then(right)
            }
        }
    })
}

/// `left` then `right`, keeping `left`'s failure when `right` succeeds.
fn pipe_join(left: CommandOutput, right: CommandOutput) -> CommandOutput {
    let left_exit = left.exit_code;
    let mut joined = left.then(right);
    if joined.exit_code == 0 {
        joined.exit_code = left_exit;
    }
    joined
}

fn within_output_max(mut out: CommandOutput) -> CommandOutput {
    if out.stdout.len() <= OUTPUT_MAX && out.stderr.len() <= OUTPUT_MAX {
        return out;
    }
    out.stdout.clear();
    out.stderr.truncate(last_line_end_within_max(&out.stderr));
    out.then(CommandOutput::failed(
        141,
        error_line(
            "run",
            format_args!("output exceeded {OUTPUT_MAX} bytes"),
            Hint::Try,
            "fewer files per command, or grep -l / grep -c to find what matters first",
        )
        .into_bytes(),
    ))
}

fn last_line_end_within_max(stream: &[u8]) -> usize {
    stream[..stream.len().min(OUTPUT_MAX)]
        .iter()
        .rposition(|&b| b == b'\n')
        .map_or(0, |newline| newline + 1)
}

fn pipe_overflow() -> CommandOutput {
    CommandOutput::failed(
        141,
        error_line(
            "pipe",
            format_args!("stage output exceeded {OUTPUT_MAX} bytes"),
            Hint::Try,
            "pipe through wc -l or head first to shrink the stream",
        )
        .into_bytes(),
    )
}

fn attachment_overflow_line() -> String {
    error_line(
        "run",
        format_args!(
            "images past {ATTACHMENTS_MAX} or {ATTACHMENT_BYTES_MAX} bytes in total were dropped"
        ),
        Hint::Try,
        "fewer images per command, such as one see per run call",
    )
}

fn attachment_bytes(attachment: &Attachment) -> usize {
    match attachment {
        Attachment::Image { bytes, .. } => bytes.len(),
    }
}

/// Dispatch one stage. An unquoted `--help` before any `--` prints usage,
/// even for commands whose bare form does real work.
async fn run_command(
    words: &[Word],
    registry: &CommandRegistry,
    turn: &TurnContext,
    stdin: Option<Vec<u8>>,
) -> CommandOutput {
    let name = words.first().map_or_default(|w| w.text.as_str());
    if name.is_empty() {
        return CommandOutput::failed(
            2,
            error_line(
                "run",
                "empty command",
                Hint::Use,
                "run <cmd> (see tool description for available commands)",
            )
            .into_bytes(),
        );
    }

    let Some(cmd) = registry.get(name) else {
        let avail = registry.sorted_names().join(", ");
        let msg = format!("[error] unknown command: {name}. Available: {avail}\n");
        return CommandOutput::failed(127, msg.into_bytes());
    };

    let args = if cmd.expands_args() {
        match expand_args(&words[1..], registry.scope()).await {
            Ok(args) => args,
            Err(err) => return glob_failure(name, &err),
        }
    } else {
        words[1..].to_vec()
    };
    if asks_for_help(&args) {
        return CommandOutput::usage(cmd.help());
    }

    let input = CommandInput {
        args,
        stdin,
        scope: registry.scope().clone(),
        turn: turn.clone(),
    };
    let out = run_rendered(cmd, input).await;

    let stderr = if out.stderr.is_empty() {
        Vec::new()
    } else {
        prefix_stderr(name, &out.stderr)
    };
    CommandOutput { stderr, ..out }
}

fn asks_for_help(args: &[Word]) -> bool {
    args.iter()
        .take_while(|arg| arg.text != "--")
        .any(|arg| !arg.quoted && arg.text == "--help")
}

fn glob_failure(name: &str, err: &ExpandError) -> CommandOutput {
    CommandOutput::failed(
        1,
        error_line(
            name,
            err,
            Hint::Try,
            "narrow the pattern, or ls the directory first",
        )
        .into_bytes(),
    )
}

fn prefix_stderr(name: &str, raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() + name.len() + 4);
    let mut line_start = true;
    for &b in raw {
        if line_start {
            out.push(b'[');
            out.extend_from_slice(name.as_bytes());
            out.push(b']');
            out.push(b'\t');
            line_start = false;
        }
        out.push(b);
        if b == b'\n' {
            line_start = true;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;

    use assistd_ipc::ImageMime;

    use super::*;
    use crate::chain::{MAX_OPERATORS, parse_chain};
    use crate::command::{
        Command, CommandError, CommandInput, CommandOutput, CommandRegistry, Usage,
    };

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
        fn usage(&self) -> Usage {
            Usage::new("", "")
        }
        async fn run(&self, _input: CommandInput) -> Result<CommandOutput, CommandError> {
            Ok(CommandOutput {
                stdout: self.stdout.to_vec(),
                stderr: self.stderr.to_vec(),
                exit_code: self.exit_code,
                ..CommandOutput::default()
            })
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
        fn usage(&self) -> Usage {
            Usage::new("", "")
        }
        async fn run(&self, input: CommandInput) -> Result<CommandOutput, CommandError> {
            Ok(CommandOutput::ok(input.stdin.unwrap_or_default()))
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
        fn usage(&self) -> Usage {
            Usage::new("", "")
        }
        async fn run(&self, input: CommandInput) -> Result<CommandOutput, CommandError> {
            let n = input
                .stdin
                .unwrap_or_default()
                .iter()
                .filter(|b| **b == b'\n')
                .count();
            Ok(CommandOutput::ok(format!("{n}\n").into_bytes()))
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
        fn usage(&self) -> Usage {
            Usage::new("", "")
        }
        async fn run(&self, input: CommandInput) -> Result<CommandOutput, CommandError> {
            let kind = if input.stdin.is_some() {
                "piped"
            } else {
                "none"
            };
            Ok(CommandOutput::ok(kind.as_bytes().to_vec()))
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
        fn usage(&self) -> Usage {
            Usage::new("", "")
        }
        async fn run(&self, _input: CommandInput) -> Result<CommandOutput, CommandError> {
            Ok(CommandOutput::ok(vec![b'x'; self.0]))
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
        fn usage(&self) -> Usage {
            Usage::new("", "")
        }
        async fn run(&self, _input: CommandInput) -> Result<CommandOutput, CommandError> {
            Ok(CommandOutput {
                attachments: vec![Attachment::Image {
                    mime: ImageMime::Png,
                    bytes: vec![0; self.size],
                }],
                ..CommandOutput::ok(b"attached\n".to_vec())
            })
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
        execute(&chain, registry, &TurnContext::default()).await
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
    async fn seq_runs_both_regardless_of_exit() {
        let r = registry_of([
            Stub::new("first", b"a\n", 5),
            Stub::new("second", b"b\n", 0),
        ]);
        let out = run_line("first ; second", &r).await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"a\nb\n");
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
    async fn overflowing_stage_fails_so_or_falls_back() {
        let mut r = registry_of([Stub::new("fallback", b"ok\n", 0)]);
        r.register(Flood(OUTPUT_MAX + 1));
        let out = run_line("flood || fallback", &r).await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"ok\n");
        assert_eq!(String::from_utf8_lossy(&out.stderr), RUN_OVERFLOW);
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
    async fn a_pipe_exits_with_its_rightmost_failing_stage() {
        let mut r = registry_of([Stub::new("fail2", b"x\n", 2), Stub::new("fail3", b"y\n", 3)]);
        r.register(LineCount);
        assert_eq!(run_line("fail2 | lc", &r).await.exit_code, 2);
        assert_eq!(run_line("fail2 | fail3 | lc", &r).await.exit_code, 3);
    }

    #[tokio::test]
    async fn only_an_unquoted_help_flag_before_a_double_dash_prints_usage() {
        let mut r = CommandRegistry::new();
        r.register(StdinKind);
        for line in ["stdin_kind --help", "stdin_kind -x --help"] {
            assert_eq!(run_line(line, &r).await.exit_code, 2, "{line:?}");
        }
        for line in ["stdin_kind '--help'", "stdin_kind -- --help"] {
            assert_eq!(run_line(line, &r).await.stdout, b"none", "{line:?}");
        }
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

    fn picture_registry(
        pictures: impl IntoIterator<Item = (&'static str, usize)>,
    ) -> CommandRegistry {
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
}
