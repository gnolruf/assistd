//! The LLM-facing `run` tool: parses a command line, executes the chain, and
//! truncates only the final output, so every stage sees its whole input.

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use assistd_utils::text::human_size;
use async_trait::async_trait;
use serde_json::{Value, json};

use crate::TurnContext;
use crate::chain::{ParseError, Redirection, execute, parse_chain};
use crate::command::{Command, CommandOutput, CommandRegistry, Hint, error_line};
use crate::presentation::{OutputCap, PresentResult, PresentSpec, present};
use crate::{RawStreams, Tool, ToolError, ToolOutput};

/// The LLM-facing `run` tool, dispatching a command line through the chain
/// parser and executor.
#[derive(Debug)]
pub struct RunTool {
    registry: Arc<CommandRegistry>,
    output_cap: Arc<OutputCap>,
    scratch_dir: Option<PathBuf>,
}

impl RunTool {
    /// Results are cut to `output_cap`, which should be the registry's own.
    pub fn new(registry: Arc<CommandRegistry>, output_cap: Arc<OutputCap>) -> Self {
        Self {
            registry,
            output_cap,
            scratch_dir: None,
        }
    }

    /// Also tell the model that `scratch_dir` is where files shared between
    /// `bash` scripts and the other commands go.
    pub fn with_scratch_dir(mut self, scratch_dir: &Path) -> Self {
        self.scratch_dir = Some(scratch_dir.to_path_buf());
        self
    }
}

/// Rebuilt per request so command summaries that track runtime state, such
/// as vision support, stay current.
fn build_description(
    registry: &CommandRegistry,
    spec: &PresentSpec,
    scratch_dir: Option<&Path>,
) -> String {
    let mut desc = String::with_capacity(1024);
    desc.push_str(
        "Execute a shell-style command line in the daemon's working \
         directory. The commands listed below exist only as the first word \
         of this tool's `command` string, e.g. {\"command\": \"wm list\"}; \
         they are not tools and cannot be called by name. Every other entry \
         in the tool list is a separate tool with its own arguments. \
         Supports pipelines (|), and/or (&&, ||), sequencing (;), `~` and \
         globs (*, ?, []) on unquoted arguments; quote an argument to pass \
         it through literally, including a `|` that belongs to the \
         argument rather than the pipeline (`grep \"a|b\" f.txt`). \
         Redirections (>, <, 2>, 2>&1), here-docs, expansion ($VAR, \
         $(…), backticks), `#` comments, a second line, and backgrounding \
         (&) are NOT supported and are rejected outside quotes: stderr is \
         already captured in the result, and `bash \"…\"` gives a real \
         shell when needed. A pipeline fails when any stage fails, exiting \
         with the last failing stage's code, so `grep x f | wc -l` exits 1 \
         when nothing matches. ",
    );
    desc.push_str(&format!(
        "Output is returned whole unless the result exceeds {max_lines} \
         lines or {max_size}; only then are stdout and stderr cut to their \
         heads and each cut stream saved in full. A cut stream opens with \
         an `[OUTPUT TRUNCATED]` or `[STDERR TRUNCATED]` banner and ends \
         with a notice giving a `Full output:` or `Full stderr:` path under \
         `{dir}/cmd-N.txt` and a `Next page:` command; when reading a file, \
         page through it before drawing conclusions about its contents. \
         No such file exists for a result under those limits.\n\nCommands \
         (first word of `command`):\n",
        max_lines = spec.max_lines,
        max_size = human_size(spec.max_bytes as u64),
        dir = spec.overflow_dir.display(),
    ));
    for (name, synopsis, summary) in registry.sorted_listings() {
        desc.push_str(&format!("  {name} {synopsis}\n      {summary}\n"));
    }
    desc.push_str(
        "\nEach command is listed with the shape of its arguments; a \
         command with subcommands needs one as its first argument. Call a \
         command with no (or insufficient) arguments to see its full \
         usage (exit code 2, stdout); `<cmd> --help` prints the same \
         text and works for commands like `ls` and `echo` whose bare \
         form does real work, while a quoted `--help`, or one after \
         `--`, is an ordinary argument. Errors in real calls exit with a \
         `[<name>]\\t` stderr prefix, distinct from help on stdout. Each \
         error line follows `[error] <cmd>: <what-went-wrong>. <Hint>: \
         <recovery>` where `<Hint>` is one of `Use:` / `Try:` / `Check:` / \
         `Available:`; the recovery clause is a concrete command or check \
         you can run next.",
    );
    if let Some(scratch_dir) = scratch_dir {
        desc.push_str(&format!(
            " `bash` scripts run sandboxed with their own empty /tmp and \
             /run, so a file that both a script and the other commands \
             need goes under `{}`, which they share; the `Full output:` \
             files are readable from scripts too.",
            scratch_dir.display()
        ));
    }
    desc
}

#[async_trait]
impl Tool for RunTool {
    fn name(&self) -> &'static str {
        "run"
    }

    fn description(&self) -> Cow<'_, str> {
        Cow::Owned(build_description(
            &self.registry,
            self.output_cap.spec(),
            self.scratch_dir.as_deref(),
        ))
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

    /// A chain reads this machine when any command it names does; one that does not parse,
    /// or names no known command, runs nothing.
    fn reads_host(&self, args: &Value) -> bool {
        args.get("command")
            .and_then(Value::as_str)
            .and_then(|command| parse_chain(command).ok())
            .is_some_and(|chain| {
                chain
                    .commands()
                    .into_iter()
                    .filter_map(|argv| self.registry.get(argv.first()?.as_str()))
                    .any(Command::reads_host)
            })
    }

    #[tracing::instrument(skip(self, turn, args), fields(cmd = tracing::field::Empty))]
    async fn invoke(&self, turn: &TurnContext, args: Value) -> Result<ToolOutput, ToolError> {
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs("`command` (string) is required".into()))?;
        let cmd_token = command.split_whitespace().next().unwrap_or("");
        tracing::Span::current().record("cmd", cmd_token);

        let start = Instant::now();
        let out = match parse_chain(command) {
            Ok(chain) => execute(&chain, &self.registry, turn).await,
            Err(e) => CommandOutput::failed(2, parse_error_line(&e).into_bytes()),
        };
        let duration = start.elapsed();
        Ok(tool_output(
            present(out, &self.output_cap, duration),
            duration,
        ))
    }
}

fn parse_error_line(e: &ParseError) -> String {
    let (hint, recovery) = match e {
        ParseError::Empty => (Hint::Use, "run <cmd>"),
        ParseError::UnterminatedQuote => (Hint::Check, "match all \" pairs"),
        ParseError::UnexpectedOperator(_) => (Hint::Try, "put a command before the operator"),
        ParseError::TrailingOperator(_) => (Hint::Try, "add a command after the operator"),
        ParseError::EmptyCommand => (Hint::Try, "add a command between operators"),
        ParseError::Unsupported(_) => (Hint::Use, "bash \"...\" for unsupported shell features"),
        ParseError::UnquotedAlternation => (
            Hint::Use,
            "a quoted ERE pattern, as in `grep \"TODO|FIXME\" FILE`",
        ),
        ParseError::Expansion => (
            Hint::Use,
            "bash \"...\" to expand it, quotes to keep it literal, or ~ for $HOME",
        ),
        ParseError::Newline => (
            Hint::Use,
            "';' between commands on one line, quotes around multi-line text, \
             or bash \"...\" for a script",
        ),
        ParseError::Comment => (
            Hint::Try,
            "dropping the comment, or quoting the word to keep a literal '#'",
        ),
        ParseError::EscapedSpace => (Hint::Use, "quotes around the argument, as in \"a b\""),
        ParseError::TooManyOperators => (
            Hint::Try,
            "splitting it into several run calls, or bash \"...\" for a script",
        ),
        ParseError::Redirection(r) => match r {
            Redirection::Output | Redirection::Append => {
                (Hint::Use, "a pipe into write, as in `<cmd> | write PATH`")
            }
            Redirection::Input => (Hint::Use, "a pipe, as in `cat FILE | <cmd>`"),
            Redirection::HereDoc => (Hint::Use, "a pipe, as in `echo TEXT | <cmd>`"),
            Redirection::Stderr => (
                Hint::Try,
                "dropping it — stderr is already in this result — or bash \"...\" to reshape it",
            ),
        },
    };
    error_line("parse", e, hint, recovery)
}

fn tool_output(presented: PresentResult, duration: Duration) -> ToolOutput {
    ToolOutput {
        output: presented.output,
        exit_code: presented.exit_code,
        duration,
        truncated: presented.truncated,
        overflow_file: presented.overflow_file,
        attachments: presented.attachments,
        streams: Some(RawStreams {
            stdout: presented.stdout_raw,
            stderr: presented.stderr_raw,
            stderr_overflow_file: presented.stderr_overflow_file,
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use base64::Engine;
    use base64::engine::general_purpose::STANDARD as B64;
    use regex::Regex;
    use tempfile::{TempDir, tempdir};

    use super::*;
    use crate::command::{Command, CommandError, CommandInput, Usage};
    use crate::commands::{register_test_readers, test_see};
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
        fn usage(&self) -> Usage {
            Usage::new("", "")
        }
        async fn run(&self, _input: CommandInput) -> Result<CommandOutput, CommandError> {
            let mut stdout = Vec::with_capacity(self.0 * 8);
            for i in 1..=self.0 {
                stdout.extend_from_slice(format!("line {i}\n").as_bytes());
            }
            Ok(CommandOutput::ok(stdout))
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
        fn usage(&self) -> Usage {
            Usage::new("", "")
        }
        async fn run(&self, input: CommandInput) -> Result<CommandOutput, CommandError> {
            Ok(CommandOutput::ok(
                format!("{}\n", input.stdin.map_or(0, |s| s.len())).into_bytes(),
            ))
        }
    }

    fn registry() -> Arc<CommandRegistry> {
        let mut reg = CommandRegistry::new();
        register_test_readers(&mut reg);
        reg.register(test_see());
        Arc::new(reg)
    }

    fn full_registry() -> Arc<CommandRegistry> {
        Arc::new(crate::commands::test_registry())
    }

    fn cap_in(dir: &Path) -> Arc<OutputCap> {
        Arc::new(OutputCap::new(PresentSpec {
            overflow_dir: dir.to_path_buf(),
            ..PresentSpec::default()
        }))
    }

    fn tool_with_dir(dir: &Path) -> RunTool {
        RunTool::new(registry(), cap_in(dir))
    }

    fn tool_with(dir: &Path, reg: Arc<CommandRegistry>) -> RunTool {
        RunTool::new(reg, cap_in(dir))
    }

    fn fresh_dir() -> TempDir {
        tempdir().expect("tempdir")
    }

    /// The result as IPC clients receive it.
    fn invoke(tool: &RunTool, cmd: &str) -> Value {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let output = rt
            .block_on(tool.invoke(&TurnContext::default(), json!({ "command": cmd })))
            .unwrap();
        serde_json::to_value(output.report()).unwrap()
    }

    #[test]
    fn only_a_chain_of_web_commands_leaves_this_machine_unread() {
        let dir = fresh_dir();
        let tool = tool_with(dir.path(), full_registry());
        let reads_host = |command: &str| tool.reads_host(&json!({ "command": command }));

        assert!(!reads_host("web fetch https://example.com/"));
        assert!(!reads_host(
            "web search rust ; web fetch https://example.com/"
        ));
        assert!(reads_host("web fetch https://example.com/ | grep install"));
        assert!(reads_host(
            "cat notes.txt && web fetch https://example.com/"
        ));
        assert!(reads_host("bash \"cat ~/.ssh/id_rsa\""));
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
        assert_eq!(freq["stdout"], "      2 b\n", "{freq}");

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

    /// The next-page hint in the truncation notice must run as printed and
    /// return the lines right after the ones shown.
    #[test]
    fn run_overflow_next_page_hint_runs_as_printed() {
        let mut reg = CommandRegistry::new();
        reg.register(Lines(5000));
        register_test_readers(&mut reg);
        let dir = fresh_dir();
        let tool = tool_with(dir.path(), Arc::new(reg));

        let first = invoke(&tool, "lines");
        let shown = first["stdout"].as_str().unwrap().lines().count();
        let hint = first["output"]
            .as_str()
            .unwrap()
            .lines()
            .find_map(|l| l.strip_prefix("Next page: "))
            .expect("next-page hint in notice");

        let followup = invoke(&tool, hint);
        assert_eq!(followup["exit_code"], 0, "{followup}");
        let expected: String = (shown + 1..=2 * shown)
            .map(|i| format!("line {i}\n"))
            .collect();
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
}
