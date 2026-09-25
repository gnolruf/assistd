//! The LLM-facing `run` tool: parses a command line, executes the chain, and
//! truncates only the final output, so every stage sees its whole input.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Instant;

use assistd_config::ToolsOutputConfig;
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use serde_json::{Value, json};

use crate::chain::{ParseError, Redirection, execute, parse_chain};
use crate::command::{Attachment, CommandOutput, CommandRegistry, Hint, error_line};
use crate::commands::cat::human_size;
use crate::presentation::{PresentResult, PresentSpec, present};
use crate::{Tool, ToolError};

/// The LLM-facing `run` tool, dispatching a command line through the chain
/// parser and executor.
pub struct RunTool {
    registry: Arc<CommandRegistry>,
    spec: PresentSpec,
    overflow_counter: Arc<AtomicU64>,
    description: String,
}

impl RunTool {
    /// `output` provides the truncation caps; `overflow_dir` is where
    /// truncated output is spilled in full.
    pub fn new(
        registry: Arc<CommandRegistry>,
        output: &ToolsOutputConfig,
        overflow_dir: PathBuf,
    ) -> Self {
        let spec = PresentSpec {
            max_lines: output.max_lines.get() as usize,
            max_bytes: output.max_bytes(),
            overflow_dir,
        };
        let description = build_description(&registry, &spec);
        Self {
            registry,
            spec,
            overflow_counter: Arc::new(AtomicU64::new(0)),
            description,
        }
    }
}

fn build_description(registry: &CommandRegistry, spec: &PresentSpec) -> String {
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
         Redirections (>, <, 2>, 2>&1), here-docs, env expansion ($VAR), \
         and backgrounding (&) are NOT supported: stderr is already \
         captured in the result, and `bash \"…\"` gives a real shell when \
         needed. ",
    );
    desc.push_str(&format!(
        "Output is returned whole unless its stdout exceeds {max_lines} \
         lines or {max_size}; only then is it cut to that head and the \
         full text saved, with the truncation notice giving a \
         `Full output: {dir}/cmd-N.txt` path that subsequent `run` calls \
         can grep/cat. No such file exists for output under those \
         limits.\n\nCommands (first word of `command`):\n",
        max_lines = spec.max_lines,
        max_size = human_size(spec.max_bytes),
        dir = spec.overflow_dir.display(),
    ));
    let pairs = registry.sorted_summaries();
    let name_width = pairs.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
    for (name, summary) in pairs {
        desc.push_str(&format!("  {name:<name_width$}: {summary}\n"));
    }
    desc.push_str(
        "\nCall a command with no (or insufficient) arguments to see its \
         usage (exit code 2, stdout); `<cmd> --help` prints the same \
         text and works for commands like `ls` and `echo` whose bare \
         form does real work. Errors in real calls exit with a \
         `[<name>]\\t` stderr prefix, distinct from help on stdout. Each \
         error line follows `[error] <cmd>: <what-went-wrong>. <Hint>: \
         <recovery>` where `<Hint>` is one of `Use:` / `Try:` / `Check:` / \
         `Available:`; the recovery clause is a concrete command or check \
         you can run next.",
    );
    desc
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

    #[tracing::instrument(skip(self, args), fields(cmd = tracing::field::Empty))]
    async fn invoke(&self, args: Value) -> Result<Value, ToolError> {
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs("`command` (string) is required".into()))?;
        let cmd_token = command.split_whitespace().next().unwrap_or("");
        tracing::Span::current().record("cmd", cmd_token);

        let start = Instant::now();
        let out = match parse_chain(command) {
            Ok(chain) => execute(&chain, &self.registry, None).await,
            Err(e) => CommandOutput::failed(2, parse_error_line(&e).into_bytes()),
        };
        let presented = present(out, &self.spec, &self.overflow_counter, start.elapsed());
        Ok(build_result(presented))
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
        ParseError::Redirection(r) => match r {
            Redirection::Output | Redirection::Append => {
                (Hint::Use, "write PATH, as in `<cmd> | write /tmp/out.txt`")
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

fn build_result(presented: PresentResult) -> Value {
    let mut result = json!({
        "output":      presented.output,
        "stdout":      presented.stdout_raw,
        "stderr":      presented.stderr_raw,
        "exit_code":   presented.exit_code,
        "truncated":   presented.truncated,
        "duration_ms": presented.duration_ms,
    });
    if let Some(path) = &presented.overflow_file {
        result["overflow_file"] = json!(path.to_string_lossy());
    }
    if !presented.attachments.is_empty() {
        let rendered: Vec<Value> = presented
            .attachments
            .iter()
            .map(render_attachment)
            .collect();
        result["attachments"] = Value::Array(rendered);
    }
    result
}

fn render_attachment(attachment: &Attachment) -> Value {
    match attachment {
        Attachment::Image { mime, bytes } => json!({
            "type": "image",
            "mime": mime,
            "data": B64.encode(bytes),
        }),
    }
}

#[cfg(test)]
mod tests;
