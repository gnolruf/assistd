//! The single LLM-facing tool: `run`. Parses a command line, executes
//! the chain over raw bytes, and hands only the final output to
//! [`crate::presentation`] for truncation, so `cat bigfile | grep foo`
//! feeds grep the whole file.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Instant;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use serde_json::{Value, json};

use crate::Tool;
use crate::chain::{ParseError, Redirection, execute, parse_chain};
use crate::command::{Attachment, CommandOutput, CommandRegistry, error_line};
use crate::presentation::{PresentResult, PresentSpec, present};
use assistd_config::ToolsOutputConfig;
#[cfg(test)]
use assistd_config::defaults::nz32;
use std::path::PathBuf;

/// The single LLM-facing `run` tool. Dispatches a command-line string
/// through the chain parser and executor, then presents the result.
pub struct RunTool {
    registry: Arc<CommandRegistry>,
    spec: PresentSpec,
    counter: Arc<AtomicU64>,
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
        let description = build_description(&registry);
        Self {
            registry,
            spec,
            counter: Arc::new(AtomicU64::new(0)),
            description,
        }
    }
}

fn build_description(registry: &CommandRegistry) -> String {
    let mut s = String::with_capacity(1024);
    s.push_str(
        "Execute a shell-style command in the daemon's working directory. \
         Supports pipelines (|), and/or (&&, ||), sequencing (;), `~` and \
         globs (*, ?, []) on unquoted arguments; quote an argument to pass \
         it through literally, including a `|` that belongs to the \
         argument rather than the pipeline (`grep \"a|b\" f.txt`). Redirections (>, <), env expansion ($VAR), \
         and backgrounding (&) are NOT supported; use `bash \"…\"` for a \
         real shell when needed. \
         Large outputs are truncated; the truncation notice includes a \
         `Full output: /tmp/assistd-output/cmd-N.txt` path that subsequent \
         `run` calls can grep/cat to read the full content.\n\n\
         Available commands:\n",
    );
    let pairs = registry.sorted_summaries();
    let name_width = pairs.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
    for (name, summary) in pairs {
        s.push_str(&format!("  {name:<name_width$}: {summary}\n"));
    }
    s.push_str(
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

    #[tracing::instrument(skip(self, args), fields(cmd = tracing::field::Empty))]
    async fn invoke(&self, args: Value) -> Result<Value> {
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("`command` (string) is required"))?;
        let cmd_token = command.split_whitespace().next().unwrap_or("");
        tracing::Span::current().record("cmd", cmd_token);

        let start = Instant::now();
        let out = match parse_chain(command) {
            Ok(chain) => execute(&chain, &self.registry, None).await?,
            Err(e) => CommandOutput::failed(2, parse_error_line(&e).into_bytes()),
        };
        let r = present(out, &self.spec, &self.counter, start.elapsed());
        Ok(build_result(r))
    }
}

fn parse_error_line(e: &ParseError) -> String {
    let (hint, recovery) = match e {
        ParseError::Empty => ("Use", "run <cmd>"),
        ParseError::UnterminatedQuote => ("Check", "match all \" pairs"),
        ParseError::UnexpectedOperator(_) => ("Try", "put a command before the operator"),
        ParseError::TrailingOperator(_) => ("Try", "add a command after the operator"),
        ParseError::EmptyCommand => ("Try", "add a command between operators"),
        ParseError::Unsupported(_) => ("Use", "bash \"...\" for unsupported shell features"),
        ParseError::UnquotedAlternation => (
            "Use",
            "a quoted ERE pattern, as in `grep \"TODO|FIXME\" FILE`",
        ),
        ParseError::Redirection(r) => match r {
            Redirection::Output | Redirection::Append => {
                ("Use", "write PATH, as in `<cmd> | write /tmp/out.txt`")
            }
            Redirection::Input => ("Use", "a pipe, as in `cat FILE | <cmd>`"),
            Redirection::HereDoc => ("Use", "a pipe, as in `echo TEXT | <cmd>`"),
            Redirection::Stderr => (
                "Try",
                "dropping it — stderr is already in this result — or bash \"...\" to reshape it",
            ),
        },
    };
    error_line("parse", e, hint, recovery)
}

fn build_result(r: PresentResult) -> Value {
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
    result
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
mod tests;
