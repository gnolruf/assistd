# Adding a tool

This guide walks through extending `assistd` with a new capability the
LLM can invoke.

If you want the architectural backstory first, read
[architecture.md](architecture.md) — this guide assumes you know what
`assistd-core`, `assistd-tools`, and the agent loop are.

## Pick a path: `Command` or `Tool`

`assistd-tools` exposes two traits. Which one to implement depends on
what you're building.

| | `Command` | `Tool` |
|---|---|---|
| **Defined in** | [`crates/assistd-tools/src/command.rs`](../crates/assistd-tools/src/command.rs) | [`crates/assistd-tools/src/lib.rs`](../crates/assistd-tools/src/lib.rs) |
| **Registered into** | `CommandRegistry` (one per daemon) | `ToolRegistry` (one per daemon) |
| **What the LLM sees** | The catalog summary baked into the `run` tool's description (your command appears as one of the verbs the model can compose with `\|`, `&&`, `;`). | A standalone OpenAI-style tool with its own JSON schema. |
| **I/O shape** | Bytes in (stdin), bytes out (stdout + stderr + `exit_code`), participates in pipes. | JSON in, JSON out. |
| **Use when** | Your capability is shell-shaped: takes args + maybe stdin, produces text or attachments, composes naturally with `cat`, `grep`, `wc`, etc. | Your capability has structured arguments (objects, arrays), is stateful, or doesn't fit a `argv + stdin → stdout` model. |
| **Examples in tree** | `cat`, `ls`, `grep`, `wc`, `head`, `tail`, `sort`, `uniq`, `bash`, `see`, `screenshot`, `wm`, `web`, `write`. | `remember`, `recall`, `reminisce`, MCP-adapted tools. |

**Rule of thumb:** start with `Command`. It's smaller, gets pipe
composition for free, and the LLM is already trained on shell
patterns. Only reach for `Tool` when you genuinely need structured
arguments — for example, "save this fact under this key with this
TTL" — that no shell-shaped surface can cleanly express.

The rest of this guide builds a `Command` end to end, then briefly
shows the `Tool` path.

## Walkthrough: a `uppercase` command

Goal: a small command that reads bytes from stdin and writes them
back with every ASCII letter uppercased. Useful inside pipelines:

```
run "cat README.md | grep -i 'license' | uppercase"
```

We'll touch four files:

1. New: `crates/assistd-tools/src/commands/uppercase.rs`.
2. Edit: `crates/assistd-tools/src/commands/mod.rs` (re-export, plus
   the test-only `test_registry()`).
3. Edit: `crates/assistd-core/src/lib.rs` (registration in `build_tools`).
4. Edit: `crates/assistd-tools/src/command/tests.rs` (extend the
   convention-compliance acceptance test so the new command is
   covered).

### Step 1 — Implement the command

Create `crates/assistd-tools/src/commands/uppercase.rs`:

```rust
use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput, Hint, error_line};

/// `uppercase`: read bytes from stdin, ASCII-uppercase letters, write to
/// stdout. Non-ASCII bytes pass through untouched (no UTF-8 case
/// folding).
pub struct UppercaseCommand;

#[async_trait]
impl Command for UppercaseCommand {
    fn name(&self) -> &str {
        "uppercase"
    }

    fn summary(&self) -> &'static str {
        "ASCII-uppercase stdin; non-ASCII bytes pass through unchanged"
    }

    fn help(&self) -> String {
        "usage: uppercase\n\
         \n\
         Read bytes from stdin and write them to stdout with ASCII \
         letters uppercased. Non-ASCII bytes are emitted unchanged.\n\
         \n\
         Takes no flags or arguments. Reading from a pipe is the only \
         supported input source; calling with arguments is rejected so \
         a stray filename can't be silently misinterpreted as input.\n"
            .to_string()
    }

    async fn run(&self, input: CommandInput) -> CommandOutput {
        if let Some(arg) = input.args.first() {
            return CommandOutput::failed(
                2,
                error_line(
                    "uppercase",
                    format_args!("unexpected argument: {arg}"),
                    Hint::Use,
                    format_args!("cat {arg} | uppercase"),
                ),
            );
        }
        let Some(mut out) = input.stdin else {
            return CommandOutput::usage(self.help());
        };
        out.make_ascii_uppercase();
        CommandOutput::ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn run_uppercase(args: Vec<String>, stdin: Option<&[u8]>) -> CommandOutput {
        UppercaseCommand
            .run(CommandInput {
                args,
                stdin: stdin.map(<[u8]>::to_vec),
            })
            .await
    }

    #[tokio::test]
    async fn uppercases_ascii_stdin() {
        let out = run_uppercase(Vec::new(), Some(b"Hello, World!".as_slice())).await;
        assert_eq!(out.stdout, b"HELLO, WORLD!");
        assert_eq!(out.exit_code, 0);
    }

    #[tokio::test]
    async fn passes_through_non_ascii_bytes() {
        let out = run_uppercase(Vec::new(), Some("café".as_bytes())).await;
        assert_eq!(out.stdout, "CAFé".as_bytes());
    }

    #[tokio::test]
    async fn no_stdin_replies_with_usage() {
        let out = run_uppercase(Vec::new(), None).await;
        assert_eq!(out.exit_code, 2);
        assert!(out.stdout.starts_with(b"usage: uppercase"), "{out:?}");
    }

    #[tokio::test]
    async fn arguments_rejected_with_convention_error() {
        let out = run_uppercase(vec!["FILE".into()], None).await;
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("[error] uppercase: "), "{stderr}");
        assert!(stderr.contains("Use:"), "{stderr}");
    }
}
```

Four things to notice:

- **`summary()` is ≤80 chars and a verb phrase.** That string lands
  verbatim in the `run` tool's description, which is the only thing
  the LLM sees when choosing whether to call you. Treat it as a
  one-line ad to a model that's read every shell man page.
- **`help()` starts with `usage: <name>`.** The help/summary
  acceptance test checks for a `usage:` line. Return it via
  `CommandOutput::usage` (stdout, `exit_code = 2`) when the command is
  invoked with insufficient input, so the model gets self-service docs
  in-band; the executor also answers `<cmd> --help` with it.
- **`run` is infallible.** It returns a `CommandOutput`, never a
  `Result`: every failure is a non-zero `exit_code` plus a stderr
  line, which is what lets `||` and `&&` react to it.
- **Failure paths use `error_line`.** Every stderr line carries
  `[error] <cmd>: <what>. <Hint>: <recovery>`, where `<Hint>` is a
  `Hint` variant. That convention is what lets the LLM recover in one
  step rather than guessing again. `CommandOutput::usage_error` is a
  shorthand for the exit-2, `Hint::Use` case above; read
  [`io_error_nav`](../crates/assistd-tools/src/command.rs) for the
  standard `NotFound` / `PermissionDenied` formatters.

### Step 2 — Re-export from the commands module

Edit [crates/assistd-tools/src/commands/mod.rs](../crates/assistd-tools/src/commands/mod.rs)
to declare and re-export the new module:

```rust
pub mod bash;
pub mod cat;
pub mod echo;
pub mod grep;
pub mod head_tail;
pub mod ls;
pub mod screenshot;
pub mod see;
pub mod sort;
pub mod uniq;
pub mod uppercase;   // <-- new
pub mod wc;
pub mod web;
pub mod wm;
pub mod write;

pub use crate::policy::BashPolicyCfg;
pub use bash::BashCommand;
pub use cat::CatCommand;
pub use echo::EchoCommand;
pub use grep::GrepCommand;
pub use head_tail::{HeadCommand, TailCommand};
pub use ls::LsCommand;
pub use screenshot::{Backend as ScreenshotBackendKind, ScreenshotCommand, ScreenshotPolicyCfg};
pub use see::SeeCommand;
pub use sort::SortCommand;
pub use uniq::UniqCommand;
pub use uppercase::UppercaseCommand;   // <-- new
pub use wc::WcCommand;
pub use web::WebCommand;
pub use wm::WmCommand;
pub use write::{WriteCommand, WritePolicyCfg};
```

The same file holds the test-only `test_registry()` the help/summary
acceptance test iterates over; register `UppercaseCommand` there too:

```rust
r.register(EchoCommand);
r.register(UppercaseCommand);   // <-- new
```

### Step 3 — Register in `build_tools`

The daemon constructs its `CommandRegistry` once at startup, in
`assistd_core::build_tools`. Add the new command alongside the others
in [crates/assistd-core/src/lib.rs](../crates/assistd-core/src/lib.rs):

```rust
let mut commands = CommandRegistry::new();
commands.register(CatCommand);
commands.register(LsCommand);
commands.register(GrepCommand);
commands.register(WcCommand);
commands.register(HeadCommand);
commands.register(TailCommand);
commands.register(SortCommand);
commands.register(UniqCommand);
commands.register(EchoCommand);
commands.register(UppercaseCommand);   // <-- new
commands.register(WriteCommand::new(write_cfg));
commands.register(SeeCommand::new(vision_gate.clone()));
commands.register(ScreenshotCommand::new(screenshot_cfg, vision_gate));
commands.register(WebCommand::new());
commands.register(BashCommand::new(
    bash_cfg.clone(),
    sandbox.clone(),
    confirmation_gate.clone(),
));
commands.register(WmCommand::new(
    window_manager,
    bash_cfg,
    sandbox,
    confirmation_gate,
));
```

You'll also need to add `UppercaseCommand` to the
`assistd_tools::commands::{…}` import at the top of the file.

That's the entire integration. The command is now visible to the LLM:
its summary appears in the dynamic catalog the `run` tool advertises,
and the parser will dispatch any pipeline stage named `uppercase` to
it. No tool schema work, no IPC changes, no client-side updates.

### Step 4 — Extend the acceptance tests

Two tests in [crates/assistd-tools/src/command/tests.rs](../crates/assistd-tools/src/command/tests.rs)
assert that every production command (a) emits convention-compliant
errors, and (b) has a non-empty, ≤80-char `summary()` and a `help()`
containing `usage:`. Neither discovers commands on its own, so a new
command is covered only once you add it.

Add `UppercaseCommand` to the `use crate::commands::{…}` list in
`every_registered_command_emits_convention_compliant_error`, and a row
to its `cases`:

```rust
(
    "uppercase",
    rt.block_on(run_cmd(
        UppercaseCommand,
        vec!["unexpected".into()],
    )),
),
```

`every_registered_command_has_nonempty_help_and_summary` iterates over
`test_registry()`, which you extended in step 2; bump its expected
`reg.len()` assertion from 15 to 16.

### Step 5 — Run it

```sh
cargo test -p assistd-tools
cargo build --release
./target/release/assistd daemon &
./target/release/assistd query \
    "list two files in /tmp using run, then send them through uppercase"
```

The agent should call `run "ls /tmp | head -2 | uppercase"` (or
something close) and you'll see the response stream back with the
uppercased filenames. To inspect the prompt-time tool catalog, point
your editor at `RunTool::new` and watch how `commands.sorted_summaries()`
folds into the description string at construction time — your new
`summary()` is in there.

## When to implement `Tool` directly

Skip `Command` and implement `Tool` when the capability has
structured inputs that can't be cleanly mapped to argv + stdin. The
in-tree examples are the memory tools.

The trait, from
[crates/assistd-tools/src/lib.rs](../crates/assistd-tools/src/lib.rs):

```rust
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> Value;
    async fn invoke(&self, args: Value) -> Result<Value, ToolError>;
}
```

`ToolError` has two variants: `InvalidArgs(String)` for arguments that
violate the schema or its constraints, and `Store` for a failed
memory-store call.

A minimal sketch — a `wait` tool that sleeps for a configurable
duration and returns the elapsed time:

```rust
use assistd_tools::{Tool, ToolError};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::time::{Duration, Instant};

pub struct WaitTool;

#[async_trait]
impl Tool for WaitTool {
    fn name(&self) -> &str { "wait" }

    fn description(&self) -> &str {
        "Sleep for the given number of milliseconds, then return the elapsed \
         duration. Use sparingly; the agent loop is blocked while waiting."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "ms": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 60_000,
                    "description": "Milliseconds to sleep (0..=60000)."
                }
            },
            "required": ["ms"]
        })
    }

    async fn invoke(&self, args: Value) -> Result<Value, ToolError> {
        let ms = args
            .get("ms")
            .and_then(Value::as_u64)
            .filter(|ms| *ms <= 60_000)
            .ok_or_else(|| ToolError::InvalidArgs("`ms` must be an integer in 0..=60000".into()))?;
        let started = Instant::now();
        tokio::time::sleep(Duration::from_millis(ms)).await;
        Ok(json!({ "elapsed_ms": started.elapsed().as_millis() as u64 }))
    }
}
```

Three notes:

- **Schema discipline.** `additionalProperties: false` and explicit
  `required` are non-negotiable: the daemon emits OpenAI-strict
  schemas, and llama-server enforces them via grammar-constrained
  decoding. Loose schemas hurt accuracy.
- **Error handling.** Return `Err(ToolError::InvalidArgs(..))` for
  arguments that violate the schema; the agent loop reports it to the
  model as a failed call. For a failure the model can recover from by
  doing something else (remote service down, nothing found), return
  `Ok(json!({ "error": "...", "hint": "..." }))` so the model sees the
  structured failure and can retry. The same navigation philosophy
  as `Command`'s stderr convention applies.
- **Registration.** Tools register directly into the
  `ToolRegistry` after the `RunTool`. In `build_tools`:

  ```rust
  tools.register(WaitTool);
  ```

  No `CommandRegistry` involvement.

## Conventions reference

A condensed list of the rules every tool/command should obey, with
links to the canonical implementation.

- **Error format.** `[error] <cmd>: <what>. <Hint>: <recovery>` —
  use [`error_line`](../crates/assistd-tools/src/command.rs) and
  [`io_error_nav`](../crates/assistd-tools/src/command.rs).
- **Hint vocabulary.** The `Hint` enum: `Use:` / `Try:` for
  actionable alternatives; `Check:` / `Available:` for diagnostics;
  `Install:` names a package; `Note:` explains a condition. The
  convention acceptance test accepts only the first four, so a
  command's failure path should use one of them.
- **`summary()` budget.** ≤80 characters, no trailing newline,
  starts with a verb. Test enforced.
- **`help()` shape.** First line must contain `usage:`. Test
  enforced.
- **Exit codes.** `0` success, `1` predictable failure with stderr,
  `2` usage error (bad args / unknown flags). Mirrors POSIX
  convention so `&&` and `||` compose the way the model expects.
- **Attachments.** A command that produces an image returns
  `CommandOutput { attachments: vec![Attachment::Image { mime,
  bytes }], .. }`. The chain executor threads attachments through
  pipes untouched, and `RunTool` base64-encodes them into the JSON
  result — so `see photo.png | wc -c` still surfaces the image as
  vision input on the next turn even though `wc` itself just sees
  bytes.
- **Sandbox-eligible work.** Anything that runs untrusted shell or
  touches the filesystem outside an allowlist must go through the
  bash sandbox or the writable-paths check. See
  [`policy`](../crates/assistd-tools/src/policy.rs) for
  `SandboxRequest`, `ConfirmationGate`, and the destructive-pattern
  matcher.
- **Vision dependency.** Tools that produce image attachments must
  consult the [`VisionGate`](../crates/assistd-tools/src/vision.rs)
  and refuse with a navigation hint when the loaded model has no
  vision projector — otherwise the LLM hallucinates that it can see
  what it can't.

## Further reading

- [architecture.md](architecture.md) — where `build_tools` fits into
  daemon startup and the agent loop.
- [`crates/assistd-tools/src/run.rs`](../crates/assistd-tools/src/run.rs)
  — the `RunTool` itself: parsing, presentation, overflow handling.
- [`crates/assistd-tools/src/chain/`](../crates/assistd-tools/src/chain/)
  — the pipeline parser, word expander and executor your `Command`
  plugs into. Note that `args` reach your `run` already tilde- and
  glob-expanded, so one written word can arrive as several arguments;
  quoted words are passed through verbatim.
- [`crates/assistd-tools/src/memory_tools.rs`](../crates/assistd-tools/src/memory_tools.rs)
  — three real `Tool` implementations to study before you write one.
