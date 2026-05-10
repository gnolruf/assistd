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
| **Examples in tree** | `cat`, `ls`, `grep`, `wc`, `bash`, `see`, `screenshot`, `wm`, `web`, `write`. | `remember`, `recall`, `reminisce`, MCP-adapted tools. |

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
2. Edit: `crates/assistd-tools/src/commands/mod.rs` (re-export).
3. Edit: `crates/assistd-core/src/lib.rs` (registration in `build_tools`).
4. Edit: `crates/assistd-tools/src/command.rs` tests (extend the
   convention-compliance and help/summary acceptance tests so the
   new command is covered).

### Step 1 — Implement the command

Create `crates/assistd-tools/src/commands/uppercase.rs`:

```rust
use anyhow::Result;
use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput, error_line};

/// `uppercase`: read bytes from stdin, ASCII-uppercase letters, write to
/// stdout. Non-ASCII bytes pass through untouched (no UTF-8 case
/// folding in v1; users who need that should call `bash 'tr a-z A-Z'`).
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

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        if !input.args.is_empty() {
            return Ok(CommandOutput::failed(
                2,
                error_line(
                    "uppercase",
                    format_args!("unexpected argument: {}", input.args[0]),
                    "Use",
                    "uppercase (no args; pipe input via stdin)",
                )
                .into_bytes(),
            ));
        }
        let mut out = input.stdin;
        out.make_ascii_uppercase();
        Ok(CommandOutput::ok(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn uppercases_ascii_stdin() {
        let out = UppercaseCommand
            .run(CommandInput {
                args: Vec::new(),
                stdin: b"Hello, World!".to_vec(),
            })
            .await
            .unwrap();
        assert_eq!(out.stdout, b"HELLO, WORLD!");
        assert_eq!(out.exit_code, 0);
    }

    #[tokio::test]
    async fn passes_through_non_ascii_bytes() {
        let out = UppercaseCommand
            .run(CommandInput {
                args: Vec::new(),
                stdin: "café".as_bytes().to_vec(),
            })
            .await
            .unwrap();
        // 'c' -> 'C', 'a' -> 'A', 'f' -> 'F'; the é (0xC3 0xA9) is
        // non-ASCII and passes through.
        assert_eq!(out.stdout, "CAFé".as_bytes());
    }

    #[tokio::test]
    async fn arguments_rejected_with_convention_error() {
        let out = UppercaseCommand
            .run(CommandInput {
                args: vec!["FILE".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("[error] uppercase: "), "{stderr}");
        assert!(stderr.contains("Use:"), "{stderr}");
    }
}
```

Three things to notice:

- **`summary()` is ≤80 chars and a verb phrase.** That string lands
  verbatim in the `run` tool's description, which is the only thing
  the LLM sees when choosing whether to call you. Treat it as a
  one-line ad to a model that's read every shell man page.
- **`help()` starts with `usage: <name>`.** The convention-compliance
  test enforces this. The same string is what the executor returns
  on stdout (with `exit_code = 2`) when the command is invoked with
  insufficient args, so the model gets self-service docs in-band.
- **Failure paths use `error_line`.** Every stderr line carries
  `[error] <cmd>: <what>. <Hint>: <recovery>`. That convention is
  what lets the LLM recover in one step rather than guessing again.
  Read [`io_error_nav`](../crates/assistd-tools/src/command.rs) for
  the standard `NotFound` / `PermissionDenied` formatters.

### Step 2 — Re-export from the commands module

Edit [crates/assistd-tools/src/commands/mod.rs](../crates/assistd-tools/src/commands/mod.rs)
to declare and re-export the new module:

```rust
pub mod bash;
pub mod cat;
pub mod echo;
pub mod grep;
pub mod ls;
pub mod screenshot;
pub mod see;
pub mod uppercase;   // <-- new
pub mod wc;
pub mod web;
pub mod wm;
pub mod write;

pub use bash::{BashCommand, BashPolicyCfg};
pub use cat::CatCommand;
pub use echo::EchoCommand;
pub use grep::GrepCommand;
pub use ls::LsCommand;
pub use screenshot::{Backend as ScreenshotBackendKind, ScreenshotCommand, ScreenshotPolicyCfg};
pub use see::SeeCommand;
pub use uppercase::UppercaseCommand;   // <-- new
pub use wc::WcCommand;
pub use web::WebCommand;
pub use wm::WmCommand;
pub use write::{WriteCommand, WritePolicyCfg};
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
commands.register(EchoCommand);
commands.register(UppercaseCommand);   // <-- new
commands.register(WriteCommand::new(write_cfg));
commands.register(SeeCommand::new(vision_gate.clone()));
commands.register(ScreenshotCommand::new(screenshot_cfg, vision_gate));
commands.register(WebCommand::new());
commands.register(BashCommand::new(bash_cfg, sandbox, confirmation_gate));
commands.register(WmCommand::new(window_manager));
```

You'll also need the import at the top of the file. The simple
commands (no constructor args, no policy) are grouped together; add
`UppercaseCommand` there:

```rust
use assistd_tools::commands::{
    BashCommand, BashPolicyCfg, CatCommand, EchoCommand, GrepCommand, LsCommand,
    ScreenshotCommand, ScreenshotPolicyCfg, SeeCommand, UppercaseCommand,
    WcCommand, WebCommand, WmCommand, WriteCommand, WritePolicyCfg,
    Backend as ScreenshotBackendKind,
};
```

That's the entire integration. The command is now visible to the LLM:
its summary appears in the dynamic catalog the `run` tool advertises,
and the parser will dispatch any pipeline stage named `uppercase` to
it. No tool schema work, no IPC changes, no client-side updates.

### Step 4 — Extend the acceptance tests

Two tests in [crates/assistd-tools/src/command.rs](../crates/assistd-tools/src/command.rs)
exhaustively assert that every registered command (a) emits
convention-compliant errors, and (b) has a non-empty `help()` and a
`usage:`-prefixed help block. They run inside the workspace so adding
a command without updating them will fail CI.

Add a row to `every_registered_command_emits_convention_compliant_error`:

```rust
(
    "uppercase",
    rt.block_on(run_cmd(
        UppercaseCommand,
        vec!["unexpected".into()],
    )),
),
```

And register `UppercaseCommand` in the body of
`every_registered_command_has_nonempty_help_and_summary`, bumping the
expected `reg.len()` assertion from 11 to 12.

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
    async fn invoke(&self, args: Value) -> Result<Value>;
}
```

A minimal sketch — a `wait` tool that sleeps for a configurable
duration and returns the elapsed time:

```rust
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::time::Duration;
use assistd_tools::Tool;

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

    async fn invoke(&self, args: Value) -> Result<Value> {
        let ms = args.get("ms")
            .and_then(Value::as_u64)
            .context("`ms` is required and must be a non-negative integer")?;
        let started = std::time::Instant::now();
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
- **Error handling.** Return `Err` for catastrophic failures the
  model can't recover from. For predictable failures (bad input,
  remote service down), return `Ok(json!({ "error": "...", "hint":
  "..." }))` so the model sees the structured failure and can
  retry. The same navigation philosophy as `Command`'s stderr
  convention applies.
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
- **Hint vocabulary.** `Use:` / `Try:` for actionable alternatives;
  `Check:` / `Available:` for diagnostics. The
  acceptance test scans for these four exact prefixes.
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
  — the pipeline parser and executor your `Command` plugs into.
- [`crates/assistd-tools/src/memory_tools.rs`](../crates/assistd-tools/src/memory_tools.rs)
  — three real `Tool` implementations to study before you write one.
