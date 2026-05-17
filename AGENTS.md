# AGENTS.md

Guidelines for AI agents (and humans) working in this repository.
`assistd` is a Rust workspace; treat these rules as load-bearing.

## Core principles

- **Idiomatic Rust always.** Prefer borrowing over cloning, iterators
  over index loops, `?` over `match` on `Result`, `Option`/`Result`
  combinators over nested conditionals, and typed errors
  (`thiserror`) at crate boundaries with `anyhow` reserved for the
  binary edge. Use `tokio`'s structured concurrency primitives; do
  not spawn unsupervised tasks. Take the time to model state with the
  type system — make invalid states unrepresentable rather than
  validating at runtime.
- **No `unsafe`.** The workspace denies `unsafe_code` at the lint
  level. Adding `#[allow(unsafe_code)]` to a module requires an
  extreme, documented justification (FFI with no safe wrapper
  available, etc.) and should be raised in discussion before being
  written. There is essentially never a good reason in this
  codebase; find another way.
- **No other clippy allows.** Do not silence clippy with
  `#[allow(clippy::...)]`, `#![allow(...)]`, or `expect(...)`
  attributes. If clippy fires, fix the code. The only acceptable
  `allow` in the entire workspace is the `unsafe_code` carve-out
  above (and even that should be avoided).
- **Comment sparingly.** Names and structure carry meaning; comments
  rot. Only document:
  - `pub` functions, `pub` items inside a public `impl`, and `pub`
    structs/enums/traits (a short doc comment explaining purpose,
    not restating the signature).
  - Private functions whose logic genuinely cannot be understood by
    reading them — a subtle invariant, a workaround for an upstream
    bug, a non-obvious ordering requirement. This should be rare.
  Do *not* annotate trivial code, narrate what the next line does,
  reference the task or PR that introduced the code, or leave
  "removed X" tombstones. If a comment only restates the code,
  delete it.
- **Stay minimal.** Don't add features, abstractions, or
  configuration knobs the task didn't ask for. A bug fix is a bug
  fix; bundling unrelated cleanup makes review harder. Three
  similar lines is fine — don't reach for a trait until the third
  use site actually exists.
- **Trust internal boundaries.** Validate at system edges (config
  load, IPC parse, tool arguments from the model, external HTTP).
  Inside the workspace, trust types and invariants — don't add
  defensive checks for conditions the type system already rules
  out.

## Required checks before declaring work done

Any change to Rust code — new logic, bug fix, refactor, even a test
update — must pass these in order. Run them from the workspace root.

1. **Format.** `cargo fmt --all` (or `cargo fmt --all -- --check` to
   verify). `rustfmt.toml` pins us to defaults; don't fight it.
2. **Lint.** `cargo clippy --workspace --all-targets --
   -D warnings`. Treat every warning as an error. The workspace
   already warns on `clippy::all` and `rust_2018_idioms` and denies
   `dbg_macro` — leftover `dbg!` will fail the build.
3. **Targeted tests.** Run the test suite for whichever crate(s)
   you touched: `cargo test -p assistd-<crate>`. For test fixes
   specifically, also run the originally failing test in isolation
   with `--nocapture` to confirm the fix is real, not flaky timing.
4. **Full suite.** `cargo test --workspace`. Some crates have
   feature-gated code paths; if your change touches one, also run
   with the relevant features (e.g. `cargo test -p assistd-voice
   --features test-support`).

If a check fails, fix the underlying issue. Do not paper over with
`#[allow]`, `#[ignore]`, or `--no-verify`. If a test is genuinely
wrong, fix the test and explain why in the commit message.

## Workspace layout

The workspace is split into one binary plus ten library crates.
`assistd-config` and `assistd-ipc` sit at the bottom (no internal
deps); `assistd-core` and the `assistd` binary sit at the top. See
[docs/architecture.md](docs/architecture.md) for the full diagram
and data-flow walkthrough.

| Crate            | Responsibility                                                                                                                  |
|------------------|---------------------------------------------------------------------------------------------------------------------------------|
| `assistd`        | Binary (`assistd` on `$PATH`). CLI parsing (`daemon`, `query`, `chat`, `ptt-*`, `cycle`, `memory …`), subsystem init, lifecycle. Feature-gated into `daemon` / `client` / `chat` so client-only builds stay small. |
| `assistd-config` | TOML schema, defaults, validation. Single source of truth for every tunable; every other crate that needs configuration depends on this one. No internal deps. |
| `assistd-core`   | Daemon glue. Owns `AppState`, the per-turn `Agent` loop, the Unix-socket server, the presence state machine, and the `build_tools()` factory that wires every subsystem together. |
| `assistd-embed`  | Embedding HTTP client + background job queue. Pulls chunks off an mpsc, batches them to the embedding `llama-server`, writes vectors into `assistd-memory`'s semantic store. |
| `assistd-ipc`    | Wire-protocol types (`Request`, `Event`, `PresenceState`, `VoiceCaptureState`, `ImageAttachment`). Intentionally dependency-light so client-only builds don't pull in the daemon graph. |
| `assistd-llm`    | `LlmBackend` trait, `LlamaChatClient` (HTTP/SSE to `llama-server`), and the child-process supervisor with health probes, restart-on-crash, and vision-capability detection. |
| `assistd-mcp`    | MCP client (stdio + SSE transports), supervisor with reconnect/backoff, and `McpToolAdapter` which exposes discovered tools through the `Tool` trait under `mcp__<server>__<tool>`. |
| `assistd-memory` | SQLite-backed persistent stores: `MemoryStore` (K/V facts), `ConversationStore` (transcripts with branching/undo), `SemanticStore` (embedding-indexed chunks). Uses `tokio-rusqlite` + `rusqlite_migration`. |
| `assistd-tools`  | `Tool` and `Command` traits, registries, the single `RunTool` the model sees, all built-in commands (`bash`, `cat`, `echo`, `grep`, `ls`, `screenshot`, `see`, `wc`, `web`, `wm`, `write`), and the policy gates (`ConfirmationGate`, `VisionGate`, `SandboxRequest`). |
| `assistd-voice`  | `VoiceInput` (Whisper STT via `whisper-rs`, push-to-talk and VAD continuous modes), `VoiceOutput` (Piper TTS streamed sentence-by-sentence), adaptive `SpeakDecision`. Feature-gated (`whisper`, `mic`, `listen`, `tts`, `cuda`). |
| `assistd-wm`     | `WindowManager` trait with i3 (`tokio-i3ipc`) and Sway (`swayipc-async`) backends, plus `NoWindowManager` fallback. Backs both the system-prompt active-window injection and the `wm` command. Feature-gated per compositor. |

**Dependency direction.** Crates lower in the table do not depend on
crates higher up. If you find yourself wanting `assistd-memory` to
call into `assistd-core`, you're holding it wrong — invert the
dependency or move the type. Keep `assistd-config` and `assistd-ipc`
free of internal deps; they're the foundation everything else builds
on (and `assistd-ipc` is shipped to client-only builds).

## Working in this codebase

- **Use the workspace deps table.** Pull versions through
  `workspace.dependencies` in the root `Cargo.toml` — do not pin
  versions per-crate. If you need a new third-party crate, add it
  there first.
- **Respect feature flags.** Crates like `assistd-voice` and
  `assistd-wm` gate large dep trees behind features. Don't promote
  a feature-gated import into an unconditional dependency to
  silence a build error — fix the feature wiring.
- **External processes are supervised, not assumed.** `llama-server`,
  the embedding `llama-server`, `piper`, `whisper`, and MCP servers
  are all child processes with supervisors that handle restart and
  reconnect. New external integrations should follow the same
  pattern (health probe, exponential backoff, surface failures as
  typed errors), not assume the process is up.
- **The TUI is a daemon client.** `assistd chat` is a long-lived
  client onto the daemon's Unix socket. The TUI should never spawn
  its own LLM/voice/MCP services; it talks IPC and renders events.
- **`config.toml` is canonical.** The annotated
  [config/config.sample.toml](config/config.sample.toml) documents
  every tunable. If you add a new option, update the schema in
  `assistd-config` *and* the sample.

## What not to do

- Add a `#[allow(...)]` to make clippy quiet.
- Write `unsafe` blocks.
- Leave `dbg!`, `println!`, or stray `eprintln!` in committed code
  (use `tracing` macros at the appropriate level).
- Bypass `cargo fmt` / `cargo clippy` / `cargo test` and ship
  anyway.
- Add cross-crate dependencies that flow upward (lower-numbered
  crates depending on higher-numbered ones in the table above).
- Sprinkle comments to explain code that explains itself; document
  *why*, not *what*, and only when *why* is non-obvious.
- Expand scope. Fix the thing in front of you; open a follow-up
  for the surrounding cleanup you noticed.
