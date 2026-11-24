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
  available, etc.), stated in the attribute's `reason = "..."`, and
  should be raised in discussion before being written. There is
  essentially never a good reason in this codebase; find another way.
- **No other clippy allows.** Do not silence clippy with
  `#[allow(clippy::...)]`, `#![allow(...)]`, or `expect(...)`
  attributes. If clippy fires, fix the code. The only acceptable
  `allow` in the entire workspace is the `unsafe_code` carve-out
  above (and even that should be avoided).
- **No comments inside function bodies.** A body that needs a
  comment has a naming or structure problem: rename the variable, or
  extract the step into a private helper whose name says what it
  does. The only exception is a one-line note on an upstream bug
  workaround.
- **Doc comments are one to three lines.** Document `pub` items with
  what they are or do, plus invariants and error conditions when
  they exist. Private items get no doc unless the logic hides a
  subtle invariant. Never write about history, regressions, bugs
  that could happen, who calls the item, how another crate uses it,
  or which ticket produced it. Never cite another file by path. If a
  doc restates the name or signature, delete it. Module `//!` docs
  are one or two sentences; the crate map lives in
  `docs/architecture.md`.
- **Config docs live in the schema.** Field docs in `assistd-config`
  are the technical reference: units, valid range, and what the
  boundary values mean. `config/config.sample.toml` gives each key
  one short line at most; it is not a manual.
- **Lay files out in one order.** Module doc, then every `use`
  (grouped std, external crates, workspace crates, `crate`/`super`),
  then `mod` declarations and re-exports, then consts, then types
  each followed by their impls, then free functions, then
  `#[cfg(test)] mod tests`. No `use` between items or inside
  function bodies.
- **Keep functions short.** Aim for under 60 lines. Past 100, split
  the function into named steps; clippy enforces the limit, so split
  long test-case tables by topic too.
- **Name things for what they are.** A reader should know what a
  function, type, or variable is for from its name alone. No vague
  verbs without an object (`handle`, `process`, `do_work`), no
  single-letter names outside short closures and loop indices, no
  abbreviations the codebase doesn't already use.
- **Test only what would hurt to break.** Don't add tests by
  default. A test is worth adding when it pins behaviour that is
  easy to get wrong and expensive to lose: a security or
  confirmation boundary, a parser or wire format, an invariant the
  type system can't hold, or a bug that actually shipped. Skip
  tests that restate the implementation, exercise std or a
  dependency, check getters, defaults, `Display` strings or log
  text, or enumerate edge cases nobody has hit. One test that
  covers the behaviour beats five that each poke one branch. Don't
  add test-only constructors, variants, parameters, or `cfg`'d
  branches to production code to make something testable; if it
  can't be tested through its real interface, don't test it.
- **Test doubles get a module of their own.** A fake, stub, or
  fixture goes in a module that holds nothing else: a `#[cfg(test)]`
  module when one crate's tests use it, a module behind the crate's
  `test-support` feature when another crate's tests do. A fake
  server is a `src/bin` target with `required-features`. A double
  is a type or a free function; it adds no variant, field, method,
  or trait impl to a production type. `test-support` gates those
  modules, their re-exports, and integration tests kept out of a
  plain `cargo test`; nothing else.
- **Tests don't need the sandbox to run.** CI has no bwrap, and
  the sandbox has no unwrapped mode. Test process handling on
  `exec::supervise` with a plain command. Test policy with
  `inert_sandbox`, whose programs do not exist: a script the policy
  lets through fails to start and exits 127. A test that must run
  a real program in the sandbox probes for bwrap and returns early
  without it; keep those few.
- **Unit tests live inline.** Put them in a
  `#[cfg(test)] mod tests { ... }` block at the end of the source
  file. A separate `tests.rs` (via `#[cfg(test)] mod tests;`) is
  allowed only where the module already has a directory for real
  submodules, or at the crate root; never create a directory just
  to hold tests.
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
`$FEATURES` below is `assistd-llm/test-support,assistd-mcp/test-support,assistd-voice/test-support`,
which CI enables so test-only code is linted and run too.

1. **Format.** `cargo fmt --all` (or `cargo fmt --all -- --check` to
   verify). `rustfmt.toml` pins us to defaults; don't fight it.
2. **Lint.** `cargo clippy --workspace --all-targets --features
   $FEATURES -- -D warnings`, then `cargo clippy -p assistd-cli
   --all-targets --no-default-features -- -D warnings`. Treat every
   warning as an error. The workspace already warns on `clippy::all`
   and `rust_2018_idioms` and denies `dbg_macro` — leftover `dbg!`
   will fail the build.
3. **Docs and deps.** `RUSTDOCFLAGS="-D warnings" cargo doc
   --workspace --no-deps --features $FEATURES`; public docs must not
   link to private items. If you touched a `Cargo.toml`, also run
   `cargo machete`.
4. **Targeted tests.** Run the existing test suite for whichever
   crate(s) you touched: `cargo test -p assistd-<crate>`. Running
   tests is required; writing new ones is not. For test fixes
   specifically, also run the originally failing test in isolation
   with `--nocapture` to confirm the fix is real, not flaky timing.
5. **Full suite.** `cargo test --workspace --features $FEATURES`.

If a check fails, fix the underlying issue. Do not paper over with
`#[allow]`, `#[ignore]`, or `--no-verify`. If a test is genuinely
wrong, fix the test and explain why in the commit message.

## Workspace layout

The workspace is split into one binary plus fourteen library crates.
`assistd-utils` sits at the very bottom (no internal deps), with
`assistd-config` and `assistd-ipc` just above it; `assistd-core` and
the `assistd` binary sit at the top. See
[docs/architecture.md](docs/architecture.md) for the full diagram
and data-flow walkthrough.

| Crate            | Responsibility                                                                                                                  |
|------------------|---------------------------------------------------------------------------------------------------------------------------------|
| `assistd`        | Binary (`assistd` on `$PATH`; package `assistd-cli`). CLI parsing (`daemon`, `chat`, `settings`, `ptt-*`, `search`, `memory …`, `documents …`), subsystem init, lifecycle. The daemon and IPC client subcommands are always built; `chat` (which pulls in `assistd-chat`), `tray`, and the compositor backends (`i3`, `sway`) are optional features; `settings` ships with `chat`. |
| `assistd-chat`   | The `assistd chat` terminal window and the `assistd settings` panel: a ratatui client that renders the events it reads off the socket. Depends only on `assistd-config`, `assistd-ipc` and `assistd-utils`, so it cannot reach a daemon-side crate. An entry-point crate like the binary, so its two `run` functions return `anyhow` errors. |
| `assistd-config` | TOML schema, defaults, validation. Single source of truth for every tunable, including each key's settings-panel label, description, advanced flag, consumer (the program that reads it) and kind (taken from the field's type), declared per table with `describe_fields!`; every other crate that needs configuration depends on this one. Depends only on `assistd-utils`. |
| `assistd-core`   | Daemon glue. Owns `AppState`, the per-turn `Agent` loop, the Unix-socket server, admission (the one queue that grants every model lane to user turns, prompt refines, task runs and session titles, the `Activity` snapshot every scheduling rule is read from, and the permits that say when each class of background work, document chunk embedding and page transcription, may run), the presence state machine, the policy task that drives it from idle time and GPU contention, the loan of the drowsy llama-server to the OCR model that transcribes PDF pages without text, the wake-word listen dispatcher, the voice confirmation gate that asks a headless turn's prompts out loud, the extraction worker's sandboxed launch, the index passes that walk `documents.roots` (with the `ignore` crate) into the document index, the `notify` watches that have changed paths indexed between passes, and the `build_tools()` factory that wires every subsystem together. |
| `assistd-documents` | Document text extraction in a child process: the line-delimited JSON protocol, the worker side the hidden `assistd extract-worker` subcommand runs (plain text, Markdown and source code decoded with `encoding_rs`, HTML rendered with `html2text`, PDF converted page by page with `pdf_oxide`, which also renders a page to a PNG for transcription, DOCX, PPTX and XLSX converted with `office_oxide`, ODT, ODS, ODP and EPUB read from their zip archives with `zip` and `quick-xml` under limits on what an archive may expand to, anything else skipped with a typed reason, text capped), and the daemon-side `Extractor` handle and supervisor. Depends only on `assistd-utils`. |
| `assistd-embed`  | Embedding HTTP client + the worker that drains four backlogs on `assistd-utils`'s `backlog` loop: the message chunks, memories, access chunks and document chunks the semantic store lists as missing a vector, in that order, document chunks only while the permit it is handed is open. Batches them to the embedding `llama-server`, writes vectors into `assistd-memory`'s semantic store. |
| `assistd-ipc`    | Wire-protocol types (`Hello`, `Request`, `Event`, `PresenceState`, `VoiceCaptureState`, `ImageAttachment`, and `SearchKind` and `SearchPlace`, the kinds a search ranks and where a hit's passage is), one module per family and all re-exported at the root, plus what the daemon and its clients must agree on beside the wire: the in-memory image `Attachment` with its format check and capped file loader, `Access` (one thing a tool call reports it touched), and the socket path with its owner-only directory and lock files. Intentionally dependency-light; depends only on `assistd-utils`. The async socket client (`IpcClient`), which refuses a daemon from another version, sits behind the `client` feature, which the binary and `assistd-chat` enable. |
| `assistd-llm`    | `LlmBackend` trait and `TurnBackend`, the part of it an agent turn drives and wrappers implement, `LlamaClient` (HTTP/SSE to `llama-server`), the router-mode launch spec run by the shared child-server supervisor with the presets file that sizes each model's context, the HTTP control plane, vision-capability detection, and `PageTranscriber`, the one-request-per-page client of the OCR model. |
| `assistd-mcp`    | MCP client for stdio and Streamable HTTP servers on top of `rmcp`: stdio servers spawned in `ServerSandbox`, restart on the next call with backoff, OAuth 2.1 bearer tokens kept in an owner-only token file, and `McpToolAdapter` which exposes discovered tools through the `Tool` trait under `mcp__<server>__<tool>`. |
| `assistd-memory` | SQLite-backed persistent stores over three files. `memory.db` holds what exists nowhere else: `MemoryStore` (K/V facts), `TranscriptStore` (transcripts with branching/undo), `TaskStore` (saved tasks and their runs). `index.db` and `vectors.db` hold what the daemon can produce again and replaces when it cannot read it, the second the vectors, kept under the hash of their text, and the text the OCR model read from scanned pages, kept under the hash of their file, so both outlast a rebuilt `index.db`: `SemanticStore` (the chunks cut from messages, memories, documents and the files and pages tool calls touched, indexed by embedding and by keyword, behind the `RecallStore`, `EmbeddingQueue` and `DocumentStore` traits its consumers each take one of; the last two also list the rows the embedding and page-transcription backlogs drain and signal the write that adds one). `MessageChunker` is the backlog that cuts saved messages and their accesses into chunks, the checks run at start keep the index's message, memory and access chunks true to `memory.db`, and a vector no chunk holds the text of is removed once every chunk of its kind is cut. Uses `tokio-rusqlite` + `rusqlite_migration`. |
| `assistd-search` | SearXNG launch spec: the `ChildServerSpec` for the daemon-supervised search instance, run with a generated settings file, an environment cleared down to an allowlist plus the `SEARXNG_*` variables the daemon sets, and a per-launch secret; plus `SearxngClient`, a typed client for its JSON search API. |
| `assistd-tools`  | `Tool` and `Command` traits, registries, the single `RunTool` the model sees, all built-in commands (`bash`, `cat`, `docs`, `echo`, `grep`, `head`, `ls`, `screenshot`, `search`, `see`, `sort`, `tail`, `uniq`, `wc`, `web`, `wm`, `write`), the policy gates (`ConfirmationGate`, `VoiceGate`, `VisionGate`, `SandboxRequest`), `ReaderSandbox`, the read-only, idle-priority bwrap profile the extraction worker runs in, and `ServerSandbox`, the bwrap profile a stdio MCP server runs in: a home of its own, the host paths its entry grants, and none of the daemon's files. |
| `assistd-utils`  | Helpers every other crate may use: exponential backoff and rolling-window `RestartPolicy`, the `backlog` loop (`Backlog`, `drain`, `keep_drained`) that drains stored rows lacking their result a page at a time, each page waiting for the backlog's permit, `inherited_env` (the allowlist of environment variables a child process may inherit), tilde expansion, `human_size`, Hugging Face file IDs, binary-content sniffing, capped child-output line forwarding, size-capped HTTP body reads, public-address classification and the `FilteredResolver` that drops non-public DNS answers, `find_program` (a `PATH` lookup that says why a program cannot be run), the chat-template token list and its neutraliser for text and JSON, the escapes that make daemon-supplied text safe to print on a terminal, the `ProcessGroup` kill-on-drop guard, `ReadinessCell` for subsystems that start in the background, and the `ChildServer` supervisor (spawn, `/health` poll checked against `/proc` listener ownership, restart, degrade) that both llama-servers run under, and the `pipe_worker` supervisor (spawn, capped line reads, per-call deadlines, restart) that the extraction and voice workers run under. No internal deps. |
| `assistd-voice`  | `VoiceInput` (Parakeet STT and Silero VAD through `sherpa-onnx`, push-to-talk and VAD continuous modes), `VoiceOutput` (sherpa-onnx TTS streamed sentence-by-sentence, Piper voices phonemized by spawning `espeak-ng`), adaptive `SpeakDecision`, the Hugging Face downloader that caches their models, plus the voice worker: the protocol, the worker side the hidden `assistd voice-worker` subcommand runs, and the daemon-side proxies and supervisor. |
| `assistd-wm`     | `WindowManager` trait with i3 (`tokio-i3ipc`) and Sway (`swayipc-async`) backends, plus `NoWindowManager` fallback. Backs both the system-prompt active-window injection and the `wm` command, including `wm open`'s launches through the compositor. Feature-gated per compositor. |

**Dependency direction.** Dependencies point one way through these
layers, each crate taking only crates from layers before its own:
`utils`; then `config` and `ipc`; then `chat`, `llm`, `memory`,
`search`, `voice` and `wm`; then `documents` and `embed`; then `tools`;
then `mcp`; then `core`; then the binary. If you find yourself wanting
`assistd-memory` to call into `assistd-core`, you're holding it wrong
— invert the dependency or move the type. `assistd-utils` has no
internal deps and is the only one `assistd-config` and `assistd-ipc`
may take; they're the foundation everything else builds on.
`assistd-llm` does not depend on `assistd-tools`, nor the other way
round: what both need lives in `assistd-ipc`, and `assistd-core`
joins them.

**Shared logic goes in `assistd-utils`.** When the same function or
type would otherwise be written in two crates, put it in
`assistd-utils` instead of copying it. A helper used in one place
stays where it is used.

## Working in this codebase

- **Use the workspace deps table.** Pull versions through
  `workspace.dependencies` in the root `Cargo.toml` — do not pin
  versions per-crate. If you need a new third-party crate, add it
  there first.
- **Respect feature flags.** Crates like `assistd-wm` gate large
  dep trees behind features. Don't promote a feature-gated import
  into an unconditional dependency to silence a build error — fix
  the feature wiring.
- **External processes are supervised, not assumed.** `llama-server`,
  the embedding `llama-server`, SearXNG, MCP servers, the voice worker and
  the extraction worker are all child processes with supervisors that
  handle restart and reconnect. New external integrations should
  follow the same pattern (health probe, exponential backoff, surface
  failures as typed errors), not assume the process is up.
- **Waiting work goes through one of two mechanisms.** Anything that
  sends requests to the main model asks admission: it waits in the
  `TaskQueue` as an entry and holds a `LaneLease` while it decodes.
  Background work is a `Backlog` on `assistd-utils`'s shared loop: the
  stored rows that lack their result, woken by the write that adds
  one, with a `Work` permit when it would compete with the user. Don't
  add a job channel, a timer loop, or a second lock around the model.
  [docs/architecture.md](docs/architecture.md#who-runs-when) says
  which one a new kind of work needs.
- **The TUI is a daemon client.** `assistd chat` is a long-lived
  client onto the daemon's Unix socket. The TUI should never spawn
  its own LLM/voice/MCP services; it talks IPC and renders events.
  `assistd-chat` takes no dependency on a daemon-side crate; a type
  it needs from one belongs on the wire in `assistd-ipc`.
- **`config.toml` is canonical.** The annotated
  [config/config.sample.toml](config/config.sample.toml) documents
  every tunable. If you add a new option, update the schema in
  `assistd-config` (with its `describe_fields!` entry), the sample, and the
  settings panel's catalog, which places the key in a section.

## What not to do

- Add a `#[allow(...)]` to make clippy quiet.
- Write `unsafe` blocks.
- Leave `dbg!`, `println!`, or stray `eprintln!` in committed code
  (use `tracing` macros at the appropriate level).
- Bypass `cargo fmt` / `cargo clippy` / `cargo test` and ship
  anyway.
- Add cross-crate dependencies that flow upward (a crate depending
  on one from its own layer or a later one, as listed under
  "Dependency direction").
- Sprinkle comments to explain code that explains itself; document
  *why*, not *what*, and only when *why* is non-obvious.
- Expand scope. Fix the thing in front of you; open a follow-up
  for the surrounding cleanup you noticed.
