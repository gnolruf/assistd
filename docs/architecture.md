# Architecture

`assistd` is a Rust workspace split into eleven crates plus a thin
binary. This page maps the crates, the external processes the daemon
supervises, and the path a user query takes from keypress to spoken
reply. Read it once before contributing; the rest of `docs/` assumes
the vocabulary established here.

## Workspace at a glance

```
┌─────────────────────────────────────────────────────────────────────┐
│                              CLIENTS                                │
│   assistd query        assistd chat (TUI)        compositor hooks   │
│   assistd ptt-start    assistd cycle             (i3 / Sway exec)   │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 │  Unix socket  ($XDG_RUNTIME_DIR/assistd.sock)
                 │  line-delimited JSON (Request / Event tagged unions)
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  assistd  (binary; `daemon` subcommand)                             │
│  CLI parsing, subsystem init, lifecycle                             │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  assistd-core                                                       │
│  AppState  ·  Agent (per-turn loop)  ·  socket server  ·  presence  │
└──┬───────────┬─────────────┬──────────────┬─────────────┬───────────┘
   │           │             │              │             │
   ▼           ▼             ▼              ▼             ▼
┌──────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐
│ llm  │  │  tools   │  │  voice   │  │   mcp    │  │   wm    │
│      │  │          │  │          │  │          │  │         │
│chat  │  │ run +    │  │ Whisper  │  │ stdio    │  │ i3 IPC  │
│loop  │  │ commands │  │ + Piper  │  │ + SSE    │  │ + Sway  │
└──┬───┘  └─┬────┬───┘  └────┬─────┘  └────┬─────┘  └────┬────┘
   │        │    │           │             │             │
   │        │    └────► ipc (wire types, shared by clients + daemon)
   │        │
   │        ▼
   │   ┌─────────┐    ┌──────────┐
   │   │ memory  │◄───│  embed   │   (SQLite + embedding queue)
   │   │ SQLite  │    │ HTTP cli │
   │   └─────────┘    └─────┬────┘
   │                        │
   │   config (TOML schema, consumed by every crate above)
   │                        │
   ▼                        ▼
┌──────────────┐   ┌────────────────┐    External processes
│ llama-server │   │ embedding      │    spawned and supervised
│ HTTP :8385   │   │ llama-server   │    by the daemon.
│ /v1/chat/    │   │ HTTP :8386     │
│ completions  │   │ /v1/embeddings │
└──────────────┘   └────────────────┘

┌──────────────┐   ┌────────────────┐    ┌────────────────┐
│ whisper.cpp  │   │ Piper TTS      │    │ MCP servers    │
│ (whisper-rs) │   │ (piper binary) │    │ (user-defined) │
└──────────────┘   └────────────────┘    └────────────────┘
       ▲                   ▲                     ▲
       └─ assistd-voice ───┘                     └─ assistd-mcp
```

## Crate map

| Crate            | Purpose                                                                                              | Depends on                                                                       |
|------------------|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| `assistd`        | Binary. CLI, daemon entry, per-subsystem init wiring.                                                | every `assistd-*` crate (daemon feature)                                         |
| `assistd-config` | TOML schema, defaults, validation. The single source of truth for every tunable.                    | none                                                                             |
| `assistd-core`   | Daemon glue. `AppState`, agent loop, presence machine, socket server, `build_tools()` factory.       | `config`, `ipc`, `llm`, `tools`, `memory`, `embed`, `mcp`, `voice`, `wm`         |
| `assistd-embed`  | Embedding HTTP client + job queue feeding the semantic store.                                        | `config`, `memory`                                                               |
| `assistd-ipc`    | Wire-protocol types (`Request`, `Event`, `PresenceState`, `VoiceCaptureState`, `ImageAttachment`).   | none (intentionally minimal so client-only builds stay small)                    |
| `assistd-llm`    | `LlmBackend` trait + `LlamaChatClient` (HTTP/SSE to llama-server) + child-process supervisor.        | `config`, `tools`                                                                |
| `assistd-mcp`    | MCP client (stdio + SSE) and adapter that exposes discovered MCP tools through the `Tool` trait.     | `tools`                                                                          |
| `assistd-memory` | SQLite-backed persistent stores: `MemoryStore`, `ConversationStore`, `SemanticStore`.                | none                                                                             |
| `assistd-tools`  | `Tool` and `Command` traits, registries, `RunTool`, all built-in commands, policy gates.             | `config`, `embed`, `memory`, `ipc`, `wm`                                         |
| `assistd-voice`  | `VoiceInput` (Whisper STT, VAD continuous mode) + `VoiceOutput` (Piper TTS) + adaptive `SpeakDecision`. | `config`, `ipc`                                                                  |
| `assistd-wm`     | `WindowManager` trait + i3 (`tokio-i3ipc`) and Sway (`swayipc-async`) backends, plus `NoWindowManager`. | none                                                                             |

`config` and `ipc` sit at the bottom because everything depends on
them. `core` sits at the top because it's where every subsystem is
wired into a working daemon. The binary itself is intentionally thin:
parse argv, load config, build `AppState`, hand off.

The same binary also ships several client subcommands that speak the
Unix-socket IPC: `query`, `chat`, `cycle` / `sleep` / `wake` /
`drowse`, `ptt-start` / `ptt-stop`, `listen-*`, `voice-*`, `memory`,
and `tray`. The tray subcommand is a long-lived
[StatusNotifierItem](https://www.freedesktop.org/wiki/Specifications/StatusNotifierItem/)
client (via the [`ksni`](https://crates.io/crates/ksni) crate): it
holds a passive `Request::Subscribe` connection, translates the
broadcast events into icon state (disconnected → generating →
listening → presence), and provides a Sleep / Wake menu that issues
`Request::SetPresence` on isolated one-shot connections. Like every
other client it depends only on `assistd-ipc` + `assistd-config`, so
client-only builds stay daemon-free.

## Subsystem walk-throughs

### LLM lifecycle (`assistd-llm` ↔ llama-server)

The daemon spawns `llama-server` as a child process at startup, with
GPU layer count, KV-cache quantization, and other knobs taken from
`[llama_server]` in the config. `LlamaService` health-probes
`GET /health` until the server reports ready, then `LlamaChatClient`
streams chat completions over `POST /v1/chat/completions`.

If the child crashes mid-stream (CUDA OOM, OOM-killer, segfault), the
supervisor restarts it with exponential backoff and the in-flight
agent turn surfaces an `LlmError::ServerRestarting`. The agent
retries once after the new child reports healthy; further failures
propagate up to the client as an `Event::Error`.

Vision support is detected dynamically: `probe_capabilities()` calls
`GET /props` on the running server to learn whether the model has a
vision projector. The `VisionGate` flips on if so, allowing the `see`
and `screenshot` commands to attach images to the next turn. A
`VisionRevalidator` re-probes at the top of every query so a model
swap (e.g., switching presets via the config and reloading) is
picked up without a daemon restart.

### Agent loop (`assistd-core::Agent`)

One per-query state machine, single-threaded per turn. The loop:

1. `backend.push_user(text, attachments)` — append user message + any
   image attachments to the conversation.
2. `backend.step(tools.openai_schemas(), tx)` — send `messages +
   tools` to llama-server, stream tokens.
3. The response is either text (`StepOutcome::Final` → emit
   `Event::Delta` per token, then `Event::Done`) or tool calls
   (`StepOutcome::ToolCalls` → step 4).
4. For each tool call: `tools.get(name).invoke(arguments)`. Emit
   `Event::ToolCall` before, `Event::ToolResult` after. Push results
   back via `backend.push_tool_results(...)`.
5. Loop back to step 2 until the model emits text or hits the
   per-turn step cap (`agent.max_steps_per_turn` in the config).

The loop does not parallelize tool calls. Tools execute serially, and
their results land in the conversation in the order the model
emitted them. This keeps the conversation deterministic and the
`Event` stream readable.

### Tools (`assistd-tools`)

Two-tier system. The LLM sees a single `run` tool whose argument is a
shell-style command line; that line is parsed into a pipeline AST
and dispatched through an internal `CommandRegistry` of in-process
Rust handlers. So `cat /etc/hosts | grep -v '^#'` is one tool call
that fans out into two in-process commands chained by a byte-level
pipe.

Three additional `Tool`s sit alongside `run`: `remember`, `recall`,
`reminisce`. They don't fit the shell mold, so they're regular
LLM-facing tools with their own JSON schemas.

Built-in commands: `bash`, `cat`, `echo`, `grep`, `ls`, `screenshot`,
`see`, `wc`, `web`, `wm`, `write`. See [tools.md](tools.md) for the
trait definitions and a complete worked example of adding your own.

Policy gates (`ConfirmationGate`, `VisionGate`, `SandboxRequest`)
intercept the dangerous paths: `bash` runs through a sandbox
(bubblewrap by default) with a destructive-pattern denylist; `write`
restricts targets to a configured allowlist; `see` and `screenshot`
no-op when the loaded model has no vision projector.

### Memory (`assistd-memory` + `assistd-embed`)

One SQLite database (default: `~/.local/share/assistd/memory.db`)
holds three schemas: a key/value `MemoryStore` for durable facts, a
`ConversationStore` for full per-session transcripts (with branching
+ undo), and a `SemanticStore` for embedding-indexed chunks.

The `remember` tool writes to all three: it inserts the K/V row,
records the turn, and queues an embedding job on a `tokio::sync::mpsc`
channel. A background task drains the queue, sends batches to the
embedding server (a second, smaller llama-server child process
configured under `[embedding]`), and inserts the resulting vectors
into the semantic store. The `recall` tool runs a cosine-similarity
query against that store; the `reminisce` tool summarizes a
transcript range and saves the summary back as a high-level memory.

### Voice (`assistd-voice`)

Push-to-talk: the daemon receives `Request::PttStart` from a client
or compositor binding, opens the configured microphone via cpal, and
streams audio into a ring buffer. On `PttStop` it stops capture and
hands the buffer to a `Transcriber` (Whisper via `whisper-rs`,
loaded once at startup). Transcripts feed back into the agent loop
as if the user had typed them.

Continuous mode (`MicContinuousListener`) keeps the mic open and uses
WebRTC VAD plus a Whisper-resident silence detector to decide when
to chop the stream into utterances. To avoid GPU thrashing during
chat generation, the `QueuedTranscriber` defers Whisper inference
when the LLM is mid-stream.

TTS: model output flows through a `SentenceBuffer` that segments the
stream into speakable units, each handed to `PiperVoiceOutput`,
which writes audio chunks to the spawned `piper` binary's HTTP
endpoint and plays the returned PCM via rodio. `SpeakDecision`
adapts: short replies are read in full, long replies are summarized
first (configurable in `[voice.synthesis]`).

### Window manager (`assistd-wm`)

Pure abstraction over the compositor IPC. The daemon optionally
queries the active window at the start of each turn (title + class +
geometry) and injects it into the system prompt, so the model knows
what you're looking at. The `wm` command surfaces this same backend
to the model as a tool: `run wm list`, `run wm focus 'firefox'`, and
similar.

If neither i3 nor Sway is reachable (no compositor running, or
running an unsupported one like Hyprland today), `NoWindowManager`
returns convention-compliant errors so commands fail gracefully
instead of panicking.

### MCP (`assistd-mcp`)

External tool servers configured under `[[mcp.servers]]`. Each entry
spawns a transport (a child process for stdio servers, a long-lived
HTTP+SSE connection for remote ones) and runs the MCP handshake.
Discovered tools are wrapped by `McpToolAdapter`, which implements
`Tool` over the discovered schema, and registered into the same
`ToolRegistry` the LLM sees, under the name
`mcp__<server-name>__<tool-name>`.

The supervisor restarts crashed stdio servers with exponential
backoff and re-runs discovery on each restart. SSE servers
auto-reconnect on transport drop. The daemon never blocks on a
slow MCP server: each call has a per-tool timeout that surfaces as
a structured error to the model.

## Data flow: end-to-end query

A walk through `assistd query "what files changed this week?"`:

1. **Client.** `assistd query` constructs `Request::Query { id,
   text, attachments: [], version: Some(1) }`, dials
   `$XDG_RUNTIME_DIR/assistd.sock`, writes the JSON line, half-closes
   the write side, and reads `Event` lines until `Done`.

2. **Socket server** ([`crates/assistd-core/src/socket.rs`](../crates/assistd-core/src/socket.rs)).
   Parses the request, looks up the handler for the variant,
   dispatches under a per-turn `Mutex` so concurrent queries
   serialize.

3. **Agent** ([`crates/assistd-core/src/agent.rs`](../crates/assistd-core/src/agent.rs)).
   The handler builds `Agent::new(...)` from the cached `AppState`
   and calls `run_turn()`. The agent emits `Event::Status { text:
   "Thinking..." }` so the TUI shows immediate feedback.

4. **LLM step** ([`crates/assistd-llm/src/chat/client.rs`](../crates/assistd-llm/src/chat/client.rs)).
   `LlamaChatClient` POSTs `messages + tools` to llama-server with
   `stream: true`. SSE deltas come back; text tokens propagate as
   `Event::Delta`, tool calls accumulate until the stream closes.

5. **Tool dispatch.** llama-server emits a `run` call with arguments
   `{"command":"git log --since='1 week ago' --name-only --pretty="}`.
   The agent emits `Event::ToolCall`, hands the args to `RunTool`,
   which parses the command line and dispatches to `BashCommand`
   (after the destructive-pattern check passes — `git log` is
   read-only).

6. **Sandbox.** `BashCommand` invokes the configured sandbox
   (bubblewrap by default), with the user's writable paths bound
   read-only and `/proc`, `/sys` restricted. The sandboxed `git`
   runs, returns stdout. `RunTool::invoke` truncates if needed
   (per `[tools.output]`), spills overflow to `~/.cache/assistd/
   tools-overflow/` if larger than the inline cap, base64-encodes
   any image attachments, and returns the JSON result.

7. **Loop back.** Result emitted as `Event::ToolResult`, pushed back
   into the conversation, agent calls `step` again. The model now
   has the file list in context and produces a written summary.
   Tokens stream out as `Event::Delta`s, the loop closes with
   `Event::Done`, the socket connection closes.

8. **Client output.** `assistd query` prints deltas as they arrive,
   exits 0 on `Done` or non-zero on `Error`.

For voice queries the only difference is step 1: the request
originates from `assistd ptt-stop` after the daemon ran the
recorded audio through Whisper. Steps 2–7 are identical.

For `assistd chat`, the TUI is the long-lived client: it dials the
socket once at startup, holds it open, and submits one `Request`
per turn over the same connection.

## Where to look next

- [Adding a tool](tools.md) — full code-level walkthrough of
  extending the registry.
- [Sample config](../config/config.sample.toml) — every tunable
  with prose explaining when to change it.
- [i3](wm/i3.md) and [Sway](wm/sway.md) — keybind recipes.
