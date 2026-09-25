# Architecture

`assistd` is a Rust workspace split into ten library crates plus a
thin binary. This page maps the crates, the external processes the daemon
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
└────────────────┬─────────────────────────────────────────┬──────────┘
                 │                                         │
                 ▼                                         │
┌───────────────────────────────────────────────────┐      │
│  assistd-core                                     │      │
│  AppState · Agent (per-turn loop) · socket server │      │
│  · presence                                       │      │
└──┬───────────┬─────────────┬──────────────┬───────┘      │
   │           │             │              │              │
   ▼           ▼             ▼              ▼              ▼
┌──────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ llm  │  │  tools   │  │  voice   │  │    wm    │  │   mcp    │
│      │  │          │  │          │  │          │  │          │
│chat  │  │ run +    │  │ Whisper  │  │ i3 IPC   │  │ stdio    │
│loop  │  │ commands │  │ + Piper  │  │ + Sway   │  │ + SSE    │
└──┬───┘  └─┬────┬───┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
   │        │    │           │             │             │
   │        │    └────► ipc (wire types, shared by clients + daemon)
   │        │
   │        ▼
   │   ┌─────────┐    ┌──────────┐
   │   │ memory  │◄───│  embed   │   (SQLite + embedding queue)
   │   │ SQLite  │    │ HTTP cli │
   │   └─────────┘    └─────┬────┘
   │                        │
   │   config (TOML schema, consumed by most crates above)
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
| `assistd`        | Binary. CLI, daemon entry, per-subsystem init wiring (including MCP).                                | `ipc` always; every other `assistd-*` crate via the `daemon` feature             |
| `assistd-config` | TOML schema, defaults, validation. The single source of truth for every tunable.                    | none                                                                             |
| `assistd-core`   | Daemon glue. `AppState`, agent loop, presence machine, socket server, `build_tools()` factory.       | `config`, `ipc`, `llm`, `tools`, `memory`, `embed`, `voice`, `wm`                |
| `assistd-embed`  | Embedding HTTP client + job queue feeding the semantic store.                                        | `config`, `memory`                                                               |
| `assistd-ipc`    | Wire-protocol types (`Request`, `Event`, `PresenceState`, `VoiceCaptureState`, `ImageAttachment`).   | none (intentionally minimal so client-only builds stay small)                    |
| `assistd-llm`    | `LlmBackend` trait + `LlamaChatClient` (HTTP/SSE to llama-server) + child-process supervisor.        | `config`, `ipc`, `tools`                                                         |
| `assistd-mcp`    | MCP client (stdio + SSE) and adapter that exposes discovered MCP tools through the `Tool` trait.     | `tools`                                                                          |
| `assistd-memory` | SQLite-backed persistent stores: `MemoryStore`, `ConversationStore`, `SemanticStore`.                | none                                                                             |
| `assistd-tools`  | `Tool` and `Command` traits, registries, `RunTool`, all built-in commands, policy gates.             | `config`, `embed`, `memory`, `ipc`, `wm`                                         |
| `assistd-voice`  | `VoiceInput` (Whisper STT, VAD continuous mode) + `VoiceOutput` (Piper TTS) + per-sentence `SpeakDecision`. | `config`, `ipc`                                                                  |
| `assistd-wm`     | `WindowManager` trait + i3 (`tokio-i3ipc`) and Sway (`swayipc-async`) backends, plus `NoWindowManager`. | none                                                                             |

`config` and `ipc` sit at the bottom: they have no internal
dependencies, and most other crates build on them. `core` sits at the top because it's where every subsystem is
wired into a working daemon. The binary itself is intentionally thin:
parse argv, load config, build `AppState`, hand off.

The same binary also ships several client subcommands that speak the
Unix-socket IPC: `query`, `chat`, `cycle` / `sleep` / `wake` /
`drowse`, `ptt-start` / `ptt-stop`, `listen-*`, `voice-*`, `memory`,
and `tray`. The tray subcommand is a long-lived
[StatusNotifierItem](https://www.freedesktop.org/wiki/Specifications/StatusNotifierItem/)
client (via the [`ksni`](https://crates.io/crates/ksni) crate): it
holds a passive `Request::Subscribe` connection, translates the
broadcast events into icon state (config error → disconnected →
generating → listening → presence), and provides a Sleep / Wake menu that issues
`Request::SetPresence` on isolated one-shot connections. The `tray`
feature itself pulls in only `assistd-ipc` + `assistd-config`; the
optional `tray-popup` feature adds `assistd-wm`, `eframe` and `winit`
for the floating reply popup. The `chat` feature, by contrast, enables
the whole `daemon` feature on purpose: when no daemon is listening, the
TUI starts one by re-executing its own binary as `assistd daemon`.

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

Vision support is detected dynamically: `probe_capabilities_routed()` calls
`GET /props` on the running server to learn whether the model has a
vision projector. The `VisionGate` flips on if so, allowing the `see`
and `screenshot` commands to attach images to the next turn. A
`VisionRevalidator` re-probes at the start of the first query after
the weights may have been reloaded (any presence transition, or a
supervisor restart of the child), and again after any failed probe,
so the gate tracks what is actually loaded without probing on every
query.

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
   back via `backend.push_tool_results(...)`, which appends them as
   OpenAI `role: "tool"` messages carrying the id of the call they
   answer. A result carrying an image is the one exception: chat
   templates render image parts only on user turns, so those ride
   back as a user message tagged `[tool:<name>]`.
5. Loop back to step 2 until the model emits text. If the model
   repeats the same call several times in a row, or the turn runs
   past a hard-coded step ceiling, the loop tells the model, in a
   one-shot message at the end of the conversation, to stop calling
   tools and answer from what it has already gathered. The tool schema
   stays in the request so a call made anyway is still parsed (and ends
   the turn) rather than streamed to the user as raw markup.

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

Built-in commands: `bash`, `cat`, `echo`, `grep`, `head`, `ls`,
`screenshot`, `see`, `sort`, `tail`, `uniq`, `wc`, `web`, `wm`,
`write`. See [tools.md](tools.md) for the trait definitions and a
complete worked example of adding your own.

Policy gates (`ConfirmationGate`, `VisionGate`, `SandboxRequest`)
intercept the dangerous paths: a `bash` script runs without asking only
when every program it can run is on `[tools.bash] allowed_programs` and
nothing in it matches a destructive pattern; otherwise the user confirms,
and can "always allow" the programs it named. It then runs in a sandbox
(bubblewrap by default). `wm open`
spawns model-chosen argv and so shares that same `[tools.bash]` policy,
widened only by a bind of `$XDG_RUNTIME_DIR` so a launched GUI
application can reach the compositor, and left running once it survives
a startup probe; `write` restricts targets to a
configured allowlist; `see` and `screenshot` refuse with an error
when the loaded model has no vision projector.

### Memory (`assistd-memory` + `assistd-embed`)

One SQLite database (default: `~/.local/share/assistd/memory.db`)
holds three schemas: a key/value `MemoryStore` for durable facts, a
`ConversationStore` for full per-session transcripts (with branching
+ undo), and a `SemanticStore` for embedding-indexed chunks.

The `remember` tool inserts the K/V row and queues an embedding job
for its value on a `tokio::sync::mpsc` channel; each persisted
conversation turn queues its chunks on the same channel. A background
task drains the queue, sends batches to the embedding server (a
second, smaller llama-server child process configured under
`[embedding]`), and inserts the resulting vectors into the semantic
store. The `recall` tool embeds its query and ranks saved memories by
cosine similarity; the `reminisce` tool runs the same kind of search
over conversation chunks from earlier sessions.

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
which spawns the `piper` binary per utterance with `--output-raw`,
writes the sentence to its stdin, reads raw PCM from its stdout, and
plays it via rodio. Before each sentence, `VoiceOutputController`
returns a `SpeakDecision`: speak it, or drop it because TTS is toggled
off or the reply was skipped. Tunables live in `[voice.synthesis]`.

### Window manager (`assistd-wm`)

Pure abstraction over the compositor IPC. The daemon optionally
queries the active window at the start of each turn (title + class +
workspace) and folds it into that turn's user message, inside a
delimited context block labelled as untrusted, so the model knows
what you're looking at. Titles are application-controlled (a web
page sets its tab's title), so they are flattened to one bounded
line and any copy of the block's delimiters inside them is
neutralised before the text reaches the model. The `wm` command surfaces this same backend
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
slow MCP server: each call is bounded by that server's
`request_timeout_secs`, and a timeout surfaces as a structured error
to the model.

## Data flow: end-to-end query

A walk through `assistd query "what files changed this week?"`:

1. **Client.** `assistd query` constructs `Request::Query { id,
   text, attachments: [] }`, dials
   `$XDG_RUNTIME_DIR/assistd.sock`, writes the JSON line, half-closes
   the write side, and reads `Event` lines until `Done`.

2. **Socket server** ([`crates/assistd-core/src/socket.rs`](../crates/assistd-core/src/socket.rs)).
   Parses the request and hands it to `AppState::dispatch`, which
   routes a `Query` to `handle_query`; that takes the per-turn agent
   lock, so concurrent queries serialize.

3. **Agent** ([`crates/assistd-core/src/agent.rs`](../crates/assistd-core/src/agent.rs)).
   The handler builds `Agent::new(...)` from the cached `AppState`
   and calls `run_turn()`.

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
   (bubblewrap by default): a read-only root with writable `$HOME`
   and `/tmp`, fresh `/dev` and `/proc`, a tmpfs `/run`, and
   unshared pid/ipc/uts namespaces. The sandboxed `git` runs, returns
   stdout. If stdout exceeds the `[tools.output]` line or byte cap,
   `RunTool::invoke` cuts it to that head and spills the full text to
   `tools.output.overflow_dir` (default `/tmp/assistd-output`,
   emptied at every daemon start); it then base64-encodes any image
   attachments and returns the JSON result.

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

For `assistd chat`, the TUI is a long-lived client but not a single
connection: each turn opens its own connection, kept writable so the
TUI can answer confirmation prompts with `Request::ConfirmResponse`,
while separate long-lived `Request::Subscribe` connections carry
broadcast events such as session titles.

## Where to look next

- [Adding a tool](tools.md) — full code-level walkthrough of
  extending the registry.
- [Sample config](../config/config.sample.toml) — every tunable
  with prose explaining when to change it.
- [i3](wm/i3.md) and [Sway](wm/sway.md) — keybind recipes.
