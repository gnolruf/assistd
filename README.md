<h1 align="center">
  <img src="docs/assets/logo.svg" alt="assistd" width="380">
</h1>

A model agent daemon for Linux system management. `assistd` runs and manages an inference server,
exposes a Unix-socket IPC protocol to clients (one-shot CLI, ratatui chat TUI,
window-manager bindings), and integrates with the running compositor for context-aware behavior.

## Features

- **Tools and pipelines.** A single `run` tool exposes shell-shaped
  commands (`cat`, `grep`, `wc`, `head`, `tail`, `sort`, `uniq`, `bash`,
  `see`, `screenshot`, `web`, `wm`, …) the model composes with `|`,
  `&&`, `;`, plus `~` and glob expansion. Alongside it, `remember`
  saves facts for later turns and `reminisce` searches transcripts
  from earlier sessions.
  - Adding your own command is a single Rust file plus one
    registration line — see [docs/tools.md](docs/tools.md).
- **Voice in/out.** Push-to-talk and continuous listening via
  `whisper.cpp`; spoken responses via Piper TTS, streamed sentence by
  sentence so playback starts before the model has finished.
- **Vision when the model has it.** `see photo.png` and `screenshot`
  attach images to the next turn whenever the loaded model has a
  vision projector; capability is detected at runtime.
- **MCP-compatible.** Connect remote or local
  [Model Context Protocol](https://modelcontextprotocol.io) servers
  under `[[mcp.servers]]`; their tools register into the same registry
  the LLM sees, namespaced as `mcp__<server>__<tool>`.
- **Coarse presence.** `Active → Drowsy → Sleeping`. The daemon watches VRAM and drops to `Sleeping` when another process
  (a game, a second model runner) claims the GPU, freeing all of its
  VRAM; it can wake again once that process exits.
- **Compositor-aware.** Native i3 and Sway backends; active-window
  context can be injected automatically into every turn.
- **Persistent memory.** SQLite-backed conversation history plus
  semantic search over past chunks for cross-session recall.

### Prerequisites

- Rust toolchain, stable channel (selected by `rust-toolchain.toml`;
  rustup installs it automatically on first build).
- `llama-server` from `llama.cpp` on your `$PATH`. the daemon will spawn it as a child process.

Optional but recommended for the full feature set: `bubblewrap` (without it the model gets no tools), `piper` for TTS, and `grim` (Wayland) or `maim` (X11) for screenshots.

### 1. Build

```sh
cargo build --release
```

### 2. Configure

```sh
assistd init-config
```

This writes a working config to `~/.config/assistd/config.toml`. The
defaults are runnable as-is. Open the file if you want to:

- swap `[model] name` for a different HuggingFace GGUF id (the
  string is passed to llama-server's `--hf` flag),
- change `[model] gpu_layers` if VRAM is tight, or pass extra server
  flags through `[model] custom_args`,
- enable voice (`[voice] enabled = true`) or TTS
  (`[voice.synthesis] enabled = true`).

The [annotated sample config](config/config.sample.toml) documents
every field and is the canonical reference; `assistd init-config`
emits a subset with the defaults inline.

### 3. Run the daemon

```sh
assistd daemon
```

First launch downloads the model and waits for `llama-server` to
report healthy before accepting IPC. Subsequent starts are
sub-second. Logs stream to stderr; they're verbose by default and
reduce to warnings once the daemon is steady-state.

### 4. First query

```sh
assistd query "what time is it?"
```

You should see status text, then the streamed reply, then the
process exits 0. If the model decides to call a tool, you'll see
the tool call and result inline before the final reply.

```text
status: thinking
The current time is 14:32 PDT (2026-05-09).
```

Or, with a tool call:

```text
status: thinking
tool: run "date"
result: Sat May  9 14:32:11 PDT 2026
The current time is 14:32 PDT.
```

### 5. Open the chat TUI

```sh
assistd chat
```

The TUI is a long-lived client onto the running daemon; it exits
with a hint if no daemon is listening (pass `--wait` to wait for one
instead). Only one chat may be open at a time, since every chat
drives the daemon's single conversation.

## Window-manager integration

The bindings that make `assistd` feel native (push-to-talk, presence
toggling, the TUI scratchpad) live in your compositor config rather
than the daemon. Compositor-specific guides:

- [i3](docs/wm/i3.md)
- [Sway](docs/wm/sway.md)

## CLI surface

A non-exhaustive tour. `assistd --help` is the canonical reference.

| Command                  | What it does                                                  |
|--------------------------|---------------------------------------------------------------|
| `assistd daemon`         | Run the daemon (foreground).                                  |
| `assistd init-config`    | Write the default config to `~/.config/assistd/config.toml`.  |
| `assistd query <text>`   | One-shot query; prints the streamed response and exits.       |
| `assistd chat`           | Interactive ratatui TUI.                                      |
| `assistd cycle`          | Step `Active → Drowsy → Sleeping → Active`.                   |
| `assistd sleep` / `wake` | Jump to a specific presence state.                            |
| `assistd ptt-start/stop` | Push-to-talk start/stop, for window-manager bindings.         |
| `assistd listen-toggle`  | Toggle hands-free continuous listening.                       |
| `assistd voice-toggle`   | Toggle Piper TTS playback.                                    |
| `assistd voice-skip`     | Abort the current spoken response.                            |
| `assistd memory …`       | Inspect or mutate persistent memory; see `memory --help`.     |

## Documentation

- [Architecture](docs/architecture.md) — crate map, data flow, and
  how the subsystems fit together.
- [Adding a tool](docs/tools.md) — extending the tool registry with
  a complete worked example.
- [Sample config](config/config.sample.toml) — every tunable with
  prose explaining when to change it.
- [i3 setup](docs/wm/i3.md) and [Sway setup](docs/wm/sway.md) —
  copy-paste keybinds for push-to-talk, presence, and the TUI
  scratchpad.
- [Echo cancellation](docs/voice/echo-cancellation.md) — let
  PipeWire or PulseAudio cancel TTS echo so continuous listening can
  be interrupted mid-reply.
