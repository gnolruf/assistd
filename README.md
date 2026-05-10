# assistd

A local-model agent daemon for Linux. `assistd` runs a LLM under
[`llama.cpp`](https://github.com/ggerganov/llama.cpp), exposes a
Unix-socket IPC protocol to clients (one-shot CLI, ratatui chat TUI,
window-manager bindings), and integrates with the running compositor
for context-aware behavior.

## Highlights

- **Local-first.** `llama-server` runs as a child process; everything
  (text, voice, images) stays on the machine.
- **Tools and pipelines.** A single `run` tool exposes shell-shaped
  commands (`cat`, `grep`, `wc`, `bash`, `see`, `screenshot`, `web`,
  `wm`, …) the model composes with `|`, `&&`, `;`. Adding your own is
  a single Rust file plus one registration line — see
  [docs/tools.md](docs/tools.md).
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
- **Coarse presence.** `Active → Drowsy → Sleeping` so a single hotkey
  frees the GPU when you switch to gaming.
- **Compositor-aware.** Native i3 and Sway backends; active-window
  context can be injected automatically into every turn.
- **Persistent memory.** SQLite-backed conversation history plus
  semantic search over past chunks for cross-session recall.

## Quick start

Aim: zero to a working text query in under ten minutes (with
dependencies installed). For window-manager bindings and TUI
integration, see the [i3](docs/wm/i3.md) and [Sway](docs/wm/sway.md)
guides after this.

### Prerequisites

- Rust toolchain. The version is pinned via `rust-toolchain.toml`
  (currently `≥ 1.85`); rustup installs it automatically on first
  build.
- `llama-server` from `llama.cpp` on your `$PATH`. Verify with
  `which llama-server`. If it isn't installed, follow the upstream
  build instructions; the daemon will spawn it as a child process.

Optional but recommended for the full feature set: a GPU with enough
VRAM to hold the model (CUDA, ROCm, or Metal builds of llama.cpp
work), `bubblewrap` for the bash sandbox, `piper` for TTS, and
`grim` (Wayland) or `maim` (X11) for screenshots.

### 1. Build

```sh
cargo build --release
```

The binary lands at `target/release/assistd`. Symlink it onto your
`$PATH` or invoke by absolute path; the snippets below assume
`assistd` is callable directly.

### 2. Configure

```sh
assistd init-config
```

This writes a working config to `~/.config/assistd/config.toml`. The
defaults are runnable as-is. Open the file if you want to:

- swap `[model] name` for a different HuggingFace GGUF id (the
  string is passed to llama-server's `--hf` flag),
- change `[llama_server] gpu_layers` if VRAM is tight,
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

In another terminal:

```sh
assistd query "what time is it?"
```

You should see status text, then the streamed reply, then the
process exits 0. If the model decides to call a tool, you'll see
the tool call and result inline before the final reply.

```
status: thinking
The current time is 14:32 PDT (2026-05-09).
```

Or, with a tool call:

```
status: thinking
tool: run "date"
result: Sat May  9 14:32:11 PDT 2026
The current time is 14:32 PDT.
```

### 5. Open the chat TUI

```sh
assistd chat
```

The TUI is a long-lived client onto the running daemon. It
auto-spawns the daemon under `setsid` if no socket is reachable, so
step 3 is technically optional — running the daemon explicitly just
front-loads the cold-start cost.

That's it. If something hangs, run `assistd daemon` directly so its
tracing output is visible; if the model never finishes downloading,
check that `[llama_server] ready_timeout_secs` is high enough for
your connection.

## Window-manager integration

The bindings that make `assistd` feel native (push-to-talk, presence
toggling, the TUI scratchpad) live in your compositor config rather
than the daemon. Compositor-specific guides:

- [i3](docs/wm/i3.md)
- [Sway](docs/wm/sway.md)

Each guide ships copy-paste snippets and finishes in under five
minutes.

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

## Configuration

Everything is in one TOML file at `~/.config/assistd/config.toml`.
`assistd init-config` writes a working starting point; the
[annotated sample config](config/config.sample.toml) is the canonical
reference, with explanatory comments on every field (including the
optional ones with sensible defaults the daemon falls back to).

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

## License

Apache 2.0 (see [LICENSE](LICENSE)).
