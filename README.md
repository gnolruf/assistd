# assistd

A local model agent OS assistant daemon for Linux. `assistd` runs a
local LLM (via [`llama.cpp`](https://github.com/ggerganov/llama.cpp)),
exposes a Unix-socket IPC protocol to clients (CLI, ratatui chat TUI,
window-manager bindings), and integrates with the running compositor
for context-aware behavior.

> Status: early development. Interfaces and config keys may shift
> between commits.

## Highlights

- **Local-first.** `llama-server` runs as a child process; nothing is
  uploaded except optional vision attachments (which still go to the
  *local* model).
- **Voice in/out.** Push-to-talk and continuous listening via
  `whisper.cpp`; spoken responses via Piper TTS.
- **Coarse presence.** `Active → Drowsy → Sleeping` so a single hotkey
  frees the GPU when you switch to gaming.
- **Compositor-aware.** Native i3 and Sway backends; Hyprland is on
  the roadmap.
- **Persistent memory.** SQLite-backed conversation history plus
  semantic search over past chunks for cross-session recall.

## Build

```sh
# Toolchain pinned by rust-toolchain.toml (Rust ≥ 1.85).
cargo build --release
```

The binary lands at `target/release/assistd`. Symlink it onto your
`$PATH` or run via the absolute path; the snippets in the docs assume
`assistd` is callable directly.

You also need `llama-server` (from `llama.cpp`) on `$PATH` — the
daemon spawns it with the model identifier from your config. See the
sample config for the exact field name.

## Quick start

```sh
# 1. Write a default config to ~/.config/assistd/config.toml.
assistd init-config

# 2. Edit the config — at minimum set [model] name to a HuggingFace
#    GGUF id and double-check [llama_server] binary_path.
$EDITOR ~/.config/assistd/config.toml

# 3. Run the daemon. Cold-starts llama-server before accepting IPC.
assistd daemon

# 4. In another terminal, send a one-shot query …
assistd query "summarize what's in this directory"

# 5. … or open the chat TUI.
assistd chat
```

`assistd chat` auto-spawns the daemon under `setsid` when no socket
is reachable, so step 3 is optional in practice. Starting the daemon
explicitly just front-loads the cold-start cost.

## Window-manager integration

The bindings that make `assistd` feel native — push-to-talk, presence
toggling, the TUI scratchpad — live in your compositor config rather
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

## License

Apache 2.0 — see [LICENSE](LICENSE).
