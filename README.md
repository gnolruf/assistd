<h1 align="center">
  <img src="docs/assets/logo.svg" alt="assistd" width="380">
</h1>

<p align="center">
  <a href="https://github.com/gnolruf/assistd/actions/workflows/ci.yml"><img src="https://github.com/gnolruf/assistd/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://github.com/gnolruf/assistd/actions/workflows/codeql.yml"><img src="https://github.com/gnolruf/assistd/actions/workflows/codeql.yml/badge.svg?branch=main" alt="CodeQL"></a>
  <a href="https://github.com/gnolruf/assistd/releases/latest"><img src="https://img.shields.io/github/v/release/gnolruf/assistd" alt="Latest release"></a>
  <a href="https://aur.archlinux.org/packages/assistd"><img src="https://img.shields.io/aur/version/assistd" alt="AUR version"></a>
</p>

A model agent daemon for Linux system management. `assistd` runs and manages an inference server,
exposes a Unix-socket IPC protocol to clients (one-shot CLI, ratatui chat TUI,
window-manager bindings), and integrates with the running compositor for context-aware behavior.

## Features

- **Tools and pipelines.** A single `run` tool exposes shell-shaped
  commands (`cat`, `grep`, `wc`, `head`, `tail`, `sort`, `uniq`, `bash`,
  `see`, `screenshot`, `web`, `wm`, …) the model composes with `|`,
  `&&`, `||`, `;`, plus `~` and glob expansion. `web` both fetches
  pages as readable text and, with a local SearXNG installed, searches
  the web ([web search](docs/web-search.md)). Alongside it, `remember`
  saves and looks up facts, and `reminisce` searches
  transcripts from earlier sessions.
- **Voice in/out.** Push-to-talk and hands-free
  [wake-word listening](docs/voice/wake-word.md) via Parakeet speech-to-text,
  with the wake phrase found by a keyword spotter and tested in the chat
  TUI's `/wakeword`; spoken responses through a sherpa-onnx voice (Piper or
  Kokoro) in a worker process the daemon supervises, streamed sentence
  by sentence so playback starts before the model has finished, and
  interruptible by talking over them.
- **Vision when the model has it.** `see photo.png` and `screenshot`
  attach images to the next turn whenever the loaded model has a
  vision projector; capability is detected at runtime.
- **MCP-compatible.** Connect stdio
  [Model Context Protocol](https://modelcontextprotocol.io) servers
  under `[[mcp.servers]]`; their tools register into the same registry
  the LLM sees, namespaced as `mcp__<server>__<tool>`.
- **Coarse presence.** `Active → Drowsy → Sleeping`. On NVIDIA GPUs
  the daemon watches VRAM and drops to `Sleeping` when another process
  (a game, a second model runner) claims the GPU, freeing all of its
  VRAM; with `[sleep] gpu_auto_wake` it wakes once that process exits.
- **Compositor-aware.** Native i3 and Sway backends; active-window
  context can be injected automatically into every turn.
- **Persistent memory.** SQLite-backed conversation history plus
  semantic search over past chunks for cross-session recall.

### Prerequisites

- Rust toolchain, stable channel (selected by `rust-toolchain.toml`;
  rustup installs it automatically on first build).
- `cmake`, `clang`, `pkg-config` and the ALSA headers (`alsa-lib` /
  `libasound2-dev`) to build.
- `llama-server` from `llama.cpp` on your `$PATH`. The daemon spawns it
  as a child process.
- `bubblewrap`. Without it the model gets no tools.

Optional: `espeak-ng` for TTS with Piper voices, `grim` (Wayland) or `maim` (X11) for
screenshots, and SearXNG's `searxng-run` for web search (`searxng-git`
on the AUR; enable with `[search] enabled = true`, see
[web search](docs/web-search.md)).

On Arch, the `assistd` and `assistd-git` AUR packages install the
binary and a systemd user unit (`systemctl --user enable --now assistd`).

### 1. Build

```sh
cargo build --release
```

The build downloads sherpa-onnx's prebuilt libraries from its GitHub
release; set `SHERPA_ONNX_ARCHIVE_DIR` to a directory holding
`sherpa-onnx-v2.0.0-linux-x64-shared-lib.tar.bz2` to build offline.
`libsherpa-onnx-c-api.so` and `libonnxruntime.so` land beside the binary
in `target/release/`; if you copy `assistd` elsewhere, put them in the
same directory or in `../lib/assistd/` relative to it.

Speech-to-text runs on the CPU unless the binary ships sherpa-onnx's CUDA
libraries. For GPU transcription, install `cuda` (13.x) and `cudnn`,
extract `sherpa-onnx-v2.0.0-cuda-13.x-cudnn-9.x-onnxruntime1.28.2-linux-x64-gpu.tar.bz2`
from the sherpa-onnx release, and build with
`SHERPA_ONNX_LIB_DIR=<extracted>/lib cargo build --release`. With
`[voice.transcription] prefer_gpu = true` (the default) the daemon then
transcribes on CUDA, using about 2.6 GB of VRAM.

### 2. Configure

```sh
assistd init-config
```

This writes a working config to `~/.config/assistd/config.toml`. The
defaults are runnable as-is. Open the file if you want to:

- swap `[model] name` for a different Hugging Face GGUF
  (`owner/repo:quant`),
- change `[model] gpu_layers` if VRAM is tight, or pass extra server
  flags through `[model] custom_args`,
- enable voice (`[voice] enabled = true`) or TTS
  (`[voice.synthesis] enabled = true`).

`init-config` refuses to overwrite an existing file. The
[sample config](config/config.sample.toml) lists every key.

### 3. Run the daemon

```sh
assistd daemon
```

The socket opens at once; the model downloads (first launch only) and
loads in the background, and queries wait for it. Logs go to stdout at
`info`; set `RUST_LOG` to change that.

### 4. First query

```sh
assistd query "what time is it?"
```

The reply streams to stdout, with a line for each tool call, and the
process exits 0:

```text
[tool call: run date]
[tool result: run exit:0]
The current time is 14:32 PDT.
```

`query` cannot answer confirmation prompts, so a command that needs
one is denied; use `chat` for those.

### 5. Open the chat TUI

```sh
assistd chat
```

The TUI is a long-lived client onto the running daemon; it exits
with a hint if no daemon is listening (pass `--wait` to wait for one
instead). Only one chat may be open at a time, since every chat
drives the daemon's single conversation.

`/settings` in the chat, or `assistd settings` on its own, opens the
config in an editable panel: hotkeys first, then each section, with
its detailed keys behind an "advanced" toggle. Saved changes take
effect when the daemon restarts.

## Window-manager integration

The daemon binds the voice hotkeys in `[hotkeys]` itself on X11, Sway
and i3. The other bindings (presence toggling, the TUI scratchpad) live
in your compositor config. Start with the
[shared setup page](docs/wm/README.md), then the [i3](docs/wm/i3.md)
or [Sway](docs/wm/sway.md) guide.

## CLI surface

A non-exhaustive tour. `assistd --help` is the canonical reference.

| Command                  | What it does                                                  |
|--------------------------|---------------------------------------------------------------|
| `assistd daemon`         | Run the daemon (foreground).                                  |
| `assistd init-config`    | Write the default config to `~/.config/assistd/config.toml`.  |
| `assistd query <text>`   | One-shot query; prints the streamed response and exits.       |
| `assistd chat`           | Interactive ratatui TUI.                                      |
| `assistd settings`       | Edit `config.toml` in a terminal panel (also `/settings` in chat). |
| `assistd sleep` / `drowse` / `wake` | Jump to a specific presence state.                 |
| `assistd ptt-start/stop` | Push-to-talk start/stop, for compositors without hotkey support. |
| `assistd listen-toggle`  | Toggle [wake-word listening](docs/voice/wake-word.md).        |
| `assistd voice-toggle`   | Toggle TTS playback.                                          |
| `assistd voice-skip`     | Abort the current spoken response.                            |
| `assistd memory …`       | Inspect or mutate persistent memory; see `memory --help`.     |
| `assistd tray`           | System tray icon and desktop notifications.                   |
