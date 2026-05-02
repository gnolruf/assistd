# Getting Started with Sway

This page mirrors the [i3 guide](./i3.md) for Sway. The IPC dialect is
nearly identical — Sway implements the i3 protocol — but a few
Wayland-specific details differ:

- Sway prefers `app_id` over `WM_CLASS` for Wayland-native windows.
  The snippets below match scratchpad windows by `app_id`; XWayland
  fallbacks (`class="…"`) work too.
- The in-daemon global hotkey listener uses `XGrabKey` and is silently
  inert under Sway. Set every `*_hotkey` field in `config.toml` to
  `""` and bind via Sway instead — the only path that works.

If you can copy a block into `~/.config/sway/config` and run `swaymsg
reload`, you can finish this guide in under five minutes.

## Prerequisites

- `assistd` on your `$PATH`. A debug build lives at
  `target/debug/assistd` after `cargo build`; a release build at
  `target/release/assistd` after `cargo build --release`.
- A config file at `~/.config/assistd/config.toml`. Create one with
  defaults via:

  ```sh
  assistd init-config
  ```

- A terminal emulator that lets you set the Wayland `app_id` (or
  XWayland `WM_CLASS`) from the command line. Working flag spellings
  for the common terminals:

  | Terminal  | Flag                        |
  |-----------|-----------------------------|
  | foot      | `--app-id assistd-tui`      |
  | alacritty | `--class assistd-tui`       |
  | kitty     | `--class assistd-tui`       |
  | wezterm   | `start --class assistd-tui` |

  Recent releases set `app_id` under Wayland; older builds may only
  set `WM_CLASS` (i.e. XWayland). The troubleshooting section below
  covers what to do if the `[app_id="…"]` criteria never match.

## Disable the in-daemon hotkey listeners

Set these fields in `~/.config/assistd/config.toml` so the unused X11
grabs don't run at all:

```toml
[voice]
hotkey = ""

[voice.continuous]
hotkey = ""

[voice.synthesis]
toggle_hotkey = ""
skip_hotkey = ""

[presence]
hotkey = ""
```

The IPC commands (`assistd ptt-start`, `assistd cycle`, …) keep
working — that is what Sway will call.

## Auto-start the daemon

The daemon manages `llama-server`, voice capture, TTS, and presence.
Start it once per session so the first query doesn't pay the cold-boot
cost:

```sway
# Launch assistd at session start. Sway exports $SWAYSOCK to child
# processes, so the daemon's compositor auto-detection picks the
# Sway backend without any further config.
exec assistd daemon
```

If you'd rather let `assistd chat` auto-spawn the daemon on first use
(it runs `assistd daemon --client-mode` under `setsid`), skip this
line. The trade-off is a slightly slower first launch.

## Push-to-talk (hold to speak)

Press-and-hold `$mod+space` to record; release to transcribe and send
to the LLM. Sway's `--release` modifier matches i3's:

```sway
# Push-to-talk: hold $mod+space to capture, release to transcribe.
# `--release` makes Sway fire the second bindsym only on key-up.
bindsym           $mod+space exec assistd ptt-start
bindsym --release $mod+space exec assistd ptt-stop
```

`assistd ptt-stop` returns immediately after kicking off the agent
turn; the daemon owns the streaming. If voice support is disabled
(`[voice] enabled = false` in the config), both commands exit with a
clear error line.

## Presence: sleep, wake, cycle

Presence is the daemon's coarse lifecycle: `Active` (model loaded and
ready), `Drowsy` (server alive, weights unloaded), and `Sleeping`
(server stopped, all VRAM freed). One binding cycles through them;
two more let you jump to a target state:

```sway
# Cycle Active → Drowsy → Sleeping → Active. One binding, three
# states; short-press repeatedly to step through them.
bindsym $mod+Escape exec assistd cycle

# Optional: jump straight to a target state.
bindsym $mod+Shift+Escape exec assistd sleep
bindsym $mod+Shift+w      exec assistd wake
```

`assistd sleep` will also suspend the machine if you set
`[sleep] suspend = true` in `config.toml`; otherwise it just frees the
GPU. `assistd wake` blocks until `llama-server` is back at `Active`.

## TUI scratchpad

The chat TUI (`assistd chat`) is a thin client onto the running
daemon. Launch it inside a known-`app_id` terminal at session start,
immediately stash the window in Sway's scratchpad, and toggle it with
one key.

```sway
# Spawn the TUI inside a tagged terminal at session start. The
# `--app-id` value ("assistd-tui") is what the criteria below match
# against. Substitute your terminal's flag from the table above.
exec foot --app-id assistd-tui assistd chat

# Move the TUI window straight to the scratchpad on map, and give it
# a comfortable floating geometry. `floating enable` is implied by
# `move scratchpad` but stating it makes the resize predictable.
for_window [app_id="assistd-tui"] floating enable, move scratchpad, \
           resize set 1200 800, move position center

# Toggle the TUI in and out of view with $mod+grave (the backtick
# key, just under Escape on US layouts).
bindsym $mod+grave [app_id="assistd-tui"] scratchpad show
```

If your terminal sets `WM_CLASS` instead of `app_id` (because it's
running under XWayland, or it's an older release), swap `app_id="…"`
for `class="…"` in the criteria. Sway treats both forms as valid.

## Optional: continuous listening and TTS controls

Continuous listening keeps the mic open between utterances; TTS
playback control lets you toggle and abort spoken responses without
opening the TUI:

```sway
# Hands-free continuous listening: toggle on/off.
bindsym $mod+Shift+l exec assistd listen-toggle

# Toggle Piper TTS playback. "Off" silences the rest of the current
# response; "on" resumes from the next sentence.
bindsym $mod+Shift+m exec assistd voice-toggle

# Abort the current TTS response without changing the on/off flag.
bindsym $mod+Shift+s exec assistd voice-skip
```

These bindings only do something useful when the relevant subsystem is
enabled in `config.toml` (`[voice.continuous] enabled = true`,
`[voice.synthesis] enabled = true`). They are safe to bind regardless
— disabled subsystems return a clear "not enabled" error rather than
crashing the daemon.

## Verifying the setup

After `swaymsg reload`, run through this checklist:

```sh
# 1. Daemon up?
assistd query "what time is it"

# 2. Presence cycle works?
assistd cycle   # Active → Drowsy
assistd cycle   # Drowsy → Sleeping
assistd wake    # blocks until Active

# 3. Scratchpad visible?
#    Press $mod+grave; the TUI should appear floating, centered.

# 4. PTT round-trips?
#    Hold $mod+space, say "hello", release. The TUI's output pane
#    streams the response.
```

If a step hangs, run `assistd daemon` directly in a terminal so its
tracing output is visible — the Sway `exec` line discards stdout/stderr
to your session's default sink. The TUI keeps its own log at
`~/.local/state/assistd/chat.log` regardless of how the daemon was
launched.

## Troubleshooting

- **`bindsym` fires but nothing happens.** The daemon isn't running.
  Either `exec assistd daemon` from the Sway config or open the TUI
  once to auto-spawn it.
- **Scratchpad criteria never match.** Run `swaymsg -t get_tree | jq
  '.. | select(.app_id?)'` and confirm the TUI's `app_id`. Some
  terminals fall back to `class` under XWayland; switch the criteria
  to `[class="assistd-tui"]` if needed.
- **PTT records silence.** Check `[voice] enabled = true` and that
  `[voice.transcription] model = …` resolves on first use (the
  download lives under `~/.cache/assistd/whisper/`).
- **Daemon picked the wrong compositor.** Auto-detection prefers
  `$SWAYSOCK`, but if it's unset (a manual `sway --no-sock` start, a
  detached process, …) the daemon falls back to `NoWindowManager`.
  Pin it explicitly:

  ```toml
  [compositor]
  type = "sway"
  ```
