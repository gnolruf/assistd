# Window-manager integration

The daemon binds the voice keys in `[hotkeys]` itself; the other
bindings that make `assistd` feel native (presence, the chat
scratchpad) live in your compositor config and call the `assistd` IPC
commands. This page covers what every setup shares; the [i3](i3.md)
and [Sway](sway.md) guides have the config snippets.

## Before you start

- `assistd` on your `$PATH` and a config at
  `~/.config/assistd/config.toml` (`assistd init-config` writes one).
- Set the voice keys in `[hotkeys]`. On Sway the daemon adds them as
  bindings over IPC, re-adds them after `swaymsg reload`, and removes
  them when it stops. On i3 and other X11 sessions it grabs them on
  the root window. Push-to-talk defaults to `Super+Space`; the others
  are empty, which leaves them unbound.

  ```toml
  [hotkeys]
  push_to_talk = "Super+Space"
  toggle_listening = "Super+Shift+L"
  ```

- Remove any `bindsym … exec assistd ptt-start` (or `listen-*`,
  `voice-*`) lines left from older setups, and any other binding on
  the same keys. On Sway the daemon's binding replaces yours and Sway
  warns about it; on i3 the daemon's grab fails and it logs a warning.
- On Wayland compositors other than Sway (Hyprland, GNOME, KDE) the
  daemon binds nothing; bind the `ptt-*`, `listen-*` and `voice-*`
  commands below in the compositor instead.

## What the commands do

- **`assistd daemon`** must be running; `assistd chat` and
  `assistd tray` never start it. If you use the systemd user unit (the
  AUR package installs it) instead of an `exec` line, plain i3 and Sway
  sessions usually never reach `graphical-session.target`, and the
  unit does not see the compositor's socket variables, so import them
  and start the unit from the compositor config (see each guide).
- **`ptt-start` / `ptt-stop`** record while a key is held, then
  transcribe and run the turn. Both exit with an error when `[voice]
  enabled = false`. An open chat shows the transcript and reply.
- **`sleep` / `drowse` / `wake`** set the presence state. `Active`
  has the model loaded, `Drowsy` keeps `llama-server` running with
  the weights unloaded, and `Sleeping` stops it, freeing its VRAM.
  `sleep` never suspends the machine; `wake` blocks until `Active`.
- **`chat --wait`** waits for the daemon's socket instead of exiting
  because it isn't up yet. If the chat exits, rerun the same command.
- **`listen-toggle`** needs `[voice.wakeword] enabled = true` and
  exits with an error otherwise; see [wake-word listening](../voice/wake-word.md). **`voice-toggle`** silences or
  resumes spoken replies, and **`voice-skip`** stops the current one;
  both need `[voice.synthesis] enabled = true` to have any effect.
- **`assistd tray`** shows a StatusNotifierItem icon (active, sleeping,
  listening, generating, or disconnected) and reconnects if the daemon
  restarts. Icon names are set under `[notifications.icons]`.

## Verifying the setup

After reloading the compositor config:

```sh
# 1. Daemon up?
assistd query "what time is it"

# 2. Presence changes work?
assistd drowse  # Active → Drowsy
assistd sleep   # Drowsy → Sleeping
assistd wake    # blocks until Active

# 3. Scratchpad visible?
#    Press $mod+grave; the chat should appear floating, centered.

# 4. PTT round-trips?
#    Hold $mod+space, say "hello", release. The chat streams the response.
```

If a step hangs, run `assistd daemon` in a terminal so its log output
is visible. The chat logs to `~/.local/state/assistd/chat.log.<date>`.

## Troubleshooting

- **A hotkey does nothing.** Check the daemon log for an
  `assistd::hotkey` line naming the key; on i3 a failed grab means
  another binding holds it.
- **`bindsym` fires but nothing happens.** The daemon isn't running.
- **PTT transcribes nothing.** The default capture device is probably
  the wrong one; set `[voice] mic_device` to a name from `arecord -L`.
