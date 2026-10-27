# Window-manager integration

assistd binds no keys itself. Every key binding, voice included, lives
in your compositor or desktop config and runs an `assistd` command
over IPC, the same way you'd bind `playerctl` or `makoctl`. This page
covers what every setup shares; the [i3](i3.md) and [Sway](sway.md)
guides have the config snippets.

## Before you start

- `assistd` on your `$PATH` and a config at
  `~/.config/assistd/config.toml` (`assistd init-config` writes one).
- Bind push-to-talk to a key's press and release. On Sway:

  ```
  bindsym --no-repeat $mod+space exec assistd ptt-start
  bindsym --release   $mod+space exec assistd ptt-stop
  ```

  i3 has no `--no-repeat`; a repeated `ptt-start` while the key is
  held is ignored. Any compositor or desktop that runs a command on key press and on
  release works the same way (Hyprland `bind` / `bindr`, or a
  GNOME or KDE custom shortcut for the toggles).
- Bind `listen-toggle`, `voice-toggle` and `voice-skip` the same way
  if you want them; they act on press only.

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

- **`bindsym` fires but nothing happens.** The daemon isn't running.
  Run the bound command in a terminal to see its error.
- **PTT transcribes nothing.** The default capture device is probably
  the wrong one; set `[voice] mic_device` to a name from `arecord -L`.
