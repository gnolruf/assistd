# Getting Started with Sway

Sway config snippets for `assistd`. Read the
[shared setup page](README.md) first: it covers the hotkeys,
what each command does, and how to verify the result.

## Terminal

The scratchpad snippet matches the chat's terminal by Wayland
`app_id`:

| Terminal  | Flag                        |
|-----------|-----------------------------|
| foot      | `--app-id assistd-tui`      |
| alacritty | `--class assistd-tui`       |
| kitty     | `--class assistd-tui`       |
| wezterm   | `start --class assistd-tui` |

A terminal running under XWayland has a `class` instead; swap
`app_id="…"` for `class="…"` in the criteria below.

## Auto-start the daemon and tray

```sway
exec assistd daemon
exec assistd tray
```

The daemon finds Sway through `$SWAYSOCK`, which Sway exports to its
children. To use the systemd user unit instead of the first line:

```sway
exec systemctl --user import-environment SWAYSOCK WAYLAND_DISPLAY DISPLAY && systemctl --user start assistd
```

`swaybar` shows the tray icon when built with tray support; Waybar's
`tray` module works too.

## Hotkeys

The daemon binds `[hotkeys]` through Sway's IPC, so push-to-talk needs
no lines in the Sway config. `Super+Space` becomes `Mod4+space`.

The default Sway config binds `$mod+space` to `focus mode_toggle`.
The daemon's binding replaces it while the daemon runs, and Sway warns
about the clash, so remove that line or pick another key.

## Presence

```sway
bindsym $mod+Shift+Escape exec assistd sleep
bindsym $mod+Shift+w      exec assistd wake
```

## Chat scratchpad

```sway
exec foot --app-id assistd-tui assistd chat --wait

for_window [app_id="assistd-tui"] floating enable, move scratchpad, \
           resize set 1200 800, move position center

bindsym $mod+grave [app_id="assistd-tui"] scratchpad show
```

If the criteria never match, check the chat's `app_id` with
`swaymsg -t get_tree | jq '.. | select(.app_id?)'`.

## Optional: continuous listening and TTS

```toml
[hotkeys]
toggle_listening = "Super+Shift+L"
toggle_speech = "Super+Shift+M"
skip_speech = "Super+Shift+S"
```

## Window commands fail

The daemon couldn't reach Sway, usually because it runs under systemd
without `$SWAYSOCK`; import it as shown above. Hotkeys need the same
socket.
