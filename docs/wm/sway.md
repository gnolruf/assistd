# Getting Started with Sway

Sway config snippets for `assistd`. Read the
[shared setup page](README.md) first: it covers the key bindings,
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

## Push-to-talk

```sway
bindsym --no-repeat $mod+space exec assistd ptt-start
bindsym --release   $mod+space exec assistd ptt-stop
```

The default Sway config binds `$mod+space` to `focus mode_toggle`;
remove that line or pick another key.

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

```sway
bindsym $mod+Shift+l exec assistd listen-toggle
bindsym $mod+Shift+m exec assistd voice-toggle
bindsym $mod+Shift+s exec assistd voice-skip
```

## Window commands fail

The daemon couldn't reach Sway, usually because it runs under systemd
without `$SWAYSOCK`; import it as shown above.
