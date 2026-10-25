# Echo Cancellation for Continuous Listening

Continuous listening keeps the mic open while a reply is spoken, so
through speakers it hears the daemon's own TTS. During a reply to a
wake-word command, speech interrupts the reply (barge-in,
`voice.wakeword.barge_in`) once its transcript holds words the
reply did not just say. That rejects echo by text alone, which works
through headphones but is unreliable through speakers: the transcriber mishears
echo mixed with your voice, and a command you give over the reply
carries some of the reply's words.

With speakers, let the sound server cancel the echo. `assistd` does not implement echo
cancellation itself; it consumes whatever mic source the system hands
it. Both PipeWire and PulseAudio ship WebRTC's echo canceller as a
loadable module that creates a virtual sink and a virtual source. Audio played to the sink is used as
the reference signal and subtracted from the source.

Two conditions must hold for this to work:

- Replies play through the echo-cancel **sink**, and the mic is read
  from the echo-cancel **source**. The easiest way is to make both the
  system defaults and leave `voice.mic_device` and
  `voice.synthesis.output_device` unset; `assistd` opens the default
  ALSA device, which the sound server routes to its defaults (this needs
  `pipewire-alsa` or `pulseaudio-alsa`).
- Speakers and mic sit on the same sound server. A Bluetooth headset
  in headset (HFP) mode usually cancels echo in hardware already and
  needs none of this.

## PipeWire

Create `~/.config/pipewire/pipewire.conf.d/echo-cancel.conf`:

```
context.modules = [
    {   name = libpipewire-module-echo-cancel
        args = {
            source.props = {
                node.name        = "echo-cancel-source"
                node.description = "Echo-cancelled mic"
            }
            sink.props = {
                node.name        = "echo-cancel-sink"
                node.description = "Echo-cancelled playback"
            }
        }
    }
]
```

Restart PipeWire and make the new nodes the defaults:

```sh
systemctl --user restart pipewire pipewire-pulse wireplumber
wpctl status            # note the ids of "Echo-cancelled mic" and "Echo-cancelled playback"
wpctl set-default <source-id>
wpctl set-default <sink-id>
```

## PulseAudio

Append to `~/.config/pulse/default.pa` (create it with `.include
/etc/pulse/default.pa` as the first line if it does not exist):

```
load-module module-echo-cancel aec_method=webrtc use_master_format=1 source_name=echo-cancel-source sink_name=echo-cancel-sink
set-default-source echo-cancel-source
set-default-sink echo-cancel-sink
```

Then `pulseaudio -k` and let it respawn.

## Check it

Restart the daemon. If barge-in fires on the daemon's own replies, the
cancelled source is not the one being captured; check `wpctl status`
or `pactl info` and set the defaults again. Bare ALSA has no
echo-cancellation module, so use headphones there.
