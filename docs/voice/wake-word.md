# Wake-Word Listening

With wake-word listening on, `assistd` keeps the mic open and acts on
speech that starts with the wake phrase: "hey computer, what's on my
calendar" is sent as a query, and anything else said in the room is
dropped. The phrase is "hey computer" until you choose your own with
`/wakeword` in the chat TUI. Everything runs locally.

## Turning it on

```toml
[voice]
enabled = true

[voice.wakeword]
enabled = true
start_on_launch = true      # otherwise it waits for listen-toggle

[hotkeys]
# toggle_listening = "Super+Shift+L"  # X11 only; on Wayland bind `assistd listen-toggle`
```

Restart the daemon. `assistd listen-toggle` turns listening on and off,
and `assistd listen-state` prints the current phase:

| Phase       | Meaning                                                        |
|-------------|----------------------------------------------------------------|
| `off`       | The mic is closed.                                             |
| `armed`     | Listening for the wake phrase.                                 |
| `capturing` | The phrase was heard; the next thing you say is the command.   |

The tray shows its listening icon only while `capturing`. In `armed`
it shows the normal icon.

If `listen-toggle` says listening is disabled in config, both `enabled`
keys above must be `true`. If the daemon log says "Silero VAD failed to
load; listening unavailable", the VAD model
(`voice.transcription.vad_model`) could not be downloaded or read.

### Privacy

No audio or text leaves the machine. The Whisper and Silero models are
downloaded from Hugging Face once, on first use. After that, all audio
is processed locally.

Every utterance the VAD marks as speech is transcribed, including
conversation not meant for `assistd`. That is the only way to check for
the phrase. Transcripts that don't start with the phrase are discarded:
they are not sent to the model and not saved. The only exception is
`trace`-level logging, which records them. Leave
`assistd::voice::listen` below `trace` unless you are debugging.

## Choosing a wake phrase: `/wakeword`

Open `assistd chat` and type `/wakeword`. The panel shows the current
phrase, whether it is built in or how many samples it was learned from,
the variants it also accepts, and whether listening is on. It has three
actions:

- **Update** records a new phrase.
- **Test** checks what would wake the listener.
- **Reset** goes back to "hey computer".

Enrollment and testing need both `enabled` keys on. They take over the
mic while they run: listening pauses and then resumes, and push-to-talk
is refused until you finish. Esc backs out of any view.

### Picking a phrase

A phrase has one to six words, using only letters, spaces and
apostrophes (so "hey r2" is refused). While you type, a badge rates the
phrase by estimated syllables: under 3 is **low**, 3 is **medium**, and
4 or more is **high**. Longer phrases wake the listener by accident less
often. Also avoid:

- common words, especially at the start ("okay", "so", "hey there");
- words Whisper may spell several ways (names, invented words), since
  each spelling has to be learned;
- phrases you would say to a person in the room.

### Enrolling

Press Enter to start, then say the phrase on its own, once per
recording, at your normal distance from the mic. After each recording
the panel shows one of these lines:

| Line                                         | Meaning                                                                                       |
|----------------------------------------------|-----------------------------------------------------------------------------------------------|
| `✓ heard "hey computer"`                     | Whisper heard the phrase clearly. Counts towards the five clear samples needed.               |
| `~ heard "hay computer", kept as a variant`  | Close to the phrase, but no match. The spelling is learned as a variant; it doesn't count.    |
| `✗ didn't sound like the phrase: heard "…"`  | Too far from the phrase to be a mishearing of it. Discarded.                                  |
| `✗ too short; say the whole phrase`          | Under a quarter of a second of speech. Not transcribed.                                       |
| `✗ too long; say just the phrase`            | Over 4 s of speech. Not transcribed.                                                          |
| `✗ clipped; turn the mic gain down`          | The recording hit full scale. Lower the input gain and try again.                             |

A clear sample whose spelling is new (for example "Hey, computers") is
also learned as a variant.

Enrollment saves itself after the **fifth clear sample**, so you never
have to decide when to stop. The new phrase replaces the old one
straight away; you don't need to restart anything. Press `t` on the
confirmation screen to test it.

Watch the `~` lines. A few mean Whisper sometimes mishears the phrase,
and the variants cover that. If they keep coming while the clear count
stalls, Whisper doesn't hear this phrase reliably. Every variant is
also a phrase that will wake the listener, so a variant that sounds
like everyday speech ("hey come here") will cause false wakes. In
either case press Esc, which saves nothing and keeps the previous
phrase, and choose a longer phrase with more distinctive words.

A session with no speech for two minutes cancels itself. Enrolling
again replaces the whole profile, including its variants.

### Testing

**Test** listens with the saved phrase and its variants and shows, for
each utterance, what Whisper heard and whether it would wake:

```text
✓ would wake: "Hey computer, what time is it?"
✗ would not wake: "Hey commuter, what time is it?"
```

Here is what to do with each kind of result:

- **The text is right but it would not wake.** The phrase was probably
  not at the start, or one word scored just below `match_similarity`.
  Re-enroll so the spelling is learned as a variant, or lower
  `match_similarity` slightly.
- **The text is wrong.** The problem is the audio or Whisper, not the
  matching: mic placement, gain, background noise, or a small Whisper
  model (`voice.transcription.model`). Changing `match_similarity`
  won't fix it.
- **Something other than the phrase would wake.** That is a false-wake
  risk. Raise `match_similarity`, or enroll a longer phrase.

The test checks only the phrase. Command windows, follow-ups and
barge-in are not part of it.

### Where the profile lives

`$XDG_DATA_HOME/assistd/wakeword/` (by default
`~/.local/share/assistd/wakeword/`) holds:

- `profile.json`: the phrase, its variants, and what Whisper heard in
  each sample;
- `samples/01.wav` onwards: the enrollment recordings, 16 kHz mono.

These are recordings of your voice. The directory is readable only by
you, and **Reset** deletes it. If the daemon can't read the profile at
startup, for example after an upgrade that changed its format, it logs
a warning and uses "hey computer" until you enroll again.

## Conversations

**Follow-ups.** After the reply to a wake-word command finishes
(including its speech), the next utterance that starts within
`follow_up_ms` is taken as a command without the phrase. Each
reply opens a new window, so a back-and-forth can continue hands-free.
Set `follow_up_ms = 0` to require the phrase every time.

**Push-to-talk** takes the mic from the listener while held. Anything
the listener had captured but not yet acted on is dropped, and the
conversation ends: no follow-up window opens until you say the phrase
again.

**Sleep.** Listening keeps running while presence is `Sleeping`. A
command wakes the model like any other query. The first reply is slower
because the model has to load, and Whisper runs on the CPU until the
model is back. If the GPU lacks room for the model
(`sleep.gpu_headroom_mb`), the wake is refused and the command fails
with the reason.

### Hearing its own replies

The mic stays open while replies are spoken, so the daemon hears its
own speech. Outside a reply to a wake-word command, that speech is
treated like any other: it is dropped unless it starts with the wake
phrase.

**Barge-in** (`barge_in = true`) applies during a reply to a wake-word
command, so you can talk over the reply. After half a second of speech,
and every second after that, the speech is transcribed. It interrupts
the reply when at least two of its words, and at least half of them,
are not in what the reply just said. That rejects coughs, bumps and the
reply's own echo, and your whole utterance becomes the next command.
Speech over the reply that is never confirmed is dropped. Replies to
typed queries can't be interrupted by voice.

Through headphones this works as is. Through speakers, echo mixed with
your voice is hard to tell apart by text alone, so set up echo
cancellation in the sound server, as described in
[Echo Cancellation](echo-cancellation.md).

## Settings

All keys are under `[voice.wakeword]`. Changes take effect when the
daemon restarts.

| Key                  | Default | Effect                                                                                                                                  |
|----------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `match_similarity`   | `0.8`   | Per-word match needed (0–1). Lower misses fewer wakes but wakes on near-misses; higher, the reverse. Also sets which samples are clear. |
| `command_timeout_ms` | `4000`  | How long to wait for the command after the phrase said alone (500–15000).                                                               |
| `follow_up_ms`       | `8000`  | Follow-up window after a reply (0–30000; `0` turns follow-ups off).                                                                     |
| `silence_ms`         | `800`   | Silence that ends an utterance. Shorter replies sooner but can split a command at a pause; longer waits out pauses.                     |
| `max_utterance_secs` | `30`    | Longest utterance before it is transcribed anyway (1–300).                                                                              |
| `barge_in`           | `true`  | Let speech over a reply to a wake-word command interrupt it.                                                                            |

Prefer re-enrolling over lowering `match_similarity`. Variants accept
the exact spellings Whisper produces for your voice, while a lower
threshold accepts every near-miss for every phrase.

The Whisper model (`voice.transcription.model`) affects accuracy more
than any of these keys. Whisper uses the GPU it shares with the model
(`prefer_gpu`) and switches to the CPU when the model sleeps or another
process crowds the GPU. Constant speech in the room means constant
transcription.
