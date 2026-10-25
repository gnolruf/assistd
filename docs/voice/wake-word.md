# Wake-Word Listening

With wake-word listening on, `assistd` keeps the mic open and acts on
speech that contains the wake phrase: "hey computer, what's on my
calendar" is sent as a query, and anything else said in the room is
dropped. The phrase is "hey computer" unless you set your own in
config. Everything runs locally.

## Turning it on

```toml
[voice]
enabled = true

[voice.wakeword]
enabled = true
start_on_launch = true      # otherwise it waits for listen-toggle
# phrase = "hey computer"

[hotkeys]
# toggle_listening = "Super+Shift+L"  # Sway, i3 and X11; elsewhere bind `assistd listen-toggle`
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
keys above must be `true`. Otherwise the daemon log says why listening
is unavailable: the Silero VAD model (`voice.transcription.vad_model`)
or the keyword model could not be downloaded or read, or the keyword
model cannot spell the phrase.

### How it works

Silero VAD cuts the mic stream into utterances. A small keyword-spotting
model (sherpa-onnx's 3.3M-parameter GigaSpeech zipformer, on one CPU
thread) checks each utterance for the phrase. Only when it finds it is
the speech after the phrase transcribed (Parakeet) and sent as the
command. If nothing follows the phrase, the next utterance is the
command.

### Privacy

No audio or text leaves the machine. The models are downloaded from
Hugging Face once, on first use. After that, all audio is processed
locally.

Utterances without the wake phrase are never transcribed. They are
checked by the keyword spotter and discarded. The exceptions are the
follow-up window after a reply and speech over a reply (barge-in),
both described below, which are transcribed so they can be acted on.

## Choosing a wake phrase

Set `voice.wakeword.phrase` in `config.toml` or in `/settings`, then
restart the daemon. A phrase has one to six words of letters and
apostrophes; case doesn't matter. The keyword model spells words out of
its own pieces, so English letters work and some accented or
non-Latin letters don't. The daemon refuses a phrase it can't spell at
startup and logs which piece is missing.

Longer phrases wake by accident less often. Also avoid:

- common words, especially at the start ("okay", "so", "hey there");
- phrases you would say to a person in the room.

## Testing: `/wakeword`

Open `assistd chat` and type `/wakeword`. The panel shows the phrase
and whether listening is on. Press Enter to test: the panel listens and
shows, for each utterance, whether the phrase was heard.

```text
✓ heard the wake word: I'd wake
✗ heard speech without the wake word
```

The test takes over the mic while it runs: listening pauses and then
resumes, and push-to-talk is refused until you finish. Esc stops it.
A test with no speech for two minutes stops itself.

If the phrase is missed often, lower `threshold`. If other speech
wakes it, raise `threshold` or choose a longer phrase. The test checks
only the phrase. Command windows, follow-ups and barge-in are not part
of it.

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
because the model has to load. If the GPU lacks room for the model
(`sleep.gpu_headroom_mb`), the wake is refused and the command fails
with the reason.

### Hearing its own replies

The mic stays open while replies are spoken, so the daemon hears its
own speech. Outside a reply to a wake-word command, that speech is
treated like any other: it is dropped unless it holds the wake phrase.

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

| Key                  | Default          | Effect                                                                                                              |
|----------------------|------------------|---------------------------------------------------------------------------------------------------------------------|
| `phrase`             | `"hey computer"` | The wake phrase: one to six words.                                                                                  |
| `threshold`          | `0.25`           | How sure the keyword spotter must be (0–1). Lower misses fewer wakes but wakes on near-misses; higher, the reverse. |
| `command_timeout_ms` | `4000`           | How long to wait for the command after the phrase said alone (500–15000).                                           |
| `follow_up_ms`       | `8000`           | Follow-up window after a reply (0–30000; `0` turns follow-ups off).                                                 |
| `silence_ms`         | `800`            | Silence that ends an utterance (at most 1400). Shorter replies sooner but can split a command at a pause.           |
| `max_utterance_secs` | `30`             | Longest utterance before it is handled anyway (1–300).                                                              |
| `barge_in`           | `true`           | Let speech over a reply to a wake-word command interrupt it.                                                        |
