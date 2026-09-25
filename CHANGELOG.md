# Changelog

All notable changes to assistd are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-09-25

### Added

- Program allowlist for spawned commands: `bash` scripts and `wm open` run
  without asking only when every program they can run is listed in
  `tools.bash.allowed_programs` and nothing matches a destructive pattern.
  Anything else prompts, and the prompt offers "always allow", persisted to
  `allowed_programs.toml` beside `config.toml` (#295).
- New `run` commands `head`, `tail`, `sort` and `uniq`; `grep -n`/`-r` over
  multiple files, `cat -n`, `echo -n`/`-e`, `wc -w`/`-c`, `ls -a`; `~` and
  glob expansion in unquoted arguments; `--help` on every command (#209,
  #216).
- `head`, `tail`, `sort`, `uniq` and `wc` accept `FILE...` arguments, so
  pipelines no longer need a leading `cat` (#216).
- `ls` on a file or symlink prints a single type/size/path row instead of
  failing, and path errors name the parent directory (#233).
- Session titles: the daemon emits a `SessionTitle` event and the chat TUI
  shows it in the status bar, updating on `/switch`, `/resume` and `/new`
  (#216).
- Daemon startup can be interrupted with Ctrl-C or SIGTERM; a second signal
  forces an immediate exit (#267).
- `assistd tray` starts even when the config fails to load and shows the
  error in its icon and tooltip instead of exiting (#242).
- CodeQL scanning for Rust and GitHub Actions.

### Changed

- Config schema simplified: every section and key is optional, dead and
  redundant keys were removed, `[[mcp.servers]]` entries are discriminated by
  `transport`, and `0` is a parse error for ports, timeouts and token
  budgets. See "Upgrading from 1.0.0" below (#218).
- Unknown config keys are ignored with a per-key warning instead of being
  accepted silently (#218, #297).
- `tools.bash.destructive_patterns` is now a structured matcher: a command
  name plus required arguments in any order, `a|b` alternatives, short-flag
  clusters, `--long` abbreviations and `key=` prefixes. The default list was
  rewritten. `denylist` entries ending in punctuation must end a shell word
  (#295).
- Tool results are sent to the model as `role: "tool"` messages carrying
  their `tool_call_id`. `agent.max_iterations` is replaced by fixed loop
  guards (three identical consecutive calls, or 200 tool steps) that emit a
  `tools_withdrawn` warning (#216).
- Per-turn context (recalled memory, desktop state) is attached to the user
  turn and the conversation summary is merged into the single leading system
  message, so llama-server's prefix cache is no longer defeated and chat
  templates cannot drop the summary (#230, #232).
- The system prompt is sent exactly as configured; the auto-appended tool
  roster is gone because tools already travel as request schemas (#229).
- `chat.request_timeout_secs` bounds only the wait for the first streamed
  byte. Generation length is unbounded; only the inter-chunk inactivity gap
  applies afterwards (#207, #265).
- `wm open` runs under the full `[tools.bash]` policy and the bubblewrap
  sandbox, with `$XDG_RUNTIME_DIR` bound for GUI apps, and reports an app
  that exits during its startup probe (#194).
- Confirmation modal: Enter no longer approves, `y` is ignored for 750 ms
  after opening, `a` answers "always allow", multi-line scripts are shown
  line by line with control characters escaped, and unanswered prompts deny
  after 120 s (#243, #295).
- The `run` tool description states the exact truncation thresholds from
  `[tools.output]` and lists `2>`, `2>&1` and here-docs as unsupported
  (#234, #249).
- `remember`/`recall` keys may contain hyphens; `reminisce` searches only
  earlier sessions (#216).
- When tools are withdrawn mid-turn the schemas stay in the request and the
  note goes at the tail, so a model that ignores it no longer streams raw
  tool-call markup as its answer (#228).
- MCP servers that die right after initialising count as failures and back
  off; a rolling cap of 10 restarts per 600 s marks a server unhealthy
  (#212).
- Continuous listening transcribes utterances through one ordered worker
  with a bounded backlog instead of spawning a Whisper job per utterance
  (#269).
- The Piper TTS circuit breaker re-arms after 60 s with a trial utterance
  instead of staying tripped until restart (#196).
- IPC clients skip events they cannot decode, so older clients keep working
  against newer daemons (#293).
- Connections close right after the terminal event, and shutdown no longer
  always waits out `daemon.shutdown_grace_secs` (#193).
- Performance: Whisper state is reused across transcriptions, embeddings are
  batched per request, the TUI wraps incrementally and coalesces redraws,
  images are base64-encoded once, vision capability is re-probed only after
  a reload, and bus events are built only when a subscriber wants them
  (#291).
- Build: MSRV is 1.92; client-only builds no longer pull in SQLite, reqwest
  or the window-manager backends; i3/sway support is a default-on feature of
  the binary (#270, #291).

### Fixed

- Destructive-command confirmation never reached the client and silently
  denied everything (#243).
- Streamed replies longer than 120 s were cut off (#265), and a long prompt
  prefill was reported as a stalled stream (#207).
- Resumed conversations were rebuilt one message per database row, splitting
  multi-call steps (#231).
- Interrupting a turn could not stop a hung tool (#200). A tool call that
  never returns is abandoned after 300 s, and subprocesses that never read
  stdin or leave pipes open no longer hang the call (#260).
- `bash` timeout and output overflow killed only the shell, orphaning
  grandchildren that kept the pipe open (#205).
- A failed wake left a llama-server the daemon could never stop (#199);
  SIGKILL now follows a SIGTERM that times out (#249).
- `cat`, `grep`, `wc` and `see` on a device, FIFO or socket such as
  `/dev/stdin` blocked the turn forever (#249).
- Piper playback stalled the daemon; speech still playing is skipped when a
  new query starts (#249).
- TTS crashed on CJK/emoji-heavy replies and on multi-byte whitespace at the
  length cutoff (#198, #245).
- TUI: a failed branch command locked out `/fork`, `/switch`, `/undo` and
  `/new` for the session, a status poll could end a streaming reply, and
  push-to-talk text was spliced into a running reply (#210); tab-completion
  and small-pane panics (#266).
- MCP: a stdio server whose read loop died stayed unusable until the child
  exited (#211); the SSE transport rejected relative `endpoint` URLs and
  killed its own stream every 30 s (#204); timed-out requests leaked
  in-flight slots until every call failed (#268).
- Embedding supervisor integer underflow on restart with zero failures
  (#247).
- The default `embedding.model` was a `.gguf` filename llama-server could not
  resolve, so an omitted `[embedding]` section spawned a server that never
  came up (#218).
- Session-title generation could deadlock; it now runs only after a
  successful turn (#207, #209).

### Security

- Command gate bypass: unlisted programs, absolute paths such as `/bin/rm`,
  reordered flags, `$(...)`, `eval`, `PATH`/`LD_*` changes and script files
  all prompt now, and unsandboxed commands run with a fixed absolute `PATH`
  (#295).
- The destructive-pattern check could be evaded with newlines, comments,
  wrappers (`sudo`, `env`, `xargs`, `timeout`, ...), leading assignments or
  letter case (#262).
- `wm open` was an ungated, unsandboxed spawn path around the whole bash
  policy (#194).
- `write` could escape `writable_paths` through a dangling symlink; files are
  now opened with `O_NOFOLLOW` (#264).
- With `HOME` unset the sandbox bound `/` writable (#263).
- `write` refuses assistd's own config directory, the sandbox mounts it
  read-only, and the daemon socket is hidden from sandboxed commands so a
  command cannot answer its own prompt (#295).
- Focused-window class and title are sanitised, length-capped and marked as
  untrusted before entering the prompt (#248).
- Local denial of service: an out-of-range `recency_secs` crashed the daemon
  (#246), a huge semantic-search `limit` forced a giant allocation (#261), a
  stalled client blocked all queries for 600 s (#244), and MCP stdout lines
  and SSE events are capped at 1 MiB (#211, #268).
- The `write` policy could fail open on an empty allowlist; that state is now
  unrepresentable (#201).
- The CI workflow token is restricted to `contents: read` (#221).

### Dependencies

- nvml-wrapper 0.13, rubato 5, cpal 0.18, eframe 0.35, base64 0.23,
  infer 0.22, plus minor and patch bumps; GitHub Actions checkout 7,
  codeql-action 4, setup-rust-toolchain 2.

### Upgrading from 1.0.0

Stale keys do not break startup; each one logs a warning until removed.

Keys that are no longer read:

- `[remote]` (whole section), `[agent] max_iterations`, `[timeouts]` (whole
  section).
- `sleep.suspend`, `memory.retention_days`, `chat.max_summary_tokens` (now
  1.2 x `summary_target_tokens`).
- `embedding.ready_timeout_secs` (use `llama_server.ready_timeout_secs`),
  `embedding.request_timeout_secs`, `embedding.chunk_chars`,
  `embedding.chunk_overlap_chars`.
- `voice.transcription.vad_silence_secs`, `gpu_busy_timeout_ms`,
  `cpu_fallback_enabled`; `voice.continuous.min_utterance_ms`, `preroll_ms`,
  `onset_confirm_ms`, `aggressiveness`; `voice.synthesis.noise_scale`,
  `noise_w`, `sentence_silence_secs`.
- `tools.screenshot.timeout_secs`; `mcp.servers[].sse_read_timeout_secs`,
  `sse_ping_interval_secs`.
- `tray.icon_*`, `tray.popup.listen_auto_hide_ms` (now 3 x `auto_hide_ms`),
  `tray.popup.truncate_chars`.

Keys to review:

- `tools.bash.destructive_patterns`: if you copied the 1.0.0 default list,
  replace it with the new default or delete the key. Bare `shutdown` and
  `reboot` entries are gone; those programs are not on the allowlist and
  always prompt.
- `tools.bash.allowed_programs`: programs not on the list prompt. `git` is
  not on the default list; approve it once with "always allow" if you want
  it silent.
- `chat.request_timeout_secs`: raise it for slow prompt prefill, not for
  long answers.
- `embedding.model`: the `:` suffix must be a quant tag such as
  `nomic-ai/nomic-embed-text-v1.5-GGUF:Q4_K_M`, not a `.gguf` filename.
- `[[mcp.servers]]`: `url` and `headers` are valid only with
  `transport = "sse"`, `command`, `args` and `env` only with `"stdio"`.

## [1.0.0] - 2026-05-28

Initial release.

[1.1.0]: https://github.com/gnolruf/assistd/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/gnolruf/assistd/releases/tag/v1.0.0
