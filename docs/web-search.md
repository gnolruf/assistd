# Web search

The `web` command has two modes. `web fetch <URL>` reads a page and
always works. `web search <query>` needs a local
[SearXNG](https://docs.searxng.org) instance, which the daemon starts,
supervises and configures itself. You only need to install SearXNG
and turn search on.

```text
web search [-n N] [--lang <tag>] <query...>   # TSV hits: rank, title, url, snippet
web fetch [--raw | --full] <URL>              # page text, or the body as sent
```

The model usually searches, picks a hit, and then fetches it. Fetching
from a host you haven't approved yet asks you first. Search queries
never ask, because they only go to the local instance.

## Setup

1. **Install SearXNG** so that `searxng-run` is on `$PATH`.

   - Arch: `searxng-git` from the AUR (an optional dependency of the
     `assistd` packages).
   - Elsewhere, install it into a virtualenv as the
     [upstream guide](https://docs.searxng.org/admin/installation-scripts.html)
     describes, then point `search.command` at that venv's
     `searxng-run` (see [Manual install](#manual-install)).

2. **Enable it** in `~/.config/assistd/config.toml`:

   ```toml
   [search]
   enabled = true
   ```

3. **Restart the daemon.** SearXNG starts straight away, alongside the
   model rather than after it. `assistd chat` names it while it is
   still starting, and the tray tooltip lists it as `web search: ready`
   once it is up.

The defaults search Bing and Yahoo. To use other engines, list them by
the names SearXNG's `settings.yml` uses. Only the engines you list are
loaded:

```toml
[search]
enabled = true
engines = ["duckduckgo", "wikipedia"]
```

An engine that needs an API key takes a table. The key lives in its
own file, read again at every launch, so it never goes in the config:

```toml
engines = [{ name = "braveapi", api_key_file = "~/.config/assistd/brave_api_key" }]
```

The [sample config](../config/config.sample.toml) lists every
`[search]` key.

## What the daemon's instance is

The daemon writes its own `settings.yml` under
`$XDG_RUNTIME_DIR/assistd/searxng/` at each launch and runs SearXNG on
`127.0.0.1:8387` with a fresh secret. It ignores
`/etc/searxng/settings.yml`, any `SEARXNG_*` variables in your
environment, and the `searxng.service` unit some packages install.
You don't need to enable that service, and running it doesn't affect
the daemon's instance unless it is set to the same port.

The instance has no authentication, so it only binds loopback.
`web fetch` refuses loopback addresses, so the model can only reach
SearXNG through `web search`.

## Manual install

`search.command` is split like a shell command line but never run
through a shell, so `~` and `$VAR` are not expanded. Give an absolute
path:

```toml
[search]
enabled = true
command = "/home/me/src/searxng/.venv/bin/searxng-run"
# or: command = "/home/me/src/searxng/.venv/bin/python -m searx.webapp"
```

Whatever you run must stay in the foreground, honour the
`SEARXNG_SETTINGS_PATH`, `SEARXNG_BIND_ADDRESS`, `SEARXNG_PORT` and
`SEARXNG_SECRET` variables the daemon sets, and answer `GET /healthz`.
`searxng-run` does all of this. A uWSGI or Granian setup needs a small
wrapper script that does too.

The listening socket must also belong to a process the daemon started:
a `200` from `/healthz` only counts if a process in the command's
process group holds the socket on `host:port`, so another program
answering on that port can't pass as SearXNG. This rules out docker
and podman. With port publishing the socket belongs to `docker-proxy`
or `rootlessport`, and with host networking it belongs to a process
under the container runtime, outside that process group.
Either way the instance never becomes ready and search degrades
after `ready_timeout_secs`. Run SearXNG directly from a package or a
venv instead.

## Troubleshooting

Check `assistd chat` or the tray tooltip for the `web search` line,
and the daemon log (`journalctl --user -u assistd`) for the line
starting `web search:`. When search is unavailable, the model is told
why and falls back to fetching URLs it already knows.

**`web search: unavailable (disabled in config (search.enabled = false))`.**
Search is off by default. Set `[search] enabled = true`.

**`web search: unavailable (cannot run search.command `searxng-run`: …)`.**
SearXNG isn't installed, or isn't on the daemon's `$PATH`. A systemd
user service gets the user manager's `PATH`, not your shell's, so a
venv that your shell profile adds to `PATH` isn't visible to it.
Install `searxng-git` or set `search.command` to an absolute path.

**`web search: unavailable (server failed to start: search-server startup failed after … attempts)`.**
SearXNG started but never became healthy. The SearXNG output just
above it in the log says why. The most common cause is that the port
is taken (`Address already in use`). Find the owner with
`ss -ltnp 'sport = :8387'` and either stop it or move the daemon's
instance with `[search] port = …`. The port must also differ from
`model.port` and `embedding.port`; config validation enforces this.
A slow first start (compiling, a cold venv) may need a larger
`ready_timeout_secs`. If `search.command` starts a container, it never
becomes ready; see [Manual install](#manual-install).

**`web search: unavailable (failed to read the <engine> API key from …)`**
or **`(the <engine> API key file … is empty)`.** Fix or fill that
engine's `api_key_file`.

**Searches return nothing.** An engine name SearXNG doesn't recognise
is logged by SearXNG and skipped. Check the names against its
`settings.yml`. Engines that scrape result pages can also be
rate-limited or blocked upstream for a while, so listing more than
one helps.
