# MCP servers

Each `[[mcp.servers]]` entry in `~/.config/assistd/config.toml` adds one
[Model Context Protocol](https://modelcontextprotocol.io) server. The
daemon starts or reaches every server in the background, and its tools
are offered to the model as `mcp__<server>__<tool>` from the next turn
after it connects. Each call asks you first until you allow that tool
for good.

```toml
[mcp]
enabled = true

[[mcp.servers]]
name = "filesystem"
command = "npx"                       # a stdio server: a child process
args = ["-y", "@modelcontextprotocol/server-filesystem", "/home/me/docs"]
read = ["/home/me/docs"]              # what its sandbox shows of your files

[[mcp.servers]]
name = "remote"
url = "https://mcp.example.com/mcp"   # a Streamable HTTP server
headers = { Authorization = "Bearer ..." }
```

See the `[mcp]` section of
[config.sample.toml](../config/config.sample.toml) for every key.

The settings panel (`/settings` in the chat, or `assistd settings`)
writes the same entries: on its Tools tab, turn MCP on and open
`servers` to add, change or delete one in a form. Its other
`[[mcp.servers]]` entries keep their comments.

## What a stdio server can reach

Every stdio server runs in a bubblewrap sandbox; there is no way to run
one outside it. A server sees:

- `/usr`, read-only, with the certificates and name resolution files from
  `/etc`, and a `/tmp` of its own;
- a home directory of its own,
  `~/.local/share/assistd/mcp/<name>`, which is its `$HOME` and working
  directory and is kept between launches, so `npx` and `uvx` keep their
  caches there. No server sees another's;
- the host paths its entry grants: `read = [...]` read-only and
  `write = [...]` writable, each at its own path. A leading `~` is your
  real home;
- the network, unless its entry sets `network = false`.

It never sees your real home, the daemon's socket, or assistd's config,
data, state and cache directories, even inside a granted path; a grant
that points into one of them is refused. It cannot open Unix sockets, so
it reaches no agent or session bus. Its environment is `PATH`, the
locale and terminal variables and what `env` sets.

A path that a task watches (a "when this path changes" trigger) is
read-only to every server, even inside a `write` grant, so a server
cannot start the task by writing there. This holds for a run of the
watching task too: its own `write` and `bash` can change the path, a
server it calls cannot. Saving, enabling, disabling or deleting such a
task stops each running server the change affects, along with a call it
is in the middle of; the next call starts it again.

The command must itself be visible: under `/usr`, or under a granted
path. One installed in your home is not, and `assistd mcp status` says
so; grant the directory that holds it and whatever it loads:

```toml
[[mcp.servers]]
name = "notes"
command = "~/.cargo/bin/notes-mcp"    # hypothetical
read = ["~/.cargo/bin", "~/notes"]
```

A server that worked unsandboxed fails until its entry grants what it
reads, and the server's own error in the daemon log is the only hint of
which file that was. What common servers need:

| Server | Grant |
|---|---|
| A filesystem server | Each directory passed in `args`, in `write` (or `read` to keep it read-only). |
| A git server | The repository checkout in `write`, and `~/.gitconfig` in `read`. |
| GitHub, GitLab and other API servers | Nothing; put the token in `env`. One that reads `gh`'s login needs `read = ["~/.config/gh"]`. |
| AWS tools | `read = ["~/.aws"]`, or the keys in `env`. |
| A SQLite server | The database's directory in `write`, since SQLite writes its journal beside the file. |
| `node` from nvm, or a tool from `uv`, `pipx` or `cargo` | `read` for the install, such as `~/.nvm`, `~/.local/share/uv` with `~/.local/bin`, or `~/.cargo/bin`. |
| A server that only computes | `network = false`, once its packages are cached in its home or installed under `/usr`. |

Granting `~` whole works but gives the server everything the sandbox was
meant to keep from it, your SSH and browser files included.

## Checking on servers

```text
assistd mcp status
```

prints one tab-separated row per configured server: its name, its
transport (`stdio` or `http`), and its state: `starting`, `ready`,
`needs authorization`, or `unavailable` with the reason. The chat shows
the same notice for every server that is not usable.

## Servers that sign in with OAuth

A remote server with `auth = "oauth"` sends an OAuth 2.1 bearer token,
which you get by signing in once through the browser:

```text
assistd mcp login <server>
```

The daemon finds the server's authorization server, registers a client
there if the entry has no `client_id`, and listens for the browser's
redirect on a random port of `127.0.0.1`. The command opens the sign-in
page with `xdg-open` and prints its URL, then waits. The browser is
handed a private file that forwards it to that URL, so the URL shows up
in no command line. Once you approve
access, the daemon checks the redirect's `state`, trades its code for a
token, stores the token in `~/.local/state/assistd/mcp-tokens.json`
(mode 0600), and restarts the server; its tools are offered from the
next turn. The redirect has to arrive within five minutes.

The token is refreshed by itself. When the server or its authorization
server stops accepting it, the server shows as `needs authorization`
and its tools fail until you run `assistd mcp login` again.

```text
assistd mcp logout <server>
```

stops the server and deletes its token, first revoking it at the
authorization server when that advertises a revocation endpoint.

**Over SSH.** The redirect goes to `127.0.0.1` on the machine running
the daemon. Open the printed URL in a browser there, or forward the
port named in its `redirect_uri` parameter
(`ssh -L <port>:127.0.0.1:<port> <host>`) before approving.

## Example: Google Drive

Google does not let clients register themselves, so you create one:

1. In the [Google Cloud console](https://console.cloud.google.com),
   create or pick a project and enable the **Google Drive API** for it.
   Enable any MCP-specific API the console lists for Drive as well.
2. Under **APIs & Services → OAuth consent screen**, set the app up for
   external users and add your own Google account as a test user.
3. Under **APIs & Services → Credentials**, create an **OAuth client
   ID** of type **Desktop app**. Desktop clients accept a redirect to
   any port on `127.0.0.1`, which the login flow needs. Copy the client
   ID and client secret.
4. Add the server:

   ```toml
   [[mcp.servers]]
   name = "drive"
   url = "https://drivemcp.googleapis.com/mcp/v1"
   auth = "oauth"
   client_id = "1234-abcd.apps.googleusercontent.com"
   client_secret = "GOCSPX-..."
   scopes = ["https://www.googleapis.com/auth/drive.readonly"]
   ```

   Keep `config.toml` mode `0600`, since it now holds the secret.
5. Restart the daemon, then sign in:

   ```text
   assistd mcp login drive
   ```

   `assistd mcp status` then lists `drive` as `ready`, and the model is
   offered its tools as `mcp__drive__<tool>`.
