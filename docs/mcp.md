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

[[mcp.servers]]
name = "remote"
url = "https://mcp.example.com/mcp"   # a Streamable HTTP server
headers = { Authorization = "Bearer ..." }
```

See the `[mcp]` section of
[config.sample.toml](../config/config.sample.toml) for every key.

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
page with `xdg-open` and prints its URL, then waits. Once you approve
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
