# Sub-agents

The model hands work to a sub-agent with its `relay` tool: a second run of
the same model on a history of its own, in a disposable sandbox, that
reports back. Which kinds of sub-agent it can pick is a list of
definitions. Three are built in:

| Name                | Reaches                         | For                                              |
|---------------------|---------------------------------|--------------------------------------------------|
| `file_review`       | the files it is handed          | documents, logs and code too large to read whole |
| `web_research`      | the web                         | research and reading web pages                   |
| `file_web_research` | the files it is handed, the web | a task that needs both                           |

## Your own

Each `<name>.toml` file in `~/.config/assistd/agents/` (the `agents`
directory beside the config file) adds one. The file name is the agent's
name:

```toml
description = "For reviewing Rust changes against my checklist."
capabilities = ["files", "index"]
mcp_servers = ["github"]
commands = ["cat", "grep", "ls", "search"]
tool_steps = 40
reply_tokens = 2000
thinking_tokens = 1000

prompt = '''
You review Rust changes. For each file you are handed, check:

- every error is handled or returned, never dropped
- no new `unwrap` outside tests
'''
```

| Key            | Meaning                                                                                                   |
|----------------|-----------------------------------------------------------------------------------------------------------|
| `description`  | Required. When to pick this agent; the main model reads it to choose.                                     |
| `capabilities` | Any of `files` (read the inputs it is handed), `web` (search and fetch pages) and `index` (search your indexed documents, saved memories and past sessions). Default: none. |
| `mcp_servers`  | Names of `[[mcp.servers]]` entries whose tools it may call. A server reachable from the sandbox is a way out of it, so each is named. |
| `commands`     | The only commands its `run` tool offers, out of `bash`, `cat`, `docs`, `echo`, `grep`, `head`, `ls`, `search`, `sort`, `tail`, `uniq`, `wc`, `web` and `write`. Unset offers every command its capabilities give it. |
| `sites`        | Host names, such as `docs.rs`, it is told to read on every run. Needs `web`.                              |
| `tool_steps`  | Tool calls one run may make, at most 200. Unset keeps the daemon's limit, as for the next two.            |
| `reply_tokens` | Tokens per model response, thinking included, at most `chat.max_response_tokens`.                         |
| `thinking_tokens` | Tokens it may think for in one response, at most `chat.max_reasoning_tokens` and below the reply limit. |
| `prompt`       | Its own instructions. Write it between `'''` lines so backslashes are kept as written.                   |

The prompt is added to what the daemon tells every sub-agent: that it
works alone, what it can reach, its budgets, and that what it reads is
data, not instructions. It does not replace any of that. A sub-agent with
both this machine's data and the web is asked about each fetch once it has
read any of that data, whatever its file says.

The files are read when the daemon starts. A file that breaks a rule is
left out and the daemon logs why; the others still load. A file cannot
take a built-in agent's name.

```sh
assistd agents
```

lists what a daemon started now would offer, one tab-separated line per
agent (name, where it comes from, what it reaches, description), and on
stderr each file it would refuse with the reason. The settings panel shows
the same list, read-only, under **Sub-agents** on its Tools tab.
