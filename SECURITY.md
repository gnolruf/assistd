# Security policy

## Reporting a vulnerability

Report vulnerabilities privately through
[GitHub's private vulnerability reporting](https://github.com/gnolruf/assistd/security/advisories/new).
Do not open a public issue or pull request for them.

In scope is anything that lets the model, an MCP server, a web page, or
another local user do something assistd is meant to stop, for example:

- running a command the permission mode should have asked about
- escaping the bubblewrap sandbox or the write allowlist
- reaching the daemon socket from another user account
- crashing or wedging the daemon from model output or IPC input

Include the version (`assistd --version`), a reproduction, and what an
attacker gains. Fixes are developed in a private fork attached to the advisory and disclosed when
the patched release is out.

## How commands are gated

`bash` scripts pass through layers, each a floor for the next:

- **Classifier.** Sorts a script as read-only, an edit of the literal
  paths it names, or unvetted, using short lists of programs whose
  effects are known. Anything it cannot vet is unvetted. A program the
  user approved with "always allow" counts as vetted for the runs the
  approval covers.
- **Path guards.** Resolve every target: destroying `$HOME`, a personal
  folder or a git work tree is refused; touching credentials, owner-only
  files, startup files, shell profiles or `PATH` directories always
  asks.
- **Permission modes.** Manual asks for every edit and unvetted script;
  Accept-edits also runs edits inside `tools.write.writable_paths`; the
  opt-in Auto puts the rest to the safety judge.
- **Safety judge.** A one-shot model verdict in Auto mode; it asks the
  user whenever it is unsure, fails or times out.
- **Sandbox.** Every command runs under bubblewrap, with a read-only
  root, no network unless configured, and assistd's own config
  read-only. Without `bwrap` no tools are offered.

## Supported versions

Only the latest release and `main` receive security fixes.
