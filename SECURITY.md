# Security policy

## Reporting a vulnerability

Report vulnerabilities privately through
[GitHub's private vulnerability reporting](https://github.com/gnolruf/assistd/security/advisories/new).
Do not open a public issue or pull request for them.

In scope is anything that lets the model, an MCP server, a web page, or
another local user do something assistd is meant to stop, for example:

- running a command without the confirmation prompt
- escaping the bubblewrap sandbox or the write allowlist
- reaching the daemon socket from another user account
- crashing or wedging the daemon from model output or IPC input

Include the version (`assistd --version`), a reproduction, and what an
attacker gains. Expect an acknowledgement within a week. Fixes are
developed in a private fork attached to the advisory and disclosed when
the patched release is out.

## Supported versions

Only the latest release and `main` receive security fixes.
