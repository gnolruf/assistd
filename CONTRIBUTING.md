# Contributing

Code rules and the required local checks live in [AGENTS.md](AGENTS.md).
This file covers the workflow around them.

## Issues

Open issues with the bug or feature form. Title bugs with the symptom
("Chat TUI panics on a link containing a line break"), not the fix.
Security problems go through [SECURITY.md](SECURITY.md), never a public
issue.

## Branches

Every branch starts from an issue. Use GitHub's **Create a branch**
button on the issue, which names it `<issue>-<title-slug>` and links it
to the issue. Branches are deleted when their pull request merges.

## Pull requests

- Title the PR in [Conventional Commits](https://www.conventionalcommits.org/)
  form: `fix(tools): ask for confirmation when xargs fills in the command`.
  Types are `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `ci`,
  `build` and `chore`; the scope is the crate name without `assistd-`.
- Put `Closes #<issue>` in the description.
- PRs are squash-merged, so the PR title becomes the commit on `main`.
- CI must pass before merging.

## Releases

1. Open a PR that bumps `version` in the root `Cargo.toml` and
   `pkgver` in `dist/aur/assistd/PKGBUILD`, and merge it.
2. Tag the merge commit and push the tag:
   `git tag -a v1.2.0 -m v1.2.0 && git push origin v1.2.0`.
3. Run the **AUR release** workflow from the Actions tab with that tag.
   It fills in the checksum, regenerates `.SRCINFO`, test-builds the
   package in an Arch container and pushes it to the AUR after approval.
   For a packaging-only fix to a release that is already out, run it again
   with the same tag and a higher `pkgrel`.
