# AUR packaging for assistd

Two PKGBUILDs live here:

| Directory       | Source                  | When to use                          |
|-----------------|-------------------------|--------------------------------------|
| `assistd/`      | `v$pkgver` GitHub tag   | Stable release.                      |
| `assistd-git/`  | `main` branch HEAD      | Bleeding edge.                       |

`assistd-git` provides and conflicts with `assistd`, so the two cannot
be installed at the same time.

## Build and test locally

From the workspace root:

```bash
cd dist/aur/assistd-git
makepkg -si                      # build, install missing deps (-s), install the package (-i)
```

`makepkg` produces `assistd-git-<version>-x86_64.pkg.tar.zst`. Inspect it
without installing via `tar -tvf <pkg>` or `bsdtar -tvf <pkg>`.

After install, check the packaged files and the user unit:

```bash
pacman -Ql assistd-git | grep -E '(/usr/bin/assistd$|/etc/assistd/config.toml$|/usr/lib/systemd/user/assistd.service$|/usr/share/licenses/assistd/LICENSE$)'
/usr/bin/assistd --version
systemctl --user daemon-reload
systemctl --user start assistd
systemctl --user status assistd
```

(The unit may fail at runtime if `llama-server` is missing; what
matters here is that it loads and starts.)

## Lint with namcap

```bash
pacman -S namcap
namcap PKGBUILD
namcap *.pkg.tar.zst
```

Aim for zero E (error) and W (warning) entries. Some informational notes
(`I:`) about depend hints or stripping are acceptable.

## Verify CPU portability

The PKGBUILD forces `GGML_NATIVE=OFF` and enables AVX/AVX2/FMA, so the
binary runs on any x86-64-v3 CPU, not just the build host. Confirm with:

```bash
objdump -d /usr/bin/assistd | grep -c vfmadd       # FMA present
objdump -d /usr/bin/assistd | grep -cE '%zmm[0-9]+' # AVX-512 absent (should be 0)
```

## Releasing `assistd`

1. Merge a PR that bumps `version` in the root `Cargo.toml` (and
   `Cargo.lock`). Leave `dist/aur/assistd/` alone.
2. Tag the merge commit and push the tag:
   `git tag -a v1.2.0 -m v1.2.0 && git push origin v1.2.0`.
3. The tag push starts the **AUR release** workflow. It stamps
   `pkgver`, fills in the checksum, regenerates `.SRCINFO`, test-builds
   the package in an Arch container and pushes it to the AUR after
   approval. For a packaging-only fix to a release that is already
   out, run it from the Actions tab with the same tag and a higher
   `pkgrel`.

## Publishing `assistd-git`

The workflow does not touch `assistd-git`. Its `pkgver()` tracks
`main`, so it needs a new push only when the PKGBUILD itself changes
(bump `pkgrel=` then). One-time setup: create an account at
<https://aur.archlinux.org/>, add your SSH public key, and test with
`ssh aur@aur.archlinux.org help`. Then:

```bash
git clone ssh://aur@aur.archlinux.org/assistd-git.git /tmp/aur-assistd-git
cd /tmp/aur-assistd-git
cp -v /path/to/assistd/dist/aur/assistd-git/{PKGBUILD,assistd.install} .
makepkg --printsrcinfo > .SRCINFO   # never hand-edit
git add PKGBUILD .SRCINFO assistd.install
git commit -m "Update PKGBUILD"
git push origin master
```

Users pick up new commits with their AUR helper's devel check (for
example `yay -Syu --devel`).
