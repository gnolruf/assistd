# AUR packaging for assistd

Two PKGBUILDs live here:

| Directory       | Source                  | When to use                                          |
|-----------------|-------------------------|------------------------------------------------------|
| `assistd/`      | `v$pkgver` GitHub tag   | Stable release. Requires the matching git tag to exist. |
| `assistd-git/`  | `main` branch HEAD      | Bleeding edge. Publishable today (no tag required).  |

`assistd-git` declares `provides=('assistd')` and `conflicts=('assistd')`,
so the two cannot be installed at the same time.

## Build and test locally

From the workspace root:

```bash
cd dist/aur/assistd-git          # or dist/aur/assistd once v0.1.0 is tagged
makepkg -si                      # build, install, and (with -i) install
```

`makepkg` produces `assistd-git-<version>-x86_64.pkg.tar.zst`. Inspect it
without installing via `tar -tvf <pkg>` or `bsdtar -tvf <pkg>`.

After install, verify acceptance criteria:

```bash
pacman -Ql assistd-git | grep -E '(/usr/bin/assistd$|/etc/assistd/config.toml$|/usr/lib/systemd/user/assistd.service$|/usr/share/licenses/assistd/LICENSE$)'
/usr/bin/assistd --version
systemctl --user daemon-reload
systemctl --user start assistd
systemctl --user status assistd
```

(The unit may fail at runtime if `llama-server` is missing — that's
expected; the criterion is that the unit loads and is *started*.)

## Lint with namcap

`namcap` is required for AUR review and is not installed by default:

```bash
pacman -S namcap
namcap PKGBUILD
namcap ../../*-x86_64.pkg.tar.zst   # against the built package
```

Aim for zero E (error) and W (warning) entries. Some informational notes
(`I:`) about depend hints or stripping are acceptable.

## Verify CPU portability

The PKGBUILD forces `GGML_NATIVE=OFF` and enables AVX/AVX2/FMA, matching
Arch's official `x86-64-v3` baseline. Confirm with:

```bash
objdump -d /usr/bin/assistd | grep -cE 'vfmadd|vmulps'   # AVX2/FMA present
objdump -d /usr/bin/assistd | grep -cE '%zmm[0-9]+'      # AVX-512 absent (should be 0)
```

## Publishing to the AUR

One-time setup:

1. Create an account at <https://aur.archlinux.org/>.
2. Add your SSH public key in the AUR account settings.
3. Test with `ssh aur@aur.archlinux.org help`.

Per-package, first publish:

```bash
# Clone the (empty) AUR repo for the package
git clone ssh://aur@aur.archlinux.org/assistd-git.git /tmp/aur-assistd-git
cd /tmp/aur-assistd-git

# Copy the source-controlled files from this repo
cp -v /path/to/assistd/dist/aur/assistd-git/{PKGBUILD,assistd.install} .

# Generate .SRCINFO (AUR requires it; never hand-edit)
makepkg --printsrcinfo > .SRCINFO

# Commit and push
git add PKGBUILD .SRCINFO assistd.install
git commit -m "Initial upload: assistd-git 0.1.0.rNN.gXXXXXXX-1"
git push origin master
```

Repeat for `assistd` after tagging `v0.1.0`:

```bash
cd /path/to/assistd
git tag v0.1.0
git push origin v0.1.0

# Then update the assistd PKGBUILD's sha256sums and publish:
cd dist/aur/assistd
updpkgsums                       # writes the real sha256sum
makepkg --printsrcinfo > .SRCINFO
# clone aur:assistd.git, copy, commit, push (same flow as above)
```

## Version bumps

For `assistd` (tagged releases):

1. Bump `pkgver=` in `dist/aur/assistd/PKGBUILD`. Reset `pkgrel=1`.
2. Tag the release upstream: `git tag vX.Y.Z && git push --tags`.
3. `updpkgsums` to refresh `sha256sums`.
4. `makepkg --printsrcinfo > .SRCINFO`.
5. Commit, push to `aur:assistd.git`.

For `assistd-git` (continuous):

1. AUR rebuilds automatically when users run `yay -Syu --aur` and a new
   `pkgver()` is computed — no version bump is needed for code changes.
2. Bump `pkgrel=` only when the PKGBUILD itself changes (deps, build
   logic, install scriptlet).
3. Regenerate `.SRCINFO` and push.

## Why two packages instead of one

The AUR's `-git` convention separates stable releases from rolling
HEAD builds. `yay -S assistd` always pins to a known tag; `yay -S
assistd-git` always tracks `main`. Users opt in deliberately. Splitting
also lets us publish today (via `-git`) before `v0.1.0` is tagged.
