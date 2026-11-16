//! Path guards: sorts where a file operation lands into catastrophic, sensitive or ordinary,
//! from a short fixed list of trees in `$HOME`, the XDG directories and the systemd unit path,
//! and from which files only their owner may read.

use std::ffi::OsStr;
use std::fmt;
use std::fs::FileType;
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use assistd_utils::path::{expand_tilde, lexical_clean, tilde_remainder};

use super::review::{Action, Analysis, Class, Effect};

/// Credentials and identity, below `$HOME`.
const HOME_CREDENTIALS: &[&str] = &[".gnupg", ".netrc", ".ssh"];

/// Keyrings, below each data directory.
const DATA_CREDENTIALS: &[&str] = &["keyrings", "kwalletd"];

/// What starts programs at login, below each config directory.
const CONFIG_STARTUP: &[&str] = &["autostart", "environment.d", "systemd/user"];

/// User units, below each data directory, for when systemd cannot be asked.
const DATA_STARTUP: &[&str] = &["systemd/user"];

/// Startup files bash and zsh read, below `$HOME`.
const HOME_SHELL_PROFILES: &[&str] = &[
    ".bash_login",
    ".bash_logout",
    ".bash_profile",
    ".bashrc",
    ".profile",
    ".zlogin",
    ".zlogout",
    ".zprofile",
    ".zshenv",
    ".zshrc",
];

/// Startup files fish reads, below each config directory.
const CONFIG_SHELL_PROFILES: &[&str] = &["fish"];

/// Where the XDG user directories are named, below a config directory.
const USER_DIRS_FILE: &str = "user-dirs.dirs";

/// The config directory below `$HOME` when `$XDG_CONFIG_HOME` is unset.
const DEFAULT_CONFIG_HOME: &str = ".config";

/// The data directory below `$HOME` when `$XDG_DATA_HOME` is unset.
const DEFAULT_DATA_HOME: &str = ".local/share";

/// How many entries of a directory read throughout are checked for owner-only files.
const MAX_SCANNED_ENTRIES: usize = 1024;

/// How many entries of a copied or moved tree are checked for symlinks and hidden entries; a
/// tree holding more is not vetted.
const MAX_CARRIED_ENTRIES: usize = 4096;

/// The personal folders, each with its `user-dirs.dirs` key and the name used without one.
const PERSONAL_FOLDERS: &[(&str, &str)] = &[
    ("XDG_DESKTOP_DIR", "Desktop"),
    ("XDG_DOCUMENTS_DIR", "Documents"),
    ("XDG_MUSIC_DIR", "Music"),
    ("XDG_PICTURES_DIR", "Pictures"),
    ("XDG_VIDEOS_DIR", "Videos"),
];

/// What the path guards make of a script's effects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathVerdict {
    /// An effect destroys `$HOME`, a personal folder or a git work tree, which is refused in
    /// every mode; holds what and where, for the error line.
    Refused(String),
    /// An effect touches credentials, an owner-only file in `$HOME`, a startup file, a shell
    /// profile or a directory on `PATH`, or hands an unvetted command `$HOME` or a personal
    /// folder, which always asks; holds what and where, for the prompt.
    Sensitive(String),
    /// Every effect is ordinary. Holds the script's class, made unvetted when a target cannot
    /// be resolved, or a copy or move carries a symlink or hidden entry no effect names.
    Ordinary(Class),
}

/// Where one user's guarded trees are found, and where relative targets resolve.
#[derive(Debug, Clone, Default)]
pub struct GuardRoots {
    pub home: Option<PathBuf>,
    /// `$XDG_CONFIG_HOME`; `~/.config` is guarded as well.
    pub config_home: Option<PathBuf>,
    /// `$XDG_DATA_HOME`; `~/.local/share` is guarded as well.
    pub data_home: Option<PathBuf>,
    pub cwd: Option<PathBuf>,
    /// Directories shells look programs up in.
    pub command_dirs: Vec<PathBuf>,
    /// Directories systemd loads user units from, as it reports them.
    pub unit_dirs: Vec<PathBuf>,
    /// Directories whose owner-only files are not sensitive, such as the shared scratch and
    /// spill directories.
    pub exempt: Vec<PathBuf>,
}

/// The guarded paths of one `$HOME`, canonical, and the working directory relative targets
/// resolve against.
#[derive(Debug, Clone, Default)]
pub struct PathGuards {
    home: Option<PathBuf>,
    cwd: Option<PathBuf>,
    exempt: Vec<PathBuf>,
    /// Trees whose recursive deletion or mode change is catastrophic, ancestors included.
    irreplaceable: Vec<(PathBuf, Guarded)>,
    sensitive: Vec<(PathBuf, Guarded)>,
}

/// Which guard a path falls under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Guarded {
    Home,
    PersonalFolder,
    WorkTree,
    Credentials,
    /// A file only its owner may read, as programs keep tokens and keys.
    OwnerOnly,
    Startup,
    ShellProfile,
    CommandDir,
}

impl Guarded {
    fn is_catastrophic(self) -> bool {
        matches!(self, Self::Home | Self::PersonalFolder | Self::WorkTree)
    }
}

impl fmt::Display for Guarded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Home => "the home directory",
            Self::PersonalFolder => "a personal folder",
            Self::WorkTree => "the root of a git work tree",
            Self::Credentials => "credentials",
            Self::OwnerOnly => "a file only its owner may read",
            Self::Startup => "a startup file",
            Self::ShellProfile => "a shell profile",
            Self::CommandDir => "a directory on PATH",
        })
    }
}

impl PathGuards {
    /// The guards of `roots`. Reads `user-dirs.dirs` and resolves symlinks, so it blocks on
    /// the filesystem.
    pub fn read(roots: GuardRoots) -> Self {
        let home = roots
            .home
            .filter(|home| home.is_absolute())
            .and_then(|home| std::fs::canonicalize(home).ok());
        let exempt = roots
            .exempt
            .iter()
            .filter_map(|dir| std::fs::canonicalize(dir).ok())
            .collect();
        let mut sensitive = tagged(roots.unit_dirs, Guarded::Startup);
        sensitive.extend(tagged(roots.command_dirs, Guarded::CommandDir));
        let Some(real_home) = home.as_deref() else {
            return Self {
                cwd: roots.cwd,
                exempt,
                sensitive,
                ..Self::default()
            };
        };
        let config = xdg_dirs(roots.config_home, real_home, DEFAULT_CONFIG_HOME);
        let data = xdg_dirs(roots.data_home, real_home, DEFAULT_DATA_HOME);
        let home_dir = [real_home.to_path_buf()];
        let below = |bases: &[PathBuf], names: &[&str]| -> Vec<PathBuf> {
            bases
                .iter()
                .flat_map(|base| names.iter().map(move |name| base.join(name)))
                .collect()
        };
        let mut credentials = below(&home_dir, HOME_CREDENTIALS);
        credentials.extend(below(&data, DATA_CREDENTIALS));
        let mut startup = below(&config, CONFIG_STARTUP);
        startup.extend(below(&data, DATA_STARTUP));
        let mut profiles = below(&home_dir, HOME_SHELL_PROFILES);
        profiles.extend(below(&config, CONFIG_SHELL_PROFILES));
        sensitive.extend(tagged(credentials, Guarded::Credentials));
        sensitive.extend(tagged(startup, Guarded::Startup));
        sensitive.extend(tagged(profiles, Guarded::ShellProfile));
        let mut irreplaceable = vec![(real_home.to_path_buf(), Guarded::Home)];
        irreplaceable.extend(tagged(
            personal_folders(real_home, &config),
            Guarded::PersonalFolder,
        ));
        Self {
            home,
            cwd: roots.cwd,
            exempt,
            irreplaceable,
            sensitive,
        }
    }

    /// Sort `analysis`'s effects: refused when one is catastrophic, else sensitive when one is,
    /// else ordinary. Blocks on the filesystem.
    pub fn verdict(&self, analysis: Analysis) -> PathVerdict {
        let mut unresolved = None;
        let mut sensitive = None;
        for effect in &analysis.effects {
            let Some(path) = self.resolve(effect) else {
                unresolved = unresolved.or(Some(effect));
                continue;
            };
            match self.guarded(effect.action, &path, effect.recursive) {
                Some(guard)
                    if guard.guarded.is_catastrophic() && effect.action != Action::Unknown =>
                {
                    return PathVerdict::Refused(describe(effect, &path, guard));
                }
                Some(guard) => {
                    sensitive = sensitive.or_else(|| Some(describe(effect, &path, guard)));
                }
                None => {}
            }
        }
        if let Some(why) = sensitive {
            return PathVerdict::Sensitive(why);
        }
        if matches!(analysis.class, Class::Unvetted { .. }) {
            return PathVerdict::Ordinary(analysis.class);
        }
        let why = unresolved
            .map(|effect| format!("where `{}` lies cannot be resolved", effect.path))
            .or_else(|| {
                let carried = &analysis.carried;
                carried.iter().find_map(|source| self.unnamed_cargo(source))
            });
        PathVerdict::Ordinary(why.map_or(analysis.class, |why| Class::Unvetted { why }))
    }

    /// Why a read of the absolute `path`, following symlinks, asks: it lands on credentials or
    /// an owner-only file in `$HOME`, or nowhere that can be resolved. `None` when it is
    /// ordinary. Blocks on the filesystem.
    pub(crate) fn sensitive_read(&self, path: &Path) -> Option<String> {
        let Some(real) = canonical_existing(path) else {
            return Some(format!(
                "reads `{}`, and where that lies cannot be resolved",
                path.display()
            ));
        };
        let guard = self.guarded(Action::Read, &real, false)?;
        let effect = Effect {
            action: Action::Read,
            path: path.display().to_string(),
            recursive: false,
        };
        Some(describe(&effect, &real, guard))
    }

    /// The directory relative targets resolve against.
    pub(crate) fn cwd(&self) -> Option<&Path> {
        self.cwd.as_deref()
    }

    /// The first path, as written, that `effects` change outside `scope`, inside `excluded`,
    /// on a scope directory itself, behind a hidden entry, or where it cannot be resolved;
    /// `None` when every change lands where [`admits_edit`] allows. Blocks on the filesystem.
    pub(crate) fn first_out_of_scope<'e>(
        &self,
        effects: &'e [Effect],
        scope: &[PathBuf],
        excluded: &[PathBuf],
    ) -> Option<&'e str> {
        effects
            .iter()
            .filter(|effect| effect.action != Action::Read)
            .find(|effect| {
                self.resolve(effect).is_none_or(|path| {
                    excluded.iter().any(|dir| path.starts_with(dir))
                        || !scope.iter().any(|dir| admits_edit(dir, &path))
                })
            })
            .map(|effect| effect.path.as_str())
    }

    /// The canonical path `effect` lands on, following a final symlink unless it deletes the
    /// link itself; `None` when `~` or the working directory is unknown.
    fn resolve(&self, effect: &Effect) -> Option<PathBuf> {
        let raw = effect.path.as_str();
        self.located(
            raw,
            effect.action != Action::Delete || names_directory_itself(raw),
        )
    }

    /// The canonical path `raw` names, a final symlink followed only when `follows_link`;
    /// `None` when `~` or the working directory is unknown.
    fn located(&self, raw: &str, follows_link: bool) -> Option<PathBuf> {
        let absolute = match tilde_remainder(raw) {
            Some(_) => expand_tilde(raw, self.home.as_deref()?),
            None if raw.starts_with('~') => return None,
            None if Path::new(raw).is_absolute() => PathBuf::from(raw),
            None => self.cwd.as_deref()?.join(raw),
        };
        if follows_link {
            return canonical_existing(&absolute);
        }
        match (absolute.parent(), absolute.file_name()) {
            (Some(parent), Some(name)) => Some(canonical_existing(parent)?.join(name)),
            _ => canonical_existing(&absolute),
        }
    }

    /// Why a copy or move of `source`, as written, cannot be vetted: it is a symlink, or a tree
    /// holding a symlink or hidden entry, which lands where no effect names it. A source the
    /// script has yet to make is neither.
    fn unnamed_cargo(&self, source: &str) -> Option<String> {
        let path = self.located(source, names_directory_itself(source))?;
        let meta = path.symlink_metadata().ok()?;
        if meta.is_symlink() {
            return Some(format!(
                "`{source}` is a symlink, so a write to where it is copied or moved may land \
                 anywhere"
            ));
        }
        let cargo = meta.is_dir().then(|| unnamed_entry(&path)).flatten()?;
        Some(format!(
            "`{source}` holds {cargo}, which a copy or move carries along unnamed"
        ))
    }

    /// The guard `action` on the canonical `path` falls under, if any.
    fn guarded(&self, action: Action, path: &Path, recursive: bool) -> Option<Guard> {
        let destroys = recursive && matches!(action, Action::Delete | Action::ChangeAttributes);
        let irreplaceable = self
            .irreplaceable
            .iter()
            .find(|(tree, _)| tree.starts_with(path))
            .map(|(tree, guarded)| Guard::of(tree, *guarded, path));
        if action == Action::Unknown {
            return irreplaceable;
        }
        if destroys {
            if irreplaceable.is_some() {
                return irreplaceable;
            }
            if path.join(".git").symlink_metadata().is_ok() {
                return Some(Guard::of(path, Guarded::WorkTree, path));
            }
        }
        self.sensitive
            .iter()
            .filter(|(_, guarded)| action != Action::Read || *guarded == Guarded::Credentials)
            .find(|(tree, _)| path.starts_with(tree) || (recursive && holds(path, tree)))
            .map(|(tree, guarded)| Guard::of(tree, *guarded, path))
            .or_else(|| self.owner_only(path, recursive && action == Action::Read))
    }

    /// The guard for an owner-only file in `$HOME` at `path`, or, when `scan_entries`, directly
    /// inside it; files under an exempt directory do not count.
    fn owner_only(&self, path: &Path, scan_entries: bool) -> Option<Guard> {
        let home = self.home.as_deref()?;
        if !path.starts_with(home) || self.exempt.iter().any(|dir| path.starts_with(dir)) {
            return None;
        }
        let guard = |holds| Guard {
            guarded: Guarded::OwnerOnly,
            holds,
        };
        if is_owner_only_file(path) {
            return Some(guard(false));
        }
        let entries = std::fs::read_dir(path).ok().filter(|_| scan_entries)?;
        entries
            .take(MAX_SCANNED_ENTRIES)
            .filter_map(Result::ok)
            .any(|entry| is_owner_only_file(&entry.path()))
            .then(|| guard(true))
    }
}

/// A guard a path falls under.
#[derive(Debug, Clone, Copy)]
struct Guard {
    guarded: Guarded,
    /// The path is an ancestor of the guarded tree rather than in it.
    holds: bool,
}

impl Guard {
    fn of(tree: &Path, guarded: Guarded, path: &Path) -> Self {
        Self {
            guarded,
            holds: tree != path && tree.starts_with(path),
        }
    }
}

/// Whether an edit of `path` stays in the scope directory `dir`: strictly below it, with no
/// hidden (dot) entry between them, where tools keep files other programs run.
pub(crate) fn admits_edit(dir: &Path, path: &Path) -> bool {
    path.strip_prefix(dir)
        .is_ok_and(|below| !below.as_os_str().is_empty() && !has_hidden_component(below))
}

/// Whether any component of `path` is a hidden (dot) entry.
fn has_hidden_component(path: &Path) -> bool {
    path.components()
        .any(|component| is_hidden(component.as_os_str()))
}

fn is_hidden(name: &OsStr) -> bool {
    name.as_encoded_bytes().starts_with(b".")
}

/// The first symlink or hidden entry found below the directory `tree`, or that it holds more
/// entries than are checked or some that cannot be listed; `None` when it holds none of these.
fn unnamed_entry(tree: &Path) -> Option<String> {
    let mut pending = vec![tree.to_path_buf()];
    let mut checked = 0;
    while let Some(dir) = pending.pop() {
        let Ok(entries) = typed_entries(&dir) else {
            return Some(format!(
                "entries of `{}` that cannot be listed",
                dir.display()
            ));
        };
        checked += entries.len();
        if checked > MAX_CARRIED_ENTRIES {
            return Some(format!(
                "more than {MAX_CARRIED_ENTRIES} entries, too many to check for symlinks and \
                 hidden entries"
            ));
        }
        for (path, kind) in entries {
            if kind.is_symlink() {
                return Some(format!("the symlink `{}`", path.display()));
            }
            if path.file_name().is_some_and(is_hidden) {
                return Some(format!("the hidden entry `{}`", path.display()));
            }
            if kind.is_dir() {
                pending.push(path);
            }
        }
    }
    None
}

/// Up to one more than [`MAX_CARRIED_ENTRIES`] entries of `dir`, each with its own type, not
/// that of what a symlink points at.
fn typed_entries(dir: &Path) -> io::Result<Vec<(PathBuf, FileType)>> {
    std::fs::read_dir(dir)?
        .take(MAX_CARRIED_ENTRIES + 1)
        .map(|entry| {
            let entry = entry?;
            Ok((entry.path(), entry.file_type()?))
        })
        .collect()
}

/// Whether a recursive operation on `path` reaches `tree`: it lies below `path` and exists.
fn holds(path: &Path, tree: &Path) -> bool {
    tree.starts_with(path) && tree.symlink_metadata().is_ok()
}

/// The deepest existing ancestor of `path`, or `path` itself, and the rest of `path` below it.
pub(crate) fn split_at_existing(path: &Path) -> (PathBuf, PathBuf) {
    let mut anchor = path.to_path_buf();
    let mut tail = PathBuf::new();
    loop {
        if anchor.symlink_metadata().is_ok() {
            return (anchor, tail);
        }
        let Some(parent) = anchor.parent() else {
            return (path.to_path_buf(), PathBuf::new());
        };
        let Some(file_name) = anchor.file_name() else {
            return (anchor, tail);
        };
        let mut new_tail = PathBuf::from(file_name);
        if !tail.as_os_str().is_empty() {
            new_tail.push(&tail);
        }
        tail = new_tail;
        anchor = parent.to_path_buf();
    }
}

/// `path` with its deepest existing ancestor canonicalized and the rest cleaned lexically.
pub(crate) fn canonical_existing(path: &Path) -> Option<PathBuf> {
    let (anchor, tail) = split_at_existing(path);
    let anchor = std::fs::canonicalize(anchor).ok()?;
    Some(lexical_clean(&anchor.join(tail)))
}

/// Whether `raw` names a directory through its contents (`dir/`, `dir/.`), so a symlink in
/// its last component is followed even by an operation on links.
fn names_directory_itself(raw: &str) -> bool {
    raw.ends_with('/') || matches!(raw, "." | "..") || raw.ends_with("/.") || raw.ends_with("/..")
}

/// Each of `paths` under `guarded`, also by its canonical spelling when a symlink leads there.
fn tagged(paths: Vec<PathBuf>, guarded: Guarded) -> Vec<(PathBuf, Guarded)> {
    paths
        .into_iter()
        .flat_map(|path| {
            let canonical = std::fs::canonicalize(&path)
                .ok()
                .filter(|real| *real != path);
            std::iter::once(path).chain(canonical)
        })
        .map(|path| (path, guarded))
        .collect()
}

/// `xdg_home` when it is absolute, and `home`'s `default` below it, deduplicated.
fn xdg_dirs(xdg_home: Option<PathBuf>, home: &Path, default: &str) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = xdg_home
        .into_iter()
        .filter(|dir| dir.is_absolute())
        .collect();
    let fallback = home.join(default);
    if !dirs.contains(&fallback) {
        dirs.push(fallback);
    }
    dirs
}

/// Whether `path` is a regular file no one but its owner may read or write.
fn is_owner_only_file(path: &Path) -> bool {
    std::fs::metadata(path)
        .is_ok_and(|meta| meta.is_file() && meta.permissions().mode() & 0o077 == 0)
}

/// The personal folders of `home`: those the first `user-dirs.dirs` among `config` names,
/// else the usual names.
fn personal_folders(home: &Path, config: &[PathBuf]) -> Vec<PathBuf> {
    let named = config
        .iter()
        .find_map(|dir| std::fs::read_to_string(dir.join(USER_DIRS_FILE)).ok())
        .unwrap_or_default();
    PERSONAL_FOLDERS
        .iter()
        .map(|&(key, default)| {
            named
                .lines()
                .find_map(|line| user_dir(line, key, home))
                .unwrap_or_else(|| home.join(default))
        })
        .filter(|folder| folder != home)
        .collect()
}

/// The folder a `user-dirs.dirs` line `KEY="$HOME/Folder"` names for `key`.
fn user_dir(line: &str, key: &str, home: &Path) -> Option<PathBuf> {
    let value = line
        .trim()
        .strip_prefix(key)?
        .strip_prefix('=')?
        .trim_matches('"');
    match value.strip_prefix("$HOME") {
        Some(rest) => Some(lexical_clean(&home.join(rest.trim_start_matches('/')))),
        None => Path::new(value).is_absolute().then(|| PathBuf::from(value)),
    }
}

/// What `effect` does to the guarded `path`, for an error line or prompt.
fn describe(effect: &Effect, path: &Path, guard: Guard) -> String {
    let verb = match effect.action {
        Action::Read => "reads",
        Action::Write => "writes",
        Action::Delete => "deletes",
        Action::ChangeAttributes => "changes the mode of",
        Action::Unknown => "hands a command that could not be vetted",
    };
    let holding = if guard.holds { "a tree holding " } else { "" };
    let class = if guard.guarded.is_catastrophic() {
        "catastrophic"
    } else {
        "sensitive"
    };
    format!(
        "{verb} {holding}{} (`{}`, {}), a {class} path",
        guard.guarded,
        effect.path,
        path.display()
    )
}

#[cfg(test)]
mod tests;
