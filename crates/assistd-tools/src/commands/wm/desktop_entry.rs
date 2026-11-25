//! Installed XDG desktop entries, and the command lines their `Exec` keys
//! describe.

use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use assistd_utils::program::find_program;

use crate::policy::is_within_visible_home_entry;

/// `$XDG_DATA_DIRS` when unset or empty, per the XDG base directory spec.
const DEFAULT_DATA_DIRS: &str = "/usr/local/share:/usr/share";

/// `$XDG_CONFIG_DIRS` when unset or empty, per the XDG base directory spec.
const DEFAULT_CONFIG_DIRS: &str = "/etc/xdg";

/// What an entry's `Exec` line accepts in place of its field codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Targets {
    None,
    /// Local paths: `%f`, or `%F` when `many`.
    Files {
        many: bool,
    },
    /// URLs or local paths: `%u`, or `%U` when `many`.
    Urls {
        many: bool,
    },
}

/// An installed application entry that sandboxed commands cannot write.
#[derive(Debug)]
pub(super) struct DesktopEntry {
    pub(super) path: PathBuf,
    pub(super) name: String,
    pub(super) terminal: bool,
    /// Listed in the `TerminalEmulator` category.
    pub(super) is_terminal_emulator: bool,
    icon: Option<String>,
    /// The first word of `Exec`, quoting removed.
    program: String,
    /// The other words of `Exec`, quoting removed and field codes kept.
    arguments: Vec<String>,
}

/// What a launch runs, as far as its command line says, and whether an approval can be
/// pinned to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Executable {
    /// The canonical path of the program, which sandboxed commands cannot write, like
    /// every other path the command line names.
    Pinned(PathBuf),
    /// A path the command line names that sandboxed commands can write.
    Writable(PathBuf),
    /// Why the program could not be found from here.
    Unresolved(String),
}

impl Executable {
    /// The same verdict for a launch that also runs `program`.
    pub(super) fn running_also(self, program: &Path) -> Self {
        match (self, writable_by_sandbox(program)) {
            (Self::Pinned(_), Some(writable)) => Self::Writable(writable),
            (verdict, _) => verdict,
        }
    }

    /// The program an approval for good is pinned to, when one can be.
    pub(super) fn pinned(&self) -> Option<&Path> {
        match self {
            Self::Pinned(program) => Some(program),
            Self::Writable(_) | Self::Unresolved(_) => None,
        }
    }

    /// Why no approval can be pinned, as a sentence for the prompt.
    pub(super) fn caution(&self) -> Option<String> {
        match self {
            Self::Pinned(_) => None,
            Self::Writable(path) => Some(format!(
                "It runs {}, which can be changed from inside the sandbox.",
                path.display()
            )),
            Self::Unresolved(why) => Some(format!(
                "The program it runs could not be found, so it cannot be approved for good: {why}."
            )),
        }
    }
}

/// Why no launchable entry was found.
#[derive(Debug, thiserror::Error)]
pub(super) enum EntryError {
    #[error("no installed application has the desktop id '{0}'")]
    NotFound(String),
    #[error("{} cannot be launched: {why}", .path.display())]
    NotLaunchable { path: PathBuf, why: &'static str },
    #[error("{} is writable by sandboxed commands", .0.display())]
    Untrusted(PathBuf),
}

impl DesktopEntry {
    /// The entry `id` names, with or without `.desktop`, from the first XDG
    /// applications directory holding it.
    pub(super) fn find(id: &str) -> Result<Self, EntryError> {
        let not_found = || EntryError::NotFound(id.to_string());
        let stem = id.strip_suffix(".desktop").unwrap_or(id);
        if stem.is_empty() || stem.starts_with('.') || stem.contains('/') {
            return Err(not_found());
        }
        let file_name = format!("{stem}.desktop");
        let path = applications_dirs()
            .into_iter()
            .map(|dir| dir.join(&file_name))
            .find(|path| path.is_file())
            .ok_or_else(not_found)?;
        let real = std::fs::canonicalize(&path).map_err(|_| not_found())?;
        if is_writable_by_sandbox(&real) {
            return Err(EntryError::Untrusted(real));
        }
        match std::fs::read_to_string(&real) {
            Ok(text) => Self::parse(real, stem, &text),
            Err(_) => Err(EntryError::NotLaunchable {
                path: real,
                why: "it cannot be read",
            }),
        }
    }

    /// The entry `text`, read from `path`, describes; `stem` names it when
    /// it has no `Name`.
    pub(super) fn parse(path: PathBuf, stem: &str, text: &str) -> Result<Self, EntryError> {
        let keys = main_group(text);
        let key = |name: &str| keys.get(name).map(String::as_str);
        let why = if key("Type") != Some("Application") {
            Some("it is not an application")
        } else if key("Hidden") == Some("true") {
            Some("it is hidden")
        } else {
            None
        };
        let exec = key("Exec").and_then(split_exec).and_then(|words| {
            let mut words = words.into_iter();
            Some((words.next()?, words.collect()))
        });
        let (None, Some((program, arguments))) = (why, exec) else {
            let why = why.unwrap_or("its Exec line is missing or malformed");
            return Err(EntryError::NotLaunchable { path, why });
        };
        Ok(Self {
            name: key("Name").unwrap_or(stem).to_string(),
            terminal: key("Terminal") == Some("true"),
            is_terminal_emulator: key("Categories")
                .is_some_and(|categories| categories.split(';').any(|c| c == "TerminalEmulator")),
            icon: key("Icon").map(str::to_string),
            program,
            arguments,
            path,
        })
    }

    /// The most demanding field code in `Exec`: URLs over files, many over
    /// one.
    pub(super) fn targets(&self) -> Targets {
        let has = |code| {
            self.exec()
                .any(|arg| field_codes(arg).any(|found| found == code))
        };
        match ['U', 'u', 'F', 'f'].into_iter().find(|code| has(*code)) {
            Some('U') => Targets::Urls { many: true },
            Some('u') => Targets::Urls { many: false },
            Some('F') => Targets::Files { many: true },
            Some(_) => Targets::Files { many: false },
            None => Targets::None,
        }
    }

    /// `Exec` with its field codes expanded for `targets`.
    pub(super) fn command_line(&self, targets: &[String]) -> Vec<String> {
        self.exec()
            .flat_map(|arg| self.expand(arg, targets))
            .collect()
    }

    /// What `Exec` runs: its first word, found through `path_env` as a shell would unless
    /// it is absolute, and any other word that is an absolute path. Blocks on the
    /// filesystem.
    pub(super) fn executable(&self, path_env: &OsStr) -> Executable {
        let program = Path::new(&self.program);
        if program.is_relative() && self.program.contains('/') {
            return Executable::Unresolved(format!("`{}` is a relative path", self.program));
        }
        let found = find_program(program, path_env)
            .map_err(|e| e.to_string())
            .and_then(|found| std::fs::canonicalize(found).map_err(|e| e.to_string()));
        match found {
            Ok(real) if is_writable_by_sandbox(&real) => Executable::Writable(real),
            Ok(real) => self
                .arguments
                .iter()
                .map(Path::new)
                .filter(|word| word.is_absolute())
                .fold(Executable::Pinned(real), Executable::running_also),
            Err(why) => Executable::Unresolved(format!("`{}`: {why}", self.program)),
        }
    }

    fn exec(&self) -> impl Iterator<Item = &String> {
        std::iter::once(&self.program).chain(&self.arguments)
    }

    /// `%F` and `%U` become every target and `%i` the icon option; an
    /// argument other field codes leave empty is dropped.
    fn expand(&self, arg: &str, targets: &[String]) -> Vec<String> {
        match arg {
            "%F" | "%U" => targets.to_vec(),
            "%i" => self
                .icon
                .iter()
                .flat_map(|icon| ["--icon".to_string(), icon.clone()])
                .collect(),
            _ => {
                let expanded = self.expand_inline(arg, targets.first().map(String::as_str));
                if expanded.is_empty() && !arg.is_empty() {
                    Vec::new()
                } else {
                    vec![expanded]
                }
            }
        }
    }

    fn expand_inline(&self, arg: &str, target: Option<&str>) -> String {
        let mut expanded = String::with_capacity(arg.len());
        let mut chars = arg.chars();
        while let Some(c) = chars.next() {
            if c != '%' {
                expanded.push(c);
                continue;
            }
            match chars.next() {
                Some('%') => expanded.push('%'),
                Some('f' | 'u') => expanded.push_str(target.unwrap_or_default()),
                Some('c') => expanded.push_str(&self.name),
                Some('k') => expanded.push_str(&self.path.to_string_lossy()),
                _ => {}
            }
        }
        expanded
    }
}

fn applications_dirs() -> Vec<PathBuf> {
    data_dirs()
        .into_iter()
        .map(|dir| dir.join("applications"))
        .collect()
}

/// `$XDG_DATA_HOME`, then each of `$XDG_DATA_DIRS`.
pub(super) fn data_dirs() -> Vec<PathBuf> {
    base_dirs(
        "XDG_DATA_HOME",
        ".local/share",
        "XDG_DATA_DIRS",
        DEFAULT_DATA_DIRS,
    )
}

/// `$XDG_CONFIG_HOME`, then each of `$XDG_CONFIG_DIRS`.
pub(super) fn config_dirs() -> Vec<PathBuf> {
    base_dirs(
        "XDG_CONFIG_HOME",
        ".config",
        "XDG_CONFIG_DIRS",
        DEFAULT_CONFIG_DIRS,
    )
}

/// The directory `home_var` names, or `home_default` under `$HOME`, then each
/// of `dirs_var`, or `dirs_default`; relative entries are ignored.
fn base_dirs(
    home_var: &str,
    home_default: &str,
    dirs_var: &str,
    dirs_default: &str,
) -> Vec<PathBuf> {
    let env_or = |name: &str, default: Option<String>| {
        std::env::var(name)
            .ok()
            .filter(|value| !value.is_empty())
            .or(default)
    };
    let home = env_or(
        home_var,
        std::env::var("HOME")
            .ok()
            .map(|home| format!("{home}/{home_default}")),
    );
    let dirs = env_or(dirs_var, Some(dirs_default.to_string()));
    home.iter()
        .chain(&dirs)
        .flat_map(std::env::split_paths)
        .filter(|dir| dir.is_absolute())
        .collect()
}

/// Whether `real` lies under a visible entry of `$HOME`, where sandboxed
/// commands can write.
fn is_writable_by_sandbox(real: &Path) -> bool {
    std::env::var_os("HOME")
        .and_then(|home| std::fs::canonicalize(home).ok())
        .is_some_and(|home| is_within_visible_home_entry(real, &home))
}

/// `path`, resolved when it exists, if sandboxed commands can write it.
fn writable_by_sandbox(path: &Path) -> Option<PathBuf> {
    let real = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    is_writable_by_sandbox(&real).then_some(real)
}

/// The keys of the `[Desktop Entry]` group, values unescaped; a repeated
/// key keeps its first value.
fn main_group(text: &str) -> HashMap<&str, String> {
    let mut in_main = false;
    let mut keys = HashMap::new();
    for line in text.lines().map(str::trim) {
        if line.starts_with('[') {
            in_main = line == "[Desktop Entry]";
        } else if in_main
            && !line.starts_with('#')
            && let Some((key, value)) = line.split_once('=')
        {
            keys.entry(key.trim())
                .or_insert_with(|| unescape_value(value.trim()));
        }
    }
    keys
}

/// The spec's string escapes `\s`, `\n`, `\t`, `\r` and `\\`; any other
/// backslash is kept for the `Exec` quoting rules.
fn unescape_value(value: &str) -> String {
    let mut unescaped = String::with_capacity(value.len());
    let mut chars = value.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            unescaped.push(c);
            continue;
        }
        match chars.next() {
            Some('s') => unescaped.push(' '),
            Some('n') => unescaped.push('\n'),
            Some('t') => unescaped.push('\t'),
            Some('r') => unescaped.push('\r'),
            Some('\\') => unescaped.push('\\'),
            Some(other) => unescaped.extend(['\\', other]),
            None => unescaped.push('\\'),
        }
    }
    unescaped
}

/// `exec` split at unquoted spaces, with double quotes removed and the
/// backslash escapes inside them resolved; `None` for an unclosed quote.
fn split_exec(exec: &str) -> Option<Vec<String>> {
    let mut args = Vec::new();
    let mut current: Option<String> = None;
    let mut quoted = false;
    let mut chars = exec.chars();
    while let Some(c) = chars.next() {
        match (quoted, c) {
            (false, ' ' | '\t') => args.extend(current.take()),
            (_, '"') => {
                quoted = !quoted;
                current.get_or_insert_default();
            }
            (true, '\\') => current.get_or_insert_default().push(chars.next()?),
            (_, c) => current.get_or_insert_default().push(c),
        }
    }
    if quoted {
        return None;
    }
    args.extend(current);
    Some(args)
}

/// The field-code letters in `arg`, skipping the `%%` escape.
fn field_codes(arg: &str) -> impl Iterator<Item = char> + '_ {
    let mut chars = arg.chars();
    std::iter::from_fn(move || {
        while let Some(c) = chars.next() {
            if c == '%' {
                match chars.next() {
                    Some('%') | None => {}
                    Some(code) => return Some(code),
                }
            }
        }
        None
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(exec: &str) -> DesktopEntry {
        let text = format!(
            "[Desktop Entry]\nType=Application\nName=Viewer\nIcon=viewer\nExec={exec}\n\
             [Desktop Action new]\nExec=ignored\n"
        );
        DesktopEntry::parse(
            PathBuf::from("/usr/share/applications/viewer.desktop"),
            "viewer",
            &text,
        )
        .expect("launchable entry")
    }

    #[test]
    fn exec_lines_unquote_and_expand_field_codes_per_the_spec() {
        let files = ["/a b".to_string(), "/c".to_string()];
        for (exec, targets, expected, accepts) in [
            (
                "viewer %F",
                &files[..],
                vec!["viewer", "/a b", "/c"],
                Targets::Files { many: true },
            ),
            (
                "viewer --url -- %u",
                &[],
                vec!["viewer", "--url", "--"],
                Targets::Urls { many: false },
            ),
            (
                "viewer --file=%f",
                &files[..1],
                vec!["viewer", "--file=/a b"],
                Targets::Files { many: false },
            ),
            (
                "viewer %i --name %c %k",
                &[],
                vec![
                    "viewer",
                    "--icon",
                    "viewer",
                    "--name",
                    "Viewer",
                    "/usr/share/applications/viewer.desktop",
                ],
                Targets::None,
            ),
            (
                r#""/opt/my app/run" "say \"hi\"" 100%% %U"#,
                &files[1..],
                vec!["/opt/my app/run", r#"say "hi""#, "100%", "/c"],
                Targets::Urls { many: true },
            ),
            (
                r#"sh -c "echo \\\\$HOME" %d"#,
                &[],
                vec!["sh", "-c", r"echo \$HOME"],
                Targets::None,
            ),
            (
                "viewer\\sarg \"a\\sb\"",
                &[],
                vec!["viewer", "arg", "a b"],
                Targets::None,
            ),
        ] {
            let entry = entry(exec);
            assert_eq!(entry.command_line(targets), expected, "{exec}");
            assert_eq!(entry.targets(), accepts, "{exec}");
        }
    }

    #[test]
    fn an_exec_line_is_pinned_only_when_nothing_it_names_is_writable_by_the_sandbox() {
        let Some(home) = std::env::var_os("HOME").and_then(|home| std::fs::canonicalize(home).ok())
        else {
            return;
        };
        let planted = home.join("assistd-absent-dir/start.sh");
        let path_env = OsStr::new("/usr/bin:/bin");
        let executable = |exec: &str| entry(exec).executable(path_env);

        assert!(executable("sh -c true /etc/profile").pinned().is_some());
        for exec in [
            format!("sh {}", planted.display()),
            format!("env X=1 {} %U", planted.display()),
        ] {
            assert_eq!(
                executable(&exec),
                Executable::Writable(planted.clone()),
                "{exec}"
            );
        }
        for exec in ["assistd-absent-program", "bin/relative", "/assistd-absent"] {
            assert!(
                matches!(executable(exec), Executable::Unresolved(_)),
                "{exec}"
            );
        }
    }

    #[test]
    fn entries_that_are_not_applications_or_have_no_command_are_refused() {
        let path = PathBuf::from("/usr/share/applications/x.desktop");
        for text in [
            "[Desktop Entry]\nType=Link\nExec=x\n",
            "[Desktop Entry]\nType=Application\nHidden=true\nExec=x\n",
            "[Desktop Entry]\nType=Application\nExec=\"x\n",
            "[Desktop Entry]\nType=Application\n[Other]\nExec=x\n",
        ] {
            assert!(
                matches!(
                    DesktopEntry::parse(path.clone(), "x", text),
                    Err(EntryError::NotLaunchable { .. })
                ),
                "{text:?}"
            );
        }
    }
}
