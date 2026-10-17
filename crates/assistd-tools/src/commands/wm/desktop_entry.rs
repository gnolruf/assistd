//! Installed XDG desktop entries, and the command lines their `Exec` keys
//! describe.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::policy::is_within_visible_home_entry;

/// `$XDG_DATA_DIRS` when unset or empty, per the XDG base directory spec.
const DEFAULT_DATA_DIRS: &str = "/usr/local/share:/usr/share";

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
    icon: Option<String>,
    /// `Exec` split into arguments, quoting removed and field codes kept.
    exec: Vec<String>,
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
        let exec = key("Exec")
            .and_then(split_exec)
            .filter(|exec| !exec.is_empty());
        let (None, Some(exec)) = (why, exec) else {
            let why = why.unwrap_or("its Exec line is missing or malformed");
            return Err(EntryError::NotLaunchable { path, why });
        };
        Ok(Self {
            name: key("Name").unwrap_or(stem).to_string(),
            terminal: key("Terminal") == Some("true"),
            icon: key("Icon").map(str::to_string),
            exec,
            path,
        })
    }

    /// The most demanding field code in `Exec`: URLs over files, many over
    /// one.
    pub(super) fn targets(&self) -> Targets {
        let has = |code| {
            self.exec
                .iter()
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
        self.exec
            .iter()
            .flat_map(|arg| self.expand(arg, targets))
            .collect()
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

/// `$XDG_DATA_HOME`, then each of `$XDG_DATA_DIRS`, joined with
/// `applications`; relative entries are ignored.
fn applications_dirs() -> Vec<PathBuf> {
    let env_or = |name: &str, default: Option<String>| {
        std::env::var(name)
            .ok()
            .filter(|value| !value.is_empty())
            .or(default)
    };
    let data_home = env_or(
        "XDG_DATA_HOME",
        std::env::var("HOME")
            .ok()
            .map(|home| format!("{home}/.local/share")),
    );
    let data_dirs = env_or("XDG_DATA_DIRS", Some(DEFAULT_DATA_DIRS.to_string()));
    data_home
        .iter()
        .chain(&data_dirs)
        .flat_map(std::env::split_paths)
        .filter(|dir| dir.is_absolute())
        .map(|dir| dir.join("applications"))
        .collect()
}

/// Whether `real` lies under a visible entry of `$HOME`, where sandboxed
/// commands can write.
fn is_writable_by_sandbox(real: &Path) -> bool {
    std::env::var_os("HOME")
        .and_then(|home| std::fs::canonicalize(home).ok())
        .is_some_and(|home| is_within_visible_home_entry(real, &home))
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
