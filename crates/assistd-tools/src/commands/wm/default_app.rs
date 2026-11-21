//! The user's default application for a file: its type from the shared MIME
//! database, then the desktop entry associated with that type.

use std::cmp::Reverse;
use std::path::{Path, PathBuf};

use super::desktop_entry::{DesktopEntry, Targets, config_dirs, data_dirs};

const TEXT_PLAIN: &str = "text/plain";
const DEFAULTS_GROUP: &str = "[Default Applications]";
const CACHE_GROUP: &str = "[MIME Cache]";

/// Why a file has no application to open it.
#[derive(Debug, thiserror::Error)]
pub(super) enum DefaultAppError {
    #[error("{} is not an existing file", .0.display())]
    NotAFile(PathBuf),
    #[error("the type of {} is not known from its name", .0.display())]
    UnknownType(PathBuf),
    #[error("no installed application is set to open {0} files")]
    NoApplication(String),
}

/// A glob of the MIME database that matched a file name.
#[derive(Debug, PartialEq, Eq)]
struct GlobMatch {
    weight: u32,
    glob_len: usize,
    mime: String,
}

/// The entry that opens `path` by default: the first launchable one that takes
/// files, for the file's type or else a type it is a kind of. Reads the
/// filesystem, so it blocks.
pub(super) fn default_app(path: &Path) -> Result<DesktopEntry, DefaultAppError> {
    if !path.is_file() {
        return Err(DefaultAppError::NotAFile(path.to_path_buf()));
    }
    let data_dirs = data_dirs();
    let config_dirs = config_dirs();
    let mime = mime_type(path, &data_dirs)
        .ok_or_else(|| DefaultAppError::UnknownType(path.to_path_buf()))?;
    let subclasses: Vec<String> = data_dirs
        .iter()
        .filter_map(|dir| std::fs::read_to_string(dir.join("mime/subclasses")).ok())
        .collect();
    with_ancestors(&mime, &subclasses)
        .iter()
        .flat_map(|kind| associated_ids(kind, &config_dirs, &data_dirs))
        .filter_map(|id| DesktopEntry::find(&id).ok())
        .find(|entry| entry.targets() != Targets::None)
        .ok_or(DefaultAppError::NoApplication(mime))
}

/// The type whose glob best matches `path`'s name in any of `data_dirs`.
fn mime_type(path: &Path, data_dirs: &[PathBuf]) -> Option<String> {
    let name = path.file_name()?.to_string_lossy();
    data_dirs
        .iter()
        .filter_map(|dir| std::fs::read_to_string(dir.join("mime/globs2")).ok())
        .filter_map(|globs| best_glob(&globs, &name))
        .min_by_key(|found| Reverse((found.weight, found.glob_len)))
        .map(|found| found.mime)
}

/// The heaviest, then longest, `weight:type:glob[:flags]` line of a `globs2`
/// file that matches `file_name`; the first of equals.
fn best_glob(globs: &str, file_name: &str) -> Option<GlobMatch> {
    globs
        .lines()
        .filter_map(|line| {
            let mut fields = line.splitn(4, ':');
            let weight = fields.next()?.parse().ok()?;
            let mime = fields.next()?;
            let glob = fields.next()?;
            let case_sensitive = fields
                .next()
                .is_some_and(|flags| flags.split(',').any(|flag| flag == "cs"));
            glob_matches(glob, file_name, case_sensitive).then(|| GlobMatch {
                weight,
                glob_len: glob.len(),
                mime: mime.to_string(),
            })
        })
        .min_by_key(|found| Reverse((found.weight, found.glob_len)))
}

/// Whether `name` is the literal `glob` or ends with the suffix of a `*suffix`
/// one; a glob with any other wildcard matches nothing.
fn glob_matches(glob: &str, name: &str, case_sensitive: bool) -> bool {
    let (suffix_only, literal) = match glob.strip_prefix('*') {
        Some(suffix) => (true, suffix),
        None => (false, glob),
    };
    if literal.is_empty() || literal.contains(['*', '?', '[']) {
        return false;
    }
    let (literal, name) = if case_sensitive {
        (literal.to_string(), name.to_string())
    } else {
        (literal.to_lowercase(), name.to_lowercase())
    };
    if suffix_only {
        name.len() > literal.len() && name.ends_with(&literal)
    } else {
        name == literal
    }
}

/// `mime`, then `text/plain` when it is a text type, then every type the
/// `subclasses` files make it a kind of, nearest first.
fn with_ancestors(mime: &str, subclasses: &[String]) -> Vec<String> {
    let mut kinds = vec![mime.to_string()];
    if mime.starts_with("text/") && mime != TEXT_PLAIN {
        kinds.push(TEXT_PLAIN.to_string());
    }
    let mut visited = 0;
    while let Some(kind) = kinds.get(visited).cloned() {
        visited += 1;
        let parents = subclasses
            .iter()
            .flat_map(|text| text.lines())
            .filter_map(|line| line.split_once(' '))
            .filter(|(child, _)| *child == kind)
            .map(|(_, parent)| parent.trim());
        for parent in parents {
            if !kinds.iter().any(|known| known == parent) {
                kinds.push(parent.to_string());
            }
        }
    }
    kinds
}

/// Desktop ids associated with `mime`: the defaults of each `mimeapps.list`,
/// then every application whose entry lists the type.
fn associated_ids(mime: &str, config_dirs: &[PathBuf], data_dirs: &[PathBuf]) -> Vec<String> {
    let applications = || data_dirs.iter().map(|dir| dir.join("applications"));
    let defaults = config_dirs
        .iter()
        .cloned()
        .chain(applications())
        .map(|dir| (dir.join("mimeapps.list"), DEFAULTS_GROUP));
    let cached = applications().map(|dir| (dir.join("mimeinfo.cache"), CACHE_GROUP));
    defaults
        .chain(cached)
        .filter_map(|(file, group)| Some((std::fs::read_to_string(file).ok()?, group)))
        .flat_map(|(text, group)| group_ids(&text, group, mime))
        .collect()
}

/// The `;`-separated desktop ids of the `mime=` key under the `group` header.
fn group_ids(text: &str, group: &str, mime: &str) -> Vec<String> {
    let mut in_group = false;
    let mut ids = Vec::new();
    for line in text.lines().map(str::trim) {
        if line.starts_with('[') {
            in_group = line == group;
        } else if in_group
            && let Some((key, value)) = line.split_once('=')
            && key.trim() == mime
        {
            ids.extend(
                value
                    .split(';')
                    .map(str::trim)
                    .filter(|id| !id.is_empty())
                    .map(str::to_string),
            );
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_name_resolves_to_its_type_then_the_types_it_is_a_kind_of() {
        let globs = "# generated\n\
                     80:text/html:*.html\n\
                     50:application/xhtml+xml:*.html\n\
                     50:text/markdown:*.md\n\
                     50:application/gzip:*.gz\n\
                     50:application/x-compressed-tar:*.tar.gz\n\
                     50:text/x-c++src:*.C:cs\n\
                     50:text/x-makefile:makefile\n\
                     50:text/x-readme:readme*\n";
        let mime = |name: &str| best_glob(globs, name).map(|found| found.mime);
        assert_eq!(mime("page.HTML").as_deref(), Some("text/html"));
        assert_eq!(
            mime("src.tar.gz").as_deref(),
            Some("application/x-compressed-tar")
        );
        assert_eq!(mime("Makefile").as_deref(), Some("text/x-makefile"));
        assert_eq!(mime("main.C").as_deref(), Some("text/x-c++src"));
        assert_eq!(mime("main.c"), None);
        assert_eq!(mime(".md"), None);
        assert_eq!(mime("readme.first"), None);

        let subclasses = ["text/x-markdown text/markdown\n\
                           image/svg+xml application/xml\n\
                           application/xml text/plain\n\
                           text/plain application/xml\n"
            .to_string()];
        assert_eq!(
            with_ancestors("text/markdown", &subclasses),
            ["text/markdown", "text/plain", "application/xml"]
        );
        assert_eq!(
            with_ancestors("image/svg+xml", &subclasses),
            ["image/svg+xml", "application/xml", "text/plain"]
        );
    }

    #[test]
    fn association_files_list_ids_only_under_their_own_group() {
        let list = "[Added Associations]\n\
                    text/plain=added.desktop;\n\
                    [Default Applications]\n\
                    text/plain = nvim.desktop; org.gnome.TextEditor.desktop;\n\
                    text/html=firefox.desktop\n";
        assert_eq!(
            group_ids(list, DEFAULTS_GROUP, "text/plain"),
            ["nvim.desktop", "org.gnome.TextEditor.desktop"]
        );
        assert!(group_ids(list, CACHE_GROUP, "text/plain").is_empty());
    }
}
