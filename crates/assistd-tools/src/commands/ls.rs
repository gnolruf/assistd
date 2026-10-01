use std::fs::Metadata;
use std::io::{self, ErrorKind};

use async_trait::async_trait;
use tokio::fs::ReadDir;

use crate::command::{Command, CommandInput, CommandOutput, io_error_nav};

/// `ls [-al] [PATH]...`: list each PATH (default CWD) as sorted
/// `<type>\t<size>\t<name>` rows, without following symlinks.
#[derive(Debug)]
pub struct LsCommand;

#[async_trait]
impl Command for LsCommand {
    fn name(&self) -> &'static str {
        "ls"
    }

    fn summary(&self) -> &'static str {
        "list directories or files (type, size, name); -a for dot-entries"
    }

    fn help(&self) -> String {
        "usage: ls [-al] [PATH]...\n\
         \n\
         List directory entries alphabetically, one per line, formatted as \
         `<type>\\t<size>\\t<name>`. `<type>` is `dir`, `file`, or `symlink`; \
         `<size>` is bytes from the entry's (symlink-preserving) metadata. \
         A PATH that is not a directory prints one row for that path. \
         With several PATHs, non-directory rows come first, then each \
         directory under a `PATH:` header, each group sorted by path. \
         Defaults to the daemon's CWD if PATH is omitted.\n\
         \n\
         Flags:\n  \
           -a  include dot-entries (hidden by default)\n  \
           -l  long format (already the only format; accepted for familiarity)\n"
            .to_string()
    }

    async fn run(&self, input: CommandInput) -> CommandOutput {
        let (show_hidden, paths) = match parse_flags(&input.args) {
            Ok(v) => v,
            Err(msg) => return CommandOutput::usage_error("ls", msg, "ls -al PATH"),
        };
        let [path] = paths.as_slice() else {
            return list_many(&paths, show_hidden).await;
        };
        match list_path(path, show_hidden).await {
            Ok(Listing::Directory(rows) | Listing::Single(rows)) => {
                CommandOutput::ok(rows.into_bytes())
            }
            Err(msg) => CommandOutput::failed(1, msg),
        }
    }
}

enum Listing {
    Directory(String),
    Single(String),
}

fn parse_flags(argv: &[String]) -> Result<(bool, Vec<&str>), String> {
    let mut show_hidden = false;
    let mut paths = Vec::new();
    for arg in argv {
        match arg.strip_prefix('-') {
            Some(flags) if !flags.is_empty() && paths.is_empty() => {
                for ch in flags.chars() {
                    match ch {
                        'a' => show_hidden = true,
                        'l' => {}
                        other => return Err(format!("unknown flag '-{other}'")),
                    }
                }
            }
            _ => paths.push(arg.as_str()),
        }
    }
    if paths.is_empty() {
        paths.push(".");
    }
    Ok((show_hidden, paths))
}

fn kind_and_size(md: &Metadata) -> (&'static str, u64) {
    let file_type = md.file_type();
    let kind = if file_type.is_symlink() {
        "symlink"
    } else if file_type.is_dir() {
        "dir"
    } else {
        "file"
    };
    (kind, md.len())
}

async fn list_many(paths: &[&str], show_hidden: bool) -> CommandOutput {
    let mut out = CommandOutput::ok(Vec::new());
    let mut files = Vec::new();
    let mut directories = Vec::new();
    for &path in paths {
        match list_path(path, show_hidden).await {
            Ok(Listing::Single(row)) => files.push((path, row)),
            Ok(Listing::Directory(rows)) => directories.push((path, rows)),
            Err(msg) => {
                out.stderr.extend_from_slice(msg.as_bytes());
                out.exit_code = 1;
            }
        }
    }
    files.sort_by(|a, b| a.0.cmp(b.0));
    directories.sort_by(|a, b| a.0.cmp(b.0));
    for (_, row) in files {
        out.stdout.extend_from_slice(row.as_bytes());
    }
    for (path, rows) in directories {
        if !out.stdout.is_empty() {
            out.stdout.push(b'\n');
        }
        out.stdout
            .extend_from_slice(format!("{path}:\n{rows}").as_bytes());
    }
    out
}

async fn list_path(path: &str, show_hidden: bool) -> Result<Listing, String> {
    let describe_error = |e: io::Error| io_error_nav("ls", path, &e);
    let reader = match tokio::fs::read_dir(path).await {
        Ok(reader) => reader,
        Err(e) if e.kind() == ErrorKind::NotADirectory => {
            return file_row(path)
                .await
                .map(Listing::Single)
                .map_err(describe_error);
        }
        Err(e) => return Err(describe_error(e)),
    };
    let mut rows = read_rows(reader, show_hidden)
        .await
        .map_err(describe_error)?;
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(Listing::Directory(
        rows.into_iter()
            .map(|(name, kind, size)| format!("{kind}\t{size}\t{name}\n"))
            .collect(),
    ))
}

async fn file_row(path: &str) -> io::Result<String> {
    let md = tokio::fs::symlink_metadata(path).await?;
    let (kind, size) = kind_and_size(&md);
    Ok(format!("{kind}\t{size}\t{path}\n"))
}

async fn read_rows(
    mut reader: ReadDir,
    show_hidden: bool,
) -> io::Result<Vec<(String, &'static str, u64)>> {
    let mut rows = Vec::new();
    while let Some(entry) = reader.next_entry().await? {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !show_hidden && name.starts_with('.') {
            continue;
        }
        let (kind, size) = tokio::fs::symlink_metadata(entry.path())
            .await
            .map_or(("file", 0), |md| kind_and_size(&md));
        rows.push((name, kind, size));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests;
