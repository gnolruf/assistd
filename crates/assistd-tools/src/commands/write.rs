use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput, error_line, io_error_nav};

/// Exit code for policy denial. POSIX "command found but not executable" is
/// the closest semantic match to "we recognize the command but refuse it".
/// Shared with `BashCommand` so the LLM sees a consistent signal.
const POLICY_DENIED_EXIT: i32 = 126;

/// Write-command policy, constructed by `assistd-core::build_tools` from
/// `config.tools.write` at daemon startup. Path strings are canonicalized
/// once here; runtime checks are fast `Path::starts_with` comparisons.
///
/// The allowlist is non-empty by construction: a policy with no permitted
/// prefixes would gate nothing, so that state is not representable.
#[derive(Debug, Clone)]
pub struct WritePolicyCfg {
    first: PathBuf,
    rest: Vec<PathBuf>,
}

impl WritePolicyCfg {
    /// Construct a policy from canonicalized writable path prefixes, or
    /// `None` if none were given.
    pub fn new(writable_paths: Vec<PathBuf>) -> Option<Self> {
        let mut prefixes = writable_paths.into_iter();
        let first = prefixes.next()?;
        Some(Self {
            first,
            rest: prefixes.collect(),
        })
    }

    /// The permitted path prefixes, in configuration order.
    pub fn prefixes(&self) -> impl Iterator<Item = &PathBuf> {
        std::iter::once(&self.first).chain(&self.rest)
    }
}

/// `write PATH [CONTENT...]`: write to PATH, subject to the configured
/// writable-path allowlist.
///
/// Two shapes:
/// - `echo "hi" | write /tmp/x`: stdin is the content (pipeline form).
/// - `write /tmp/x hello world`: args beyond the path are joined with
///   single spaces and written (convenience form for when the model
///   already has the content inline).
///
/// If both args and stdin are provided, args win and stdin is silently
/// discarded; callers who want stdin should avoid passing extra argv.
///
/// # Policy
///
/// Rejections return exit 126 with a convention-compliant `[error]` line
/// that names the offending path and points at `[tools.write] writable_paths`
/// so the user can widen the allowlist if needed. Relative paths are always
/// rejected because the daemon's cwd is not a meaningful anchor.
pub struct WriteCommand {
    cfg: Arc<WritePolicyCfg>,
}

impl WriteCommand {
    /// Construct a `WriteCommand` with the given policy configuration.
    pub fn new(cfg: Arc<WritePolicyCfg>) -> Self {
        Self { cfg }
    }

    /// Test-only constructor: allowlists the filesystem root, so any
    /// absolute path is writable. Never used in production.
    #[cfg(test)]
    pub fn permissive_for_tests() -> Self {
        Self {
            cfg: Arc::new(
                WritePolicyCfg::new(vec![PathBuf::from("/")]).expect("non-empty allowlist"),
            ),
        }
    }
}

#[async_trait]
impl Command for WriteCommand {
    fn name(&self) -> &str {
        "write"
    }

    fn summary(&self) -> &'static str {
        "write a file (allowlist-gated) from stdin or inline args"
    }

    fn help(&self) -> String {
        "usage: write PATH [CONTENT...]\n\
         \n\
         Write bytes to PATH. Two shapes:\n  \
           `echo \"hi\" | write /tmp/x`   - stdin is the file content (pipeline form)\n  \
           `write /tmp/x hello world`   - args beyond PATH are joined by spaces and written\n\
         \n\
         PATH must be absolute (relative paths are rejected) and must fall \
         under one of the prefixes in `[tools.write] writable_paths`. \
         Tilde expansion is supported. Exit 126 on policy denial, 1 on \
         write failure (permissions, no-such-dir, etc.).\n"
            .to_string()
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        if input.args.is_empty() {
            return Ok(CommandOutput {
                stdout: self.help().into_bytes(),
                stderr: Vec::new(),
                exit_code: 2,
                attachments: Vec::new(),
            });
        }
        let raw_path = input.args[0].clone();
        let content: Vec<u8> = if input.args.len() > 1 {
            input.args[1..].join(" ").into_bytes()
        } else {
            input.stdin
        };

        let home = std::env::var("HOME").ok();
        let write_target: PathBuf = match resolve_for_allowlist(&raw_path, home.as_deref()) {
            Ok(resolved) => {
                if !self
                    .cfg
                    .prefixes()
                    .any(|prefix| resolved.starts_with(prefix))
                {
                    return Ok(CommandOutput::failed(
                        POLICY_DENIED_EXIT,
                        error_line(
                            "write",
                            format_args!("{raw_path}: path not in writable allowlist"),
                            "Check",
                            "[tools.write] writable_paths in config",
                        )
                        .into_bytes(),
                    ));
                }
                resolved
            }
            Err(PathResolveError::Relative) => {
                return Ok(CommandOutput::failed(
                    POLICY_DENIED_EXIT,
                    error_line(
                        "write",
                        format_args!("{raw_path}: relative paths not permitted"),
                        "Try",
                        "an absolute path under an allowlisted directory",
                    )
                    .into_bytes(),
                ));
            }
            Err(PathResolveError::HomeNotSet) => {
                return Ok(CommandOutput::failed(
                    POLICY_DENIED_EXIT,
                    error_line(
                        "write",
                        format_args!("{raw_path}: cannot expand ~ ($HOME not set)"),
                        "Try",
                        "writing an explicit absolute path instead of ~",
                    )
                    .into_bytes(),
                ));
            }
            Err(PathResolveError::AnchorMissing(anchor)) => {
                return Ok(CommandOutput::failed(
                    POLICY_DENIED_EXIT,
                    error_line(
                        "write",
                        format_args!("{raw_path}: cannot resolve ancestor {anchor}"),
                        "Check",
                        "that the directory exists or widen [tools.write] writable_paths",
                    )
                    .into_bytes(),
                ));
            }
        };

        match tokio::fs::write(&write_target, &content).await {
            Ok(()) => Ok(CommandOutput::ok(Vec::new())),
            Err(e) => Ok(CommandOutput::failed(
                1,
                io_error_nav("write", &raw_path, &e).into_bytes(),
            )),
        }
    }
}

/// Failure modes for [`resolve_for_allowlist`], turned into distinct error
/// messages by the caller so the LLM can recover accurately.
#[derive(Debug)]
enum PathResolveError {
    Relative,
    HomeNotSet,
    AnchorMissing(String),
}

fn resolve_for_allowlist(raw: &str, home: Option<&str>) -> Result<PathBuf, PathResolveError> {
    let expanded = expand_tilde(raw, home)?;
    if !expanded.is_absolute() {
        return Err(PathResolveError::Relative);
    }
    let cleaned = lexical_clean(&expanded);
    let (anchor, tail) = split_at_existing(&cleaned);
    let canonical_anchor = std::fs::canonicalize(&anchor)
        .map_err(|_| PathResolveError::AnchorMissing(anchor.to_string_lossy().into_owned()))?;
    Ok(canonical_anchor.join(tail))
}

fn expand_tilde(raw: &str, home: Option<&str>) -> Result<PathBuf, PathResolveError> {
    if let Some(rest) = raw.strip_prefix("~/") {
        let home = home.ok_or(PathResolveError::HomeNotSet)?;
        Ok(PathBuf::from(home).join(rest))
    } else if raw == "~" {
        let home = home.ok_or(PathResolveError::HomeNotSet)?;
        Ok(PathBuf::from(home))
    } else {
        Ok(PathBuf::from(raw))
    }
}

/// Pure-Rust path normalization: collapses `.` and `..` components. Does
/// not touch disk. On absolute paths, a leading `/..` is silently discarded
/// (same as the kernel's behaviour).
pub(crate) fn lexical_clean(path: &Path) -> PathBuf {
    let mut out: Vec<Component<'_>> = Vec::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => match out.last() {
                Some(Component::Normal(_)) => {
                    out.pop();
                }
                Some(Component::RootDir) => {
                    // `/..` == `/`: don't pop past root.
                }
                _ => {
                    // Leading `..` on a relative path: keep, since the
                    // caller (above) already rejects relatives. Still
                    // preserve for completeness
                    out.push(comp);
                }
            },
            _ => out.push(comp),
        }
    }
    let mut result = PathBuf::new();
    for c in out {
        result.push(c.as_os_str());
    }
    result
}

fn split_at_existing(path: &Path) -> (PathBuf, PathBuf) {
    let mut anchor = path.to_path_buf();
    let mut tail = PathBuf::new();
    loop {
        if anchor.exists() {
            return (anchor, tail);
        }
        let Some(parent) = anchor.parent() else {
            return (path.to_path_buf(), PathBuf::new());
        };
        let Some(file_name) = anchor.file_name() else {
            return (anchor, tail);
        };
        let new_tail = if tail.as_os_str().is_empty() {
            PathBuf::from(file_name)
        } else {
            let mut nt = PathBuf::from(file_name);
            nt.push(&tail);
            nt
        };
        tail = new_tail;
        anchor = parent.to_path_buf();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn cfg_from<P: AsRef<Path>>(paths: &[P]) -> Arc<WritePolicyCfg> {
        let abs: Vec<PathBuf> = paths
            .iter()
            .map(|p| std::fs::canonicalize(p.as_ref()).expect("canonicalize tempdir"))
            .collect();
        Arc::new(WritePolicyCfg::new(abs).expect("non-empty allowlist"))
    }

    #[test]
    fn empty_allowlist_yields_no_policy() {
        assert!(WritePolicyCfg::new(Vec::new()).is_none());
    }

    #[tokio::test]
    async fn persists_stdin_to_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("out.txt");
        let out = WriteCommand::new(cfg_from(&[dir.path()]))
            .run(CommandInput {
                args: vec![path.to_string_lossy().into_owned()],
                stdin: b"hi there\n".to_vec(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
        let content = tokio::fs::read(&path).await.unwrap();
        assert_eq!(content, b"hi there\n");
    }

    #[tokio::test]
    async fn persists_args_content_to_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("out.txt");
        let out = WriteCommand::new(cfg_from(&[dir.path()]))
            .run(CommandInput {
                args: vec![
                    path.to_string_lossy().into_owned(),
                    "hello".into(),
                    "world".into(),
                ],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
        let content = tokio::fs::read(&path).await.unwrap();
        assert_eq!(content, b"hello world");
    }

    #[tokio::test]
    async fn args_content_wins_over_stdin() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("out.txt");
        let out = WriteCommand::new(cfg_from(&[dir.path()]))
            .run(CommandInput {
                args: vec![path.to_string_lossy().into_owned(), "args".into()],
                stdin: b"stdin".to_vec(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
        let content = tokio::fs::read(&path).await.unwrap();
        assert_eq!(content, b"args");
    }

    #[tokio::test]
    async fn no_args_errors() {
        let out = WriteCommand::permissive_for_tests()
            .run(CommandInput {
                args: Vec::new(),
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 2);
    }

    #[tokio::test]
    async fn path_only_with_empty_stdin_creates_empty_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("out.txt");
        let out = WriteCommand::new(cfg_from(&[dir.path()]))
            .run(CommandInput {
                args: vec![path.to_string_lossy().into_owned()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
        let content = tokio::fs::read(&path).await.unwrap();
        assert!(content.is_empty());
    }

    #[tokio::test]
    async fn write_rejected_outside_allowlist() {
        let dir = tempdir().unwrap();
        let cmd = WriteCommand::new(cfg_from(&[dir.path()]));
        let out = cmd
            .run(CommandInput {
                args: vec!["/etc/passwd".into(), "oops".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 126);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] write: /etc/passwd: path not in writable allowlist"),
            "{stderr}"
        );
        assert!(
            stderr.contains("Check: [tools.write] writable_paths in config"),
            "{stderr}"
        );
    }

    #[tokio::test]
    async fn write_allowlist_permits_tmp() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("permitted.txt");
        let out = WriteCommand::new(cfg_from(&[dir.path()]))
            .run(CommandInput {
                args: vec![path.to_string_lossy().into_owned(), "ok".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
    }

    #[tokio::test]
    async fn write_allowlist_resolves_dotdot() {
        let dir = tempdir().unwrap();
        let tricky = format!("{}/../../../etc/passwd", dir.path().display());
        let out = WriteCommand::new(cfg_from(&[dir.path()]))
            .run(CommandInput {
                args: vec![tricky, "oops".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 126);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("not in writable allowlist"), "{stderr}");
    }

    #[tokio::test]
    async fn write_allowlist_rejects_relative_path() {
        let dir = tempdir().unwrap();
        let out = WriteCommand::new(cfg_from(&[dir.path()]))
            .run(CommandInput {
                args: vec!["relative.txt".into(), "hi".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 126);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("relative paths not permitted"), "{stderr}");
    }

    #[test]
    fn write_allowlist_expands_tilde() {
        let dir = tempdir().unwrap();
        let home = dir.path().to_str().expect("tempdir path is valid utf-8");
        let resolved =
            resolve_for_allowlist("~/tilde-target.txt", Some(home)).expect("tilde resolves");
        let expected = std::fs::canonicalize(dir.path())
            .expect("tempdir canonicalizes")
            .join("tilde-target.txt");
        assert_eq!(resolved, expected);
    }

    #[test]
    fn expand_tilde_without_home_errors() {
        let err = expand_tilde("~/foo", None).expect_err("missing home should error");
        assert!(matches!(err, PathResolveError::HomeNotSet));
    }

    #[tokio::test]
    async fn write_allowlist_handles_nonexistent_parent() {
        let dir = tempdir().unwrap();
        // tokio::fs::write fails on a missing parent too, so we only assert
        // the allowlist check passes; the underlying write may still fail,
        // but with exit 1 (I/O error), not 126 (policy).
        let target = dir.path().join("newsub").join("file.txt");
        let cmd = WriteCommand::new(cfg_from(&[dir.path()]));
        let out = cmd
            .run(CommandInput {
                args: vec![target.to_string_lossy().into_owned(), "hi".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        // Policy must not reject (126); the I/O error path is exit 1.
        assert_ne!(out.exit_code, 126, "{:?}", out.stderr);
    }

    #[tokio::test]
    async fn unwritable_path_exits_1() {
        let dir = tempdir().unwrap();
        // Keep the allowlist wide: the path is under the tempdir but its
        // direct parent doesn't exist.
        let out = WriteCommand::new(cfg_from(&[dir.path()]))
            .run(CommandInput {
                args: vec![
                    format!("{}/definitely/not/a/writable/path", dir.path().display()),
                    "hi".into(),
                ],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        // Parent-missing is a policy success but an I/O failure; exit 1.
        assert_eq!(out.exit_code, 1);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("[error] write: "), "{stderr}");
    }

    #[test]
    fn lexical_clean_collapses_dotdot() {
        assert_eq!(
            lexical_clean(Path::new("/tmp/foo/../bar")),
            PathBuf::from("/tmp/bar")
        );
        assert_eq!(
            lexical_clean(Path::new("/tmp/./foo")),
            PathBuf::from("/tmp/foo")
        );
        // `/..` stays at root.
        assert_eq!(lexical_clean(Path::new("/..")), PathBuf::from("/"));
    }
}
