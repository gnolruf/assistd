//! Command-execution policy: the denylist, command review, confirmation
//! gates, and the bwrap sandbox.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tracing::warn;

use crate::command::{CommandOutput, Hint, error_line};
use crate::exec::POLICY_DENIED_EXIT;

mod allowlist;
mod confirm;
mod review;
mod sandbox;
mod shell;

pub use allowlist::{APPROVALS_FILE, Allowlist, AllowlistError, SearchPath};
#[cfg(any(test, feature = "test-support"))]
pub use confirm::{AlwaysAllowGate, DenyAllGate};
pub use confirm::{
    Approval, CONFIRM_ROUTER, CONFIRM_TIMEOUT, ConfirmRouter, ConfirmationGate,
    ConfirmationRequest, IpcConfirmationGate, MAX_PENDING_CONFIRMS, NoPendingConfirm,
    inherit_confirm_router,
};
pub use review::{Confirmation, DestructivePattern, Rules, check_argv, check_script};
pub use sandbox::{
    Protected, ResolvedSandboxMode, SandboxAccess, SandboxError, SandboxInfo, SandboxRequest,
    probe_sandbox,
};

/// Policy for the commands that spawn subprocesses. A command runs
/// unasked only when [`check_script`] clears it; the denylist refuses
/// outright.
#[derive(Debug, Clone)]
pub struct BashPolicyCfg {
    pub timeout: Duration,
    pub denylist: Vec<String>,
    pub destructive_patterns: Vec<DestructivePattern>,
    /// Programs that run without confirmation, shared by every
    /// subprocess-spawning command so an approval applies to all of them.
    pub allowlist: Arc<Allowlist>,
    /// Directories a command may not name without confirmation.
    pub protected: Vec<PathBuf>,
}

impl Default for BashPolicyCfg {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            denylist: Vec::new(),
            destructive_patterns: Vec::new(),
            allowlist: Arc::new(Allowlist::unsaved(
                assistd_config::defaults::default_bash_allowed_programs(),
                SandboxInfo::none().search_path(),
            )),
            protected: Vec::new(),
        }
    }
}

impl BashPolicyCfg {
    /// The rules [`check_script`] and [`check_argv`] apply.
    pub fn rules(&self) -> Rules<'_> {
        Rules {
            patterns: &self.destructive_patterns,
            allowlist: &self.allowlist,
            protected: &self.protected,
        }
    }
}

/// The policy, sandbox and gate a command runs model-chosen argv under.
pub(crate) struct SubprocessPolicy {
    pub(crate) cfg: Arc<BashPolicyCfg>,
    pub(crate) sandbox: Arc<SandboxInfo>,
    pub(crate) gate: Arc<dyn ConfirmationGate>,
}

impl SubprocessPolicy {
    /// Refuse `script` when it hits the denylist, or when the review's
    /// `confirmation` is set and the gate declines. `tool` and `op` name
    /// the command and operation in the error line.
    pub(crate) async fn authorize(
        &self,
        tool: &str,
        op: &str,
        script: &str,
        confirmation: Option<Confirmation>,
    ) -> Result<(), CommandOutput> {
        self.refuse_denylisted(tool, op, script)?;
        let Some(confirmation) = confirmation else {
            return Ok(());
        };
        if self.confirmed(tool, script, &confirmation).await {
            Ok(())
        } else {
            Err(cancelled(tool, op, &confirmation))
        }
    }

    fn refuse_denylisted(&self, tool: &str, op: &str, script: &str) -> Result<(), CommandOutput> {
        let Some(pat) = matches_denylist(script, &self.cfg.denylist) else {
            return Ok(());
        };
        warn!(
            target: "assistd::policy",
            tool = %tool,
            script = %script,
            matched = %pat,
            "denied by denylist"
        );
        Err(CommandOutput::failed(
            POLICY_DENIED_EXIT,
            error_line(
                tool,
                format_args!("{op} denied by policy. Matched denylist pattern: {pat}"),
                Hint::Try,
                "a non-destructive alternative",
            )
            .into_bytes(),
        ))
    }

    /// Ask the gate; an "always" answer also adds the offered programs to
    /// the allowlist.
    async fn confirmed(&self, tool: &str, script: &str, confirmation: &Confirmation) -> bool {
        let offered = confirmation.always_allow().to_vec();
        let approval = self
            .gate
            .confirm(ConfirmationRequest {
                tool: tool.to_string(),
                script: script.to_string(),
                matched_pattern: confirmation.to_string(),
                always_allow: offered.clone(),
            })
            .await;
        match approval {
            Approval::Always if !offered.is_empty() => {
                if let Err(e) = self.cfg.allowlist.approve(&offered).await {
                    warn!(
                        target: "assistd::policy",
                        error = %e,
                        "approval holds until the daemon exits but was not saved"
                    );
                }
                true
            }
            Approval::Once | Approval::Always => true,
            Approval::Deny => false,
        }
    }
}

fn cancelled(tool: &str, op: &str, confirmation: &Confirmation) -> CommandOutput {
    let reason = match confirmation {
        Confirmation::Pattern(pattern) => format!("Matched destructive pattern: {pattern}"),
        Confirmation::Unverifiable(why) => {
            format!("Could not rule out a destructive command: {why}")
        }
        Confirmation::Unlisted { programs, .. } => {
            format!("Not on the allowlist: {}", programs.join(", "))
        }
    };
    CommandOutput::failed(
        POLICY_DENIED_EXIT,
        error_line(
            tool,
            format_args!("{op} cancelled by user. {reason}"),
            Hint::Try,
            "a different approach",
        )
        .into_bytes(),
    )
}

/// The first of `patterns` found in `script`, ignoring ASCII case. A
/// pattern ending in a non-alphanumeric only matches where a shell word
/// ends (`rm -rf /` misses `rm -rf /tmp`); others also match a longer word
/// (`mkfs` hits `mkfs.ext4`). Empty patterns never match.
pub fn matches_denylist<'a>(script: &str, patterns: &'a [String]) -> Option<&'a str> {
    let haystack = script.to_ascii_lowercase();
    patterns
        .iter()
        .find(|pattern| denylist_hit(&haystack, &pattern.to_ascii_lowercase()))
        .map(String::as_str)
}

fn denylist_hit(haystack: &str, needle: &str) -> bool {
    let Some(last) = needle.chars().next_back() else {
        return false;
    };
    if last.is_ascii_alphanumeric() {
        return haystack.contains(needle);
    }
    haystack
        .char_indices()
        .filter_map(|(at, _)| haystack[at..].strip_prefix(needle))
        .any(ends_word)
}

fn ends_word(rest: &str) -> bool {
    rest.chars()
        .next()
        .is_none_or(|c| c.is_whitespace() || ";&|()<>'\"`".contains(c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn denylist_returns_first_case_insensitive_substring_match() {
        let patterns = ["", "rm -rf /", "mkfs", "> /dev/nvme"].map(String::from);
        for (script, expected) in [
            ("rm -rf /", Some("rm -rf /")),
            ("RM -RF /", Some("rm -rf /")),
            ("sudo mkfs.ext4 /dev/sda1", Some("mkfs")),
            ("mkfs /dev/sda1 && rm -rf /", Some("rm -rf /")),
            ("ls -l /tmp", None),
            ("rm -rf / --no-preserve-root", Some("rm -rf /")),
            ("rm -rf /;true", Some("rm -rf /")),
            ("bash -c 'rm -rf /'", Some("rm -rf /")),
            ("rm -rf /tmp/build", None),
            ("rm -rf /tmp/build && rm -rf /", Some("rm -rf /")),
            ("echo x > /dev/nvme0n1", Some("> /dev/nvme")),
        ] {
            assert_eq!(matches_denylist(script, &patterns), expected, "{script:?}");
        }
    }
}
