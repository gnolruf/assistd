//! `BranchHostReads`: what the history of the user's active branch has read from this machine.

use tokio::sync::Mutex;
use tracing::warn;

use assistd_memory::{BranchId, HistoryRow, HistoryScope, MessageKind, TranscriptStore};
use assistd_tools::{HostReads, ToolRegistry};

use super::branches::tool_call_arguments;

/// The reads of the branch the last turn ran on, shared by its turns. Another branch's are
/// found again in its saved rows.
#[derive(Debug, Default)]
pub(super) struct BranchHostReads {
    tracked: Mutex<Option<(Option<BranchId>, HostReads)>>,
}

impl BranchHostReads {
    /// The reads of `branch`, `None` being a session with nothing saved: those the last turn
    /// recorded if it ran on `branch`, else what the `offered` tools find in its saved rows.
    pub(super) async fn of(
        &self,
        branch: Option<BranchId>,
        transcripts: &dyn TranscriptStore,
        offered: &ToolRegistry,
    ) -> HostReads {
        let mut tracked = self.tracked.lock().await;
        if let Some((of, host_reads)) = tracked.as_ref()
            && *of == branch
        {
            return host_reads.clone();
        }
        let host_reads = match branch {
            Some(branch) => saved_host_reads(branch, transcripts, offered).await,
            None => HostReads::default(),
        };
        *tracked = Some((branch, host_reads.clone()));
        host_reads
    }
}

/// A branch that cannot be loaded counts as having read this machine.
async fn saved_host_reads(
    branch: BranchId,
    transcripts: &dyn TranscriptStore,
    offered: &ToolRegistry,
) -> HostReads {
    let seen = match transcripts.load_branch(branch, HistoryScope::Whole).await {
        Ok(history) => rows_read_host(&history.rows, offered),
        Err(e) => {
            warn!(
                target: "assistd::policy",
                error = %e,
                "load_branch failed; treating the branch as having read this machine"
            );
            true
        }
    };
    HostReads::known(seen)
}

/// Whether any tool result in `rows` came from a call that `offered` says reads this machine.
/// A call to a tool no longer offered, or whose arguments were not kept, counts as one.
fn rows_read_host(rows: &[HistoryRow], offered: &ToolRegistry) -> bool {
    let arguments = tool_call_arguments(rows);
    rows.iter().any(|row| match &row.message.kind {
        MessageKind::Tool {
            call_id, tool_name, ..
        } => match (offered.get(tool_name), arguments.get(call_id)) {
            (Some(tool), Some(args)) => tool.reads_host(args),
            _ => true,
        },
        MessageKind::User | MessageKind::Assistant { .. } => false,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use assistd_memory::{MessageId, PersistedMessage, PersistedToolCall, Seq, ToolOutcome};
    use assistd_tools::commands::WebCommand;
    use assistd_tools::{
        AlwaysAllowGate, ApprovalGate, Approvals, CommandRegistry, OutputCap, PresentSpec, RunTool,
    };

    use super::*;

    /// A step calling `tool` with `command`, and its result, as saved rows.
    fn saved_call(id: i64, tool: &str, command: &str) -> [HistoryRow; 2] {
        let call = PersistedToolCall {
            id: id.to_string(),
            name: tool.into(),
            arguments: json!({ "command": command }),
        };
        let outcome = ToolOutcome {
            exit_code: 0,
            duration_ms: 1,
        };
        [
            PersistedMessage::assistant_tool_calls("", vec![call], None),
            PersistedMessage::tool_result("output", id.to_string(), tool, outcome),
        ]
        .map(|message| HistoryRow {
            message_id: MessageId(id),
            seq: Seq(id),
            message,
            turn_stats: None,
        })
    }

    #[test]
    fn a_saved_branch_has_read_this_machine_unless_every_call_only_reached_the_web() {
        let gate = ApprovalGate::new(Arc::new(AlwaysAllowGate), Arc::new(Approvals::unsaved()));
        let mut commands = CommandRegistry::new();
        commands.register(WebCommand::new(gate, None));
        let output_cap = Arc::new(OutputCap::new(PresentSpec::default()));
        let mut offered = ToolRegistry::with_output_cap(Arc::clone(&output_cap));
        offered.register(RunTool::new(Arc::new(commands), output_cap));
        let fetch = saved_call(1, "run", "web fetch https://example.com/");

        let web_only = fetch.to_vec();
        let with_gone_tool = [fetch.clone(), saved_call(2, "mcp__notes__read", "")].concat();
        let mut without_arguments = fetch.to_vec();
        without_arguments.remove(0);

        assert!(!rows_read_host(&web_only, &offered));
        assert!(rows_read_host(&with_gone_tool, &offered));
        assert!(rows_read_host(&without_arguments, &offered));
    }
}
