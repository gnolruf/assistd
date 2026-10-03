//! MCP servers run as child processes and spoken to over their
//! stdin/stdout by `rmcp`; stderr is forwarded to tracing.

use std::collections::{BTreeSet, HashMap};
use std::ffi::OsString;
use std::fmt;
use std::io;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use rmcp::ServiceExt;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ClientCapabilities, ClientConfig, Implementation, Tool,
};
use rmcp::service::{Peer, RoleClient, RunningService, ServiceError};
use rustix::process::Signal;
use serde_json::Value;
use tokio::process::{Child, ChildStderr, Command};
use tokio_util::task::AbortOnDropHandle;
use tracing::{debug, info, warn};

use assistd_utils::log_lines::forward_lines;
use assistd_utils::process_group::ProcessGroup;

use crate::McpClient;
use crate::error::McpError;

/// Daemon variables a server inherits besides `LC_*`; anything else,
/// credentials included, must come from [`StdioConfig::env`].
const INHERITED_ENV: &[&str] = &[
    "HOME", "LANG", "LANGUAGE", "LOGNAME", "PATH", "SHELL", "TERM", "TMPDIR", "TZ", "USER",
];

/// How long a shutdown waits for the MCP session to close the child's
/// stdin before signalling the process group.
const SESSION_CLOSE_TIMEOUT: Duration = Duration::from_millis(500);

/// Per-server stdio transport configuration. `Debug` lists env var names,
/// never values.
#[derive(Clone)]
pub struct StdioConfig {
    pub command: String,
    pub args: Vec<String>,
    /// Set on top of the few daemon variables every server inherits.
    pub env: HashMap<String, String>,
    /// Bounds the initialize handshake and every request.
    pub request_timeout: Duration,
    /// Server name used in tracing logs.
    pub label: String,
}

impl StdioConfig {
    /// Config that runs `command` with no args or extra env and a 30s
    /// request timeout.
    pub fn new(label: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            args: Vec::new(),
            env: HashMap::new(),
            request_timeout: Duration::from_secs(30),
            label: label.into(),
        }
    }
}

impl fmt::Debug for StdioConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StdioConfig")
            .field("command", &self.command)
            .field("args", &self.args)
            .field("env_names", &self.env.keys().collect::<BTreeSet<_>>())
            .field("request_timeout", &self.request_timeout)
            .field("label", &self.label)
            .finish()
    }
}

/// [`McpClient`] over a child process's pipes.
#[derive(Debug)]
pub(crate) struct StdioMcpClient {
    peer: Peer<RoleClient>,
    request_timeout: Duration,
}

impl StdioMcpClient {
    /// Spawn the server in its own process group and run the initialize
    /// handshake. Errors if either fails; a failed handshake kills the child.
    pub(crate) async fn spawn(cfg: StdioConfig) -> Result<(Arc<Self>, ChildLifeline), McpError> {
        let (mut child, group) = spawn_child(&cfg)?;
        let stdout = child.stdout.take().expect("stdout piped");
        let stdin = child.stdin.take().expect("stdin piped");
        let stderr = child.stderr.take().expect("stderr piped");
        let stderr_task =
            AbortOnDropHandle::new(tokio::spawn(forward_stderr(stderr, cfg.label.clone())));

        let handshake =
            tokio::time::timeout(cfg.request_timeout, client_config().serve((stdout, stdin))).await;
        let service = match handshake {
            Ok(Ok(service)) => service,
            Ok(Err(e)) => {
                return Err(abandon(child, group, McpError::Initialize(Box::new(e))).await);
            }
            Err(_) => {
                let timeout = McpError::RequestTimeout(cfg.request_timeout);
                return Err(abandon(child, group, timeout).await);
            }
        };

        info!(
            target: "assistd::mcp",
            server = %cfg.label,
            pid = group.id().as_raw_nonzero(),
            "MCP stdio server initialized",
        );
        let client = Arc::new(Self {
            peer: service.peer().clone(),
            request_timeout: cfg.request_timeout,
        });
        let lifeline = ChildLifeline {
            label: cfg.label,
            child,
            group,
            service: Some(service),
            stderr_task,
        };
        Ok((client, lifeline))
    }

    async fn within_timeout<T>(
        &self,
        request: impl Future<Output = Result<T, ServiceError>>,
    ) -> Result<T, McpError> {
        tokio::time::timeout(self.request_timeout, request)
            .await
            .map_err(|_| McpError::RequestTimeout(self.request_timeout))?
            .map_err(McpError::from)
    }
}

#[async_trait]
impl McpClient for StdioMcpClient {
    async fn list_tools(&self) -> Result<Vec<Tool>, McpError> {
        self.within_timeout(self.peer.list_all_tools()).await
    }

    async fn invoke(&self, name: &str, arguments: Value) -> Result<CallToolResult, McpError> {
        let mut params = CallToolRequestParams::new(name.to_owned());
        params.arguments = match arguments {
            Value::Object(arguments) => Some(arguments),
            Value::Null => None,
            _ => return Err(McpError::ArgumentsNotObject),
        };
        self.within_timeout(self.peer.call_tool(params)).await
    }
}

/// The spawned child plus its MCP session. Dropping it SIGKILLs the
/// child's whole process group.
#[derive(Debug)]
pub(crate) struct ChildLifeline {
    label: String,
    child: Child,
    group: ProcessGroup,
    service: Option<RunningService<RoleClient, ClientConfig>>,
    stderr_task: AbortOnDropHandle<()>,
}

impl ChildLifeline {
    /// Resolves when the child exits or its MCP session ends; a child
    /// that can no longer answer counts as dead.
    pub(crate) async fn wait_for_death(&mut self) {
        let service = self.service.take();
        tokio::select! {
            status = self.child.wait() => debug!(
                target: "assistd::mcp",
                server = %self.label,
                "MCP child exited: {status:?}",
            ),
            () = session_end(service) => debug!(
                target: "assistd::mcp",
                server = %self.label,
                "MCP session ended while the child was still alive",
            ),
        }
    }

    /// Close the session, SIGTERM the process group, wait `term_timeout`
    /// for the child, then SIGKILL whatever is left of the group.
    pub(crate) async fn shutdown(self, term_timeout: Duration) {
        let Self {
            label,
            mut child,
            group,
            service,
            stderr_task,
        } = self;
        if let Some(mut service) = service {
            let _ = service.close_with_timeout(SESSION_CLOSE_TIMEOUT).await;
        }
        group.signal(Signal::TERM);
        match tokio::time::timeout(term_timeout, child.wait()).await {
            Ok(Ok(status)) => info!(
                target: "assistd::mcp",
                server = %label,
                "MCP server exited after SIGTERM: {status}",
            ),
            Ok(Err(e)) => warn!(
                target: "assistd::mcp",
                server = %label,
                "MCP server wait error: {e}",
            ),
            Err(_) => {
                warn!(
                    target: "assistd::mcp",
                    server = %label,
                    "MCP server did not exit within {term_timeout:?}; sending SIGKILL",
                );
                group.signal(Signal::KILL);
                let _ = child.wait().await;
            }
        }
        drop(group);
        let _ = tokio::time::timeout(Duration::from_millis(500), stderr_task).await;
    }
}

fn spawn_child(cfg: &StdioConfig) -> Result<(Child, ProcessGroup), McpError> {
    let spawn_error = |source| McpError::Spawn {
        path: cfg.command.clone(),
        source,
    };
    let child = Command::new(&cfg.command)
        .args(&cfg.args)
        .env_clear()
        .envs(inherited_env(std::env::vars_os()))
        .envs(cfg.env.iter())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .process_group(0)
        .spawn()
        .map_err(spawn_error)?;
    let group = ProcessGroup::led_by(&child)
        .ok_or_else(|| spawn_error(io::Error::other("spawned child reported no pid")))?;
    Ok((child, group))
}

fn client_config() -> ClientConfig {
    ClientConfig::new(
        ClientCapabilities::default(),
        Implementation::new("assistd", env!("CARGO_PKG_VERSION")),
    )
}

/// SIGKILL the group of a child whose handshake failed, reap it, and
/// hand back `error`.
async fn abandon(mut child: Child, group: ProcessGroup, error: McpError) -> McpError {
    warn!(target: "assistd::mcp", error = %error, "MCP initialize failed; killing the server");
    drop(group);
    let _ = child.wait().await;
    error
}

async fn session_end(service: Option<RunningService<RoleClient, ClientConfig>>) {
    match service {
        Some(service) => {
            let _ = service.waiting().await;
        }
        None => std::future::pending().await,
    }
}

fn inherited_env(
    vars: impl IntoIterator<Item = (OsString, OsString)>,
) -> impl Iterator<Item = (OsString, OsString)> {
    let inherited = |name: &str| INHERITED_ENV.contains(&name) || name.starts_with("LC_");
    vars.into_iter()
        .filter(move |(name, _)| name.to_str().is_some_and(inherited))
}

async fn forward_stderr(stream: ChildStderr, label: String) {
    let forwarded = forward_lines(
        stream,
        |line| warn!(target: "assistd::mcp", server = %label, "{line}"),
    )
    .await;
    if let Err(e) = forwarded {
        warn!(target: "assistd::mcp", server = %label, "stderr read error: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_lists_env_names_but_not_values() {
        let mut cfg = StdioConfig::new("local", "server");
        cfg.env.insert("API_TOKEN".into(), "hunter2".into());
        let rendered = format!("{cfg:?}");
        assert!(rendered.contains("API_TOKEN"), "{rendered}");
        assert!(!rendered.contains("hunter2"), "{rendered}");
    }

    #[test]
    fn servers_inherit_only_basic_session_and_locale_variables() {
        let vars = [
            "PATH",
            "HOME",
            "LC_ALL",
            "TZ",
            "GITHUB_TOKEN",
            "AWS_SECRET_ACCESS_KEY",
            "SSH_AUTH_SOCK",
        ]
        .map(|name| (OsString::from(name), OsString::from("value")));
        let names: Vec<_> = inherited_env(vars).map(|(name, _)| name).collect();
        assert_eq!(names, ["PATH", "HOME", "LC_ALL", "TZ"]);
    }
}
