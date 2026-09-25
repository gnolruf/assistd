use std::fmt::Display;
use std::time::Duration;

use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput, Hint, error_line};

/// Hard cap on response bytes.
pub const BODY_MAX: usize = 10 * 1024 * 1024;

const UNREACHABLE: &str = "a different URL or check the endpoint is reachable";

/// `web URL`: HTTP GET a URL and return the response body as stdout.
pub struct WebCommand {
    client: reqwest::Client,
}

impl WebCommand {
    /// A command with a 30-second request timeout.
    pub fn new() -> Self {
        Self::with_timeout(Duration::from_secs(30))
    }

    /// A command whose requests time out after `timeout`, with connecting
    /// capped at 10 seconds.
    pub fn with_timeout(timeout: Duration) -> Self {
        let client = reqwest::Client::builder()
            .no_proxy()
            .connect_timeout(Duration::from_secs(10))
            .timeout(timeout)
            .build()
            .expect("reqwest client builds with valid config");
        Self { client }
    }
}

impl Default for WebCommand {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Command for WebCommand {
    fn name(&self) -> &str {
        "web"
    }

    fn summary(&self) -> &'static str {
        "HTTP GET a URL and return the body as stdout (http/https only)"
    }

    fn help(&self) -> String {
        "usage: web URL\n\
         \n\
         HTTP GET the URL (http or https only) and return the response body \
         as stdout. 30-second timeout; response body capped at 10 MiB.\n\
         \n\
         Non-2xx statuses exit 1 so `||` fallbacks fire; transport errors \
         also exit 1. Exit 2 on usage errors (wrong arg count, non-http scheme).\n"
            .to_string()
    }

    async fn run(&self, input: CommandInput) -> CommandOutput {
        if input.args.is_empty() {
            return CommandOutput::usage(self.help());
        }
        if input.args.len() != 1 {
            return CommandOutput::usage_error(
                "web",
                "expects exactly one URL argument",
                "web <URL>",
            );
        }
        let url = &input.args[0];
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return CommandOutput::usage_error(
                "web",
                format_args!("only http(s):// URLs are allowed: {url}"),
                "web https://... or web http://...",
            );
        }

        let response = match self.client.get(url).send().await {
            Ok(r) => r,
            Err(e) => {
                return fetch_failed(format_args!("transport error: {url}: {e}"), UNREACHABLE);
            }
        };
        let status = response.status();
        if !status.is_success() {
            return fetch_failed(
                format_args!(
                    "HTTP {} {}: {url}",
                    status.as_u16(),
                    status.canonical_reason().unwrap_or("")
                ),
                UNREACHABLE,
            );
        }

        let body = match response.bytes().await {
            Ok(b) => b,
            Err(e) => {
                return fetch_failed(
                    format_args!("body read failed: {url}: {e}"),
                    "re-running or a different URL",
                );
            }
        };
        if body.len() > BODY_MAX {
            return fetch_failed(
                format_args!(
                    "response body exceeded {BODY_MAX} bytes (got {}): {url}",
                    body.len()
                ),
                "a URL path that returns less content",
            );
        }
        CommandOutput::ok(body.to_vec())
    }
}

fn fetch_failed(what: impl Display, recovery: &str) -> CommandOutput {
    CommandOutput::failed(1, error_line("web", what, Hint::Try, recovery).into_bytes())
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::task::JoinHandle;

    use super::*;

    /// A one-shot HTTP server answering a single request with `body`.
    async fn serve_once(
        status_line: &'static str,
        body: &'static [u8],
    ) -> (SocketAddr, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            if let Ok((mut stream, _)) = listener.accept().await {
                let mut buf = [0u8; 1024];
                let _ = stream.read(&mut buf).await;
                let response = format!(
                    "{status_line}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.write_all(body).await;
            }
        });
        (addr, server)
    }

    async fn run_web(cmd: &WebCommand, args: &[&str]) -> CommandOutput {
        cmd.run(CommandInput {
            args: args.iter().map(|s| s.to_string()).collect(),
            stdin: None,
        })
        .await
    }

    #[tokio::test]
    async fn fetches_response_body() {
        let (addr, server) = serve_once("HTTP/1.1 200 OK", b"hello from server").await;
        let out = run_web(&WebCommand::new(), &[&format!("http://{addr}/")]).await;
        server.await.unwrap();
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"hello from server");
    }

    #[tokio::test]
    async fn non_2xx_exits_1_with_status_in_stderr() {
        let (addr, server) = serve_once("HTTP/1.1 404 Not Found", b"missing").await;
        let url = format!("http://{addr}/");
        let out = run_web(&WebCommand::new(), &[&url]).await;
        server.await.unwrap();
        assert_eq!(out.exit_code, 1);
        assert!(out.stdout.is_empty());
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            format!(
                "[error] web: HTTP 404 Not Found: {url}. \
                 Try: a different URL or check the endpoint is reachable\n"
            )
        );
    }

    #[tokio::test]
    async fn rejects_non_http_scheme() {
        let out = run_web(&WebCommand::new(), &["file:///etc/hostname"]).await;
        assert_eq!(out.exit_code, 2);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] web: only http(s):// URLs are allowed: file:///etc/hostname. \
             Use: web https://... or web http://...\n"
        );
    }

    #[tokio::test]
    async fn no_args_emits_usage() {
        let out = run_web(&WebCommand::new(), &[]).await;
        assert_eq!(out.exit_code, 2);
        assert!(out.stdout.starts_with(b"usage: web"), "{out:?}");
    }

    #[tokio::test]
    async fn connection_failure_to_reserved_port_exits_1() {
        let cmd = WebCommand::with_timeout(Duration::from_millis(200));
        let out = run_web(&cmd, &["http://127.0.0.1:1/"]).await;
        assert_eq!(out.exit_code, 1);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.starts_with("[error] web: transport error: http://127.0.0.1:1/: "),
            "{stderr}"
        );
    }
}
