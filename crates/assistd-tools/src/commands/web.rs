//! `web URL` — HTTP GET, return response body as stdout. http(s) only,
//! 30s default timeout, 10 MiB body cap. Non-2xx statuses exit 1 so
//! `||` fallbacks fire; transport errors also exit 1.

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput, error_line};

/// Hard cap on response bytes. Mirrors the chain executor's
/// `PIPE_BUF_MAX` so a multi-megabyte page doesn't blow the next
/// stage either.
pub const BODY_MAX: usize = 10 * 1024 * 1024;

pub struct WebCommand {
    client: reqwest::Client,
}

impl WebCommand {
    pub fn new() -> Self {
        Self::with_timeout(Duration::from_secs(30))
    }

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

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        if input.args.is_empty() {
            return Ok(CommandOutput {
                stdout: self.help().into_bytes(),
                stderr: Vec::new(),
                exit_code: 2,
                attachments: Vec::new(),
            });
        }
        if input.args.len() != 1 {
            return Ok(CommandOutput::failed(
                2,
                error_line(
                    "web",
                    "expects exactly one URL argument",
                    "Use",
                    "web <URL>",
                )
                .into_bytes(),
            ));
        }
        let url = &input.args[0];
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Ok(CommandOutput::failed(
                2,
                error_line(
                    "web",
                    format_args!("only http(s):// URLs are allowed: {url}"),
                    "Use",
                    "web https://... or web http://...",
                )
                .into_bytes(),
            ));
        }

        let response = match self.client.get(url).send().await {
            Ok(r) => r,
            Err(e) => {
                return Ok(CommandOutput::failed(
                    1,
                    error_line(
                        "web",
                        format_args!("transport error: {url}: {e}"),
                        "Try",
                        "a different URL or check the endpoint is reachable",
                    )
                    .into_bytes(),
                ));
            }
        };
        let status = response.status();
        if !status.is_success() {
            return Ok(CommandOutput::failed(
                1,
                error_line(
                    "web",
                    format_args!(
                        "HTTP {} {}: {url}",
                        status.as_u16(),
                        status.canonical_reason().unwrap_or("")
                    ),
                    "Try",
                    "a different URL or check the endpoint is reachable",
                )
                .into_bytes(),
            ));
        }

        let body = match response.bytes().await {
            Ok(b) => b,
            Err(e) => {
                return Ok(CommandOutput::failed(
                    1,
                    error_line(
                        "web",
                        format_args!("body read failed: {url}: {e}"),
                        "Try",
                        "re-running or a different URL",
                    )
                    .into_bytes(),
                ));
            }
        };
        if body.len() > BODY_MAX {
            return Ok(CommandOutput::failed(
                1,
                error_line(
                    "web",
                    format_args!(
                        "response body exceeded {BODY_MAX} bytes (got {}): {url}",
                        body.len()
                    ),
                    "Try",
                    "a URL path that returns less content",
                )
                .into_bytes(),
            ));
        }
        Ok(CommandOutput::ok(body.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// Spin up a one-shot HTTP server that replies with `body`, then return
    /// its address. Handles a single request then closes.
    async fn serve_once(status_line: &'static str, body: &'static [u8]) -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
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
        addr
    }

    #[tokio::test]
    async fn fetches_response_body() {
        let addr = serve_once("HTTP/1.1 200 OK", b"hello from server").await;
        let out = WebCommand::new()
            .run(CommandInput {
                args: vec![format!("http://{addr}/")],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"hello from server");
    }

    #[tokio::test]
    async fn non_2xx_exits_1_with_status_in_stderr() {
        let addr = serve_once("HTTP/1.1 404 Not Found", b"missing").await;
        let out = WebCommand::new()
            .run(CommandInput {
                args: vec![format!("http://{addr}/")],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 1);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("[error] web: HTTP 404"), "{stderr}");
        assert!(stderr.contains("Try: "), "{stderr}");
    }

    #[tokio::test]
    async fn rejects_non_http_scheme() {
        let out = WebCommand::new()
            .run(CommandInput {
                args: vec!["file:///etc/passwd".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] web: only http(s):// URLs are allowed"),
            "{stderr}"
        );
        assert!(stderr.contains("Use: web http"), "{stderr}");
    }

    #[tokio::test]
    async fn no_args_errors() {
        let out = WebCommand::new()
            .run(CommandInput {
                args: Vec::new(),
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 2);
    }

    #[tokio::test]
    async fn connection_failure_exits_1() {
        // 127.0.0.1:1 is reserved and won't answer.
        let out = WebCommand::with_timeout(Duration::from_millis(200))
            .run(CommandInput {
                args: vec!["http://127.0.0.1:1/".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 1);
    }
}
