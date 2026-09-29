use std::net::SocketAddr;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

use super::*;
use crate::commands::test_support::RecordingGate;
use crate::policy::{AlwaysAllowGate, Approval, DenyAllGate};

/// A server answering one connection per entry of `responses`, in order,
/// each with that raw HTTP response.
async fn serve(responses: Vec<Vec<u8>>) -> (SocketAddr, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    (addr, answer(listener, responses))
}

fn answer(listener: TcpListener, responses: Vec<Vec<u8>>) -> JoinHandle<()> {
    tokio::spawn(async move {
        for response in responses {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf).await;
            let _ = stream.write_all(&response).await;
        }
    })
}

fn response(status_line: &str, body: &[u8]) -> Vec<u8> {
    let mut out = format!(
        "{status_line}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    out.extend_from_slice(body);
    out
}

fn redirect_to(location: &str) -> Vec<u8> {
    format!(
        "HTTP/1.1 302 Found\r\nLocation: {location}\r\nContent-Length: 0\r\n\
         Connection: close\r\n\r\n"
    )
    .into_bytes()
}

fn allowing() -> WebCommand {
    WebCommand::new(Arc::new(AlwaysAllowGate), Arc::new(Approvals::unsaved()))
}

async fn run_web(cmd: &WebCommand, args: &[&str]) -> CommandOutput {
    cmd.run(CommandInput {
        args: args.iter().map(ToString::to_string).collect(),
        stdin: None,
    })
    .await
}

fn stderr(out: &CommandOutput) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[tokio::test]
async fn fetches_response_body() {
    let (addr, server) = serve(vec![response("HTTP/1.1 200 OK", b"hello from server")]).await;
    let out = run_web(&allowing(), &[&format!("http://{addr}/")]).await;
    server.await.unwrap();
    assert_eq!(out.exit_code, 0, "{}", stderr(&out));
    assert_eq!(out.stdout, b"hello from server");
}

#[tokio::test]
async fn non_2xx_exits_1_with_status_in_stderr() {
    let (addr, server) = serve(vec![response("HTTP/1.1 404 Not Found", b"missing")]).await;
    let url = format!("http://{addr}/");
    let out = run_web(&allowing(), &[&url]).await;
    server.await.unwrap();
    assert_eq!(out.exit_code, 1);
    assert!(out.stdout.is_empty());
    assert_eq!(
        stderr(&out),
        format!(
            "[error] web: HTTP 404 Not Found: {url}. \
             Try: a different URL or check the endpoint is reachable\n"
        )
    );
}

#[tokio::test]
async fn rejects_non_http_scheme() {
    let out = run_web(&allowing(), &["file:///etc/hostname"]).await;
    assert_eq!(out.exit_code, 2);
    assert_eq!(
        stderr(&out),
        "[error] web: only http(s):// URLs are allowed: file:///etc/hostname. \
         Use: web https://... or web http://...\n"
    );
}

#[tokio::test]
async fn connection_failure_to_reserved_port_exits_1() {
    let cmd = WebCommand::with_timeout(
        Arc::new(AlwaysAllowGate),
        Arc::new(Approvals::unsaved()),
        Duration::from_millis(200),
    );
    let out = run_web(&cmd, &["http://127.0.0.1:1/"]).await;
    assert_eq!(out.exit_code, 1);
    assert!(
        stderr(&out).starts_with("[error] web: transport error: http://127.0.0.1:1/: "),
        "{}",
        stderr(&out)
    );
}

#[tokio::test]
async fn declined_fetch_exits_126_without_connecting() {
    let cmd = WebCommand::new(Arc::new(DenyAllGate), Arc::new(Approvals::unsaved()));
    let out = run_web(&cmd, &["http://127.0.0.1:1/?secret=x"]).await;
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
    assert_eq!(
        stderr(&out),
        "[error] web: fetch cancelled by user: http://127.0.0.1:1/?secret=x. \
         Try: answering without this page\n"
    );
}

#[tokio::test]
async fn approved_host_is_fetched_without_asking() {
    let hosts = Arc::new(Approvals::unsaved());
    hosts.approve("127.0.0.1").await.expect("unsaved approve");
    let (addr, server) = serve(vec![response("HTTP/1.1 200 OK", b"ok")]).await;
    let gate = RecordingGate::answering(Approval::Deny);
    let out = run_web(
        &WebCommand::new(gate.clone(), hosts),
        &[&format!("http://{addr}/")],
    )
    .await;
    server.await.unwrap();
    assert_eq!(out.exit_code, 0, "{}", stderr(&out));
    assert!(gate.requests().is_empty());
}

#[tokio::test]
async fn always_answer_approves_the_host_for_good() {
    let gate = RecordingGate::answering(Approval::Always);
    let hosts = Arc::new(Approvals::unsaved());
    let cmd = WebCommand::new(gate.clone(), Arc::clone(&hosts));
    let (addr, server) = serve(vec![response("HTTP/1.1 200 OK", b"ok")]).await;
    let url = format!("http://{addr}/page");
    let out = run_web(&cmd, &[&url]).await;
    server.await.unwrap();
    assert_eq!(out.exit_code, 0, "{}", stderr(&out));
    assert!(hosts.contains("127.0.0.1"));
    let asked = gate.requests();
    let [request] = asked.as_slice() else {
        panic!("expected one prompt, got {asked:?}");
    };
    assert_eq!(request.tool, "web");
    assert_eq!(request.script, format!("web {url}"));
    assert_eq!(request.always_allow, ["127.0.0.1"]);
}

#[tokio::test]
async fn redirect_within_the_same_host_is_followed() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = answer(
        listener,
        vec![
            redirect_to(&format!("http://{addr}/next")),
            response("HTTP/1.1 200 OK", b"landed"),
        ],
    );
    let out = run_web(&allowing(), &[&format!("http://{addr}/")]).await;
    server.await.unwrap();
    assert_eq!(out.exit_code, 0, "{}", stderr(&out));
    assert_eq!(out.stdout, b"landed");
}

#[tokio::test]
async fn redirect_to_an_unapproved_host_is_refused() {
    let (addr, server) = serve(vec![redirect_to("http://localhost:1/?leak=x")]).await;
    let out = run_web(&allowing(), &[&format!("http://{addr}/")]).await;
    server.await.unwrap();
    assert_eq!(out.exit_code, 1);
    assert!(
        stderr(&out).contains("redirect to unapproved host localhost"),
        "{}",
        stderr(&out)
    );
}

#[tokio::test]
async fn declared_oversized_body_is_refused() {
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        BODY_MAX + 1
    );
    let (addr, server) = serve(vec![head.into_bytes()]).await;
    let url = format!("http://{addr}/");
    let out = run_web(&allowing(), &[&url]).await;
    server.await.unwrap();
    assert_eq!(out.exit_code, 1);
    assert!(
        stderr(&out).contains(&format!("response body exceeded {BODY_MAX} bytes")),
        "{}",
        stderr(&out)
    );
}

#[tokio::test]
async fn undeclared_oversized_body_stops_at_the_cap() {
    let mut raw = b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n".to_vec();
    raw.resize(raw.len() + BODY_MAX + 1, b'x');
    let (addr, server) = serve(vec![raw]).await;
    let out = run_web(&allowing(), &[&format!("http://{addr}/")]).await;
    server.abort();
    assert_eq!(out.exit_code, 1);
    assert!(
        stderr(&out).contains(&format!("response body exceeded {BODY_MAX} bytes")),
        "{}",
        stderr(&out)
    );
}
