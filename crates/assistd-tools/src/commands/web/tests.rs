use std::net::SocketAddr;
use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

use super::*;
use crate::commands::test_support::RecordingGate;
use crate::policy::{AlwaysAllowGate, Approval, ApprovalGate, Approvals, DenyAllGate};

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

/// A command that may reach loopback test servers, and no other address.
fn loopback_only(hosts: ApprovalGate) -> WebCommand {
    WebCommand::connecting_to(|ip| ip.is_loopback(), hosts, Duration::from_secs(30))
}

fn allowing_gate() -> ApprovalGate {
    ApprovalGate::new(Arc::new(AlwaysAllowGate), Arc::new(Approvals::unsaved()))
}

fn allowing() -> WebCommand {
    loopback_only(allowing_gate())
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
async fn declined_fetch_exits_126_without_connecting() {
    let cmd = WebCommand::new(ApprovalGate::new(
        Arc::new(DenyAllGate),
        Arc::new(Approvals::unsaved()),
    ));
    let out = run_web(&cmd, &["http://example.com/?secret=x"]).await;
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
    assert_eq!(
        stderr(&out),
        "[error] web: fetch cancelled by user: http://example.com/?secret=x. \
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
        &loopback_only(ApprovalGate::new(gate.clone(), hosts)),
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
    let cmd = loopback_only(ApprovalGate::new(gate.clone(), Arc::clone(&hosts)));
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
async fn redirect_to_a_non_public_address_is_refused() {
    let (addr, server) = serve(vec![redirect_to("http://10.0.0.1/")]).await;
    let out = run_web(&allowing(), &[&format!("http://{addr}/")]).await;
    server.await.unwrap();
    assert_eq!(out.exit_code, 1);
    assert!(
        stderr(&out).contains("10.0.0.1 is not a public address"),
        "{}",
        stderr(&out)
    );
}

#[tokio::test]
async fn non_public_literal_is_denied_without_asking() {
    let gate = RecordingGate::answering(Approval::Always);
    let cmd = WebCommand::new(ApprovalGate::new(
        gate.clone(),
        Arc::new(Approvals::unsaved()),
    ));
    let out = run_web(&cmd, &["http://[::ffff:169.254.169.254]/latest/"]).await;
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
    assert_eq!(
        stderr(&out),
        "[error] web: fetch denied by policy: ::ffff:169.254.169.254 is not a \
         public address. Try: a URL on the public internet\n"
    );
    assert!(gate.requests().is_empty());
}

#[tokio::test]
async fn name_resolving_only_to_loopback_is_refused() {
    let cmd = WebCommand::new(allowing_gate());
    let out = run_web(&cmd, &["http://localhost:1/"]).await;
    assert_eq!(out.exit_code, 1);
    assert!(
        stderr(&out).contains("localhost resolves only to non-public addresses"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn public_addresses_are_told_from_non_public_ones() {
    let cases: [(&str, bool); 22] = [
        ("1.1.1.1", true),
        ("8.8.8.8", true),
        ("100.63.255.255", true),
        ("172.32.0.1", true),
        ("0.0.0.0", false),
        ("127.0.0.1", false),
        ("10.1.2.3", false),
        ("100.64.0.1", false),
        ("169.254.169.254", false),
        ("172.31.255.255", false),
        ("192.168.1.1", false),
        ("198.18.0.1", false),
        ("224.0.0.1", false),
        ("255.255.255.255", false),
        ("2606:4700:4700::1111", true),
        ("::ffff:8.8.8.8", true),
        ("::1", false),
        ("::", false),
        ("::ffff:127.0.0.1", false),
        ("fd00::1", false),
        ("fe80::1", false),
        ("2001:db8::1", false),
    ];
    for (addr, public) in cases {
        assert_eq!(is_public(addr.parse().unwrap()), public, "{addr}");
    }
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
