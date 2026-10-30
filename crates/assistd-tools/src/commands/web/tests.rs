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
    WebCommand::under(EgressPolicy::any_port(hosts, |ip| ip.is_loopback()))
}

fn allowing_gate() -> ApprovalGate {
    ApprovalGate::new(Arc::new(AlwaysAllowGate), Arc::new(Approvals::unsaved()))
}

fn allowing() -> WebCommand {
    loopback_only(allowing_gate())
}

async fn fetch(cmd: &WebCommand, url: &str) -> CommandOutput {
    cmd.run(CommandInput {
        args: vec!["fetch".to_string(), url.to_string()],
        stdin: None,
        scope: crate::FileScope::Host,
    })
    .await
}

fn stderr(out: &CommandOutput) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[tokio::test]
async fn rejects_non_http_scheme() {
    let out = fetch(&allowing(), "file:///etc/hostname").await;
    assert_eq!(out.exit_code, 2);
    assert_eq!(
        stderr(&out),
        "[error] web: only http(s):// URLs are allowed: file:///etc/hostname. \
         Use: web fetch https://... or web fetch http://...\n"
    );
}

#[tokio::test]
async fn declined_fetch_exits_126_without_connecting() {
    let cmd = WebCommand::new(
        ApprovalGate::new(Arc::new(DenyAllGate), Arc::new(Approvals::unsaved())),
        None,
    );
    let out = fetch(&cmd, "http://example.com/?secret=x").await;
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
    let out = fetch(
        &loopback_only(ApprovalGate::new(gate.clone(), hosts)),
        &format!("http://{addr}/"),
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
    let out = fetch(&cmd, &url).await;
    server.await.unwrap();
    assert_eq!(out.exit_code, 0, "{}", stderr(&out));
    assert!(hosts.contains("127.0.0.1"));
    let asked = gate.requests();
    let [request] = asked.as_slice() else {
        panic!("expected one prompt, got {asked:?}");
    };
    assert_eq!(request.tool, "web");
    assert_eq!(request.script, format!("web fetch {url}"));
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
    let out = fetch(&allowing(), &format!("http://{addr}/")).await;
    server.await.unwrap();
    assert_eq!(out.exit_code, 0, "{}", stderr(&out));
    assert_eq!(out.stdout, b"landed");
}

#[tokio::test]
async fn redirect_to_an_unapproved_host_is_refused() {
    let (addr, server) = serve(vec![redirect_to("http://localhost:1/?leak=x")]).await;
    let out = fetch(&allowing(), &format!("http://{addr}/")).await;
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
    let out = fetch(&allowing(), &format!("http://{addr}/")).await;
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
    let cmd = WebCommand::new(
        ApprovalGate::new(gate.clone(), Arc::new(Approvals::unsaved())),
        None,
    );
    let out = fetch(&cmd, "http://[::ffff:169.254.169.254]/latest/").await;
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
    let cmd = WebCommand::new(allowing_gate(), None);
    let out = fetch(&cmd, "http://localhost:1/").await;
    assert_eq!(out.exit_code, 1);
    assert!(
        stderr(&out).contains("localhost resolves only to non-public addresses"),
        "{}",
        stderr(&out)
    );
}

#[tokio::test]
async fn undeclared_oversized_body_stops_at_the_cap() {
    let mut raw = b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n".to_vec();
    raw.resize(raw.len() + BODY_MAX + 1, b'x');
    let (addr, server) = serve(vec![raw]).await;
    let out = fetch(&allowing(), &format!("http://{addr}/")).await;
    server.abort();
    assert_eq!(out.exit_code, 1);
    assert!(
        stderr(&out).contains(&format!("response body exceeded {BODY_MAX} bytes")),
        "{}",
        stderr(&out)
    );
}

#[test]
fn accept_language_follows_the_posix_locale() {
    let cases = [
        (Some("en_GB.UTF-8"), "en-GB,en;q=0.5"),
        (Some("de_DE@euro"), "de-DE,en;q=0.5"),
        (Some("fr"), "fr,en;q=0.5"),
        (Some("en"), "en"),
        (Some("C.UTF-8"), "en"),
        (None, "en"),
    ];
    for (locale, expected) in cases {
        assert_eq!(accept_language(locale), expected, "{locale:?}");
    }
}

/// Fetches `page`, served as `content_type`, once per entry of `flag_sets`.
async fn fetch_page(content_type: &str, page: &[u8], flag_sets: &[&[&str]]) -> Vec<String> {
    let html = response(
        &format!("HTTP/1.1 200 OK\r\nContent-Type: {content_type}"),
        page,
    );
    let (addr, server) = serve(vec![html; flag_sets.len()]).await;
    let mut bodies = Vec::new();
    for flags in flag_sets {
        let args = std::iter::once("fetch")
            .chain(flags.iter().copied())
            .map(String::from)
            .chain([format!("http://{addr}/blog/post")])
            .collect();
        let out = allowing()
            .run(CommandInput {
                args,
                stdin: None,
                scope: crate::FileScope::Host,
            })
            .await;
        assert_eq!(out.exit_code, 0, "{}", stderr(&out));
        bodies.push(String::from_utf8_lossy(&out.stdout).replace(&addr.to_string(), "HOST"));
    }
    server.await.unwrap();
    bodies
}

#[tokio::test]
async fn html_page_is_read_as_text_unless_raw() {
    const PAGE: &str = "<!doctype html><html><head><title>Rust &amp; You</title>\
        <style>body { color: red }</style><script>trackVisitor()</script></head>\
        <body><h1>Intro</h1><p>Read <a href=\"/docs/guide?x=1\">the guide</a> \
        or <a href=\"#top\">jump up</a>.</p><ul><li>one</li></ul></body></html>";
    let [text, raw] = fetch_page(
        "text/html; charset=utf-8",
        PAGE.as_bytes(),
        &[&[], &["--raw"]],
    )
    .await
    .try_into()
    .unwrap();
    assert_eq!(
        text,
        "Rust & You\n\n# Intro\n\nRead the guide (http://HOST/docs/guide?x=1) or jump up.\n- one\n"
    );
    assert_eq!(raw, PAGE);
}

#[tokio::test]
async fn legacy_charsets_are_decoded() {
    let cases: [(&str, &[u8], &str); 3] = [
        ("text/html; Charset=ISO-8859-1", b"<p>caf\xe9</p>", "café"),
        (
            "text/html",
            b"<meta charset=\"windows-1252\"><p>caf\xe9</p>",
            "café",
        ),
        (
            "text/html",
            b"<meta http-equiv=\"Content-Type\" content=\"text/html; charset=Shift_JIS\">\
              <p>\x93\xfa\x96\x7b</p>",
            "日本",
        ),
    ];
    for (content_type, page, expected) in cases {
        let [text] = fetch_page(content_type, page, &[&[]])
            .await
            .try_into()
            .unwrap();
        assert_eq!(text.trim(), expected, "{content_type}");
    }
}

#[tokio::test]
async fn article_page_keeps_its_body_and_drops_chrome_unless_full() {
    const PAGE: &str = include_str!("fixtures/article.html");
    let [article, full] = fetch_page("text/html", PAGE.as_bytes(), &[&[], &["--full"]])
        .await
        .try_into()
        .unwrap();
    assert!(
        article.starts_with(
            "Building a Ferris Wheel | Field Notes\n\n# Building a Ferris Wheel\n\n\
             The first Ferris wheel"
        ),
        "{article}"
    );
    assert!(article.contains("about twenty minutes"), "{article}");
    for chrome in ["Subscribe", "Shop", "Popular posts", "Comments", "Privacy"] {
        assert!(!article.contains(chrome), "{chrome} in {article}");
        assert!(full.contains(chrome), "{chrome} missing from {full}");
    }
}

#[tokio::test]
async fn non_article_page_falls_back_to_the_pruned_page() {
    const PAGE: &str = include_str!("fixtures/results.html");
    let [text] = fetch_page("text/html", PAGE.as_bytes(), &[&[]])
        .await
        .try_into()
        .unwrap();
    for result in ["Ferris wheel - Wikipedia", "Ain Dubai", "The London Eye"] {
        assert!(text.contains(result), "{result} missing from {text}");
    }
    assert!(!text.contains("Next page"), "{text}");
}
