use std::net::SocketAddr;

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

use super::*;

fn datum(index: usize, embedding: &[f32]) -> EmbedDatum {
    EmbedDatum {
        index,
        embedding: embedding.to_vec(),
    }
}

async fn read_request_body(stream: &mut TcpStream) -> Value {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        let n = stream.read(&mut chunk).await.unwrap();
        assert!(n > 0, "client closed before sending headers");
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
    };
    let headers = String::from_utf8_lossy(&buf[..header_end]).to_ascii_lowercase();
    let len: usize = headers
        .lines()
        .find_map(|l| l.strip_prefix("content-length:"))
        .map(|v| v.trim().parse().unwrap())
        .unwrap_or(0);
    while buf.len() < header_end + len {
        let n = stream.read(&mut chunk).await.unwrap();
        assert!(n > 0, "client closed mid-body");
        buf.extend_from_slice(&chunk[..n]);
    }
    serde_json::from_slice(&buf[header_end..header_end + len]).unwrap()
}

/// Serve one scripted JSON response per connection, returning the request bodies
/// received once the script is exhausted.
async fn serve(responses: Vec<Value>) -> (SocketAddr, JoinHandle<Vec<Value>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let mut bodies = Vec::new();
        for response in responses {
            let (mut stream, _) = listener.accept().await.unwrap();
            bodies.push(read_request_body(&mut stream).await);
            let body = response.to_string();
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(head.as_bytes()).await.unwrap();
            stream.write_all(body.as_bytes()).await.unwrap();
        }
        bodies
    });
    (addr, server)
}

async fn embedder_at(addr: SocketAddr) -> LlamaEmbedder {
    LlamaEmbedder::new(
        &addr.ip().to_string(),
        addr.port(),
        "m".into(),
        Duration::from_secs(5),
    )
    .await
    .expect("probe succeeds")
}

fn probe_response() -> Value {
    json!({ "data": [{ "index": 0, "embedding": [1.0, 0.0] }] })
}

#[test]
fn order_by_index_places_entries_by_index_not_response_order() {
    let data = vec![datum(2, &[2.0]), datum(0, &[0.0]), datum(1, &[1.0])];
    let ordered = order_by_index(data, 3).unwrap();
    assert_eq!(ordered, vec![vec![0.0], vec![1.0], vec![2.0]]);
}

#[test]
fn order_by_index_rejects_count_mismatch() {
    let err = order_by_index(vec![datum(0, &[1.0])], 2).unwrap_err();
    assert!(
        matches!(
            err,
            EmbedError::CountMismatch {
                got: 1,
                expected: 2
            }
        ),
        "{err:?}"
    );
    let err = order_by_index(Vec::new(), 1).unwrap_err();
    assert!(
        matches!(
            err,
            EmbedError::CountMismatch {
                got: 0,
                expected: 1
            }
        ),
        "{err:?}"
    );
}

#[test]
fn order_by_index_rejects_out_of_range_and_repeated_indices() {
    for (label, data) in [
        ("out of range", vec![datum(0, &[1.0]), datum(2, &[1.0])]),
        ("repeated", vec![datum(1, &[1.0]), datum(1, &[1.0])]),
    ] {
        let err = order_by_index(data, 2).unwrap_err();
        assert!(
            matches!(err, EmbedError::BadIndex { .. }),
            "{label}: {err:?}"
        );
    }
}

#[tokio::test]
async fn embed_batch_sends_one_request_and_returns_normalised_vectors_in_input_order() {
    let (addr, server) = serve(vec![
        probe_response(),
        json!({ "data": [
            { "index": 1, "embedding": [0.0, 2.0] },
            { "index": 0, "embedding": [3.0, 4.0] },
        ]}),
    ])
    .await;
    let embedder = embedder_at(addr).await;

    let vectors = embedder.embed_batch(&["first", "second"]).await.unwrap();
    let expected = [[0.6, 0.8], [0.0, 1.0]];
    assert_eq!(vectors.len(), expected.len());
    for (got, want) in vectors.iter().zip(&expected) {
        assert_eq!(got.len(), want.len());
        assert!(
            got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-6),
            "got {vectors:?}, expected {expected:?}"
        );
    }

    let bodies = server.await.unwrap();
    assert_eq!(bodies.len(), 2);
    assert_eq!(bodies[1]["input"], json!(["first", "second"]));
    assert_eq!(bodies[1]["model"], "m");
}

#[tokio::test]
async fn embed_batch_rejects_any_vector_with_the_wrong_dimension() {
    let (addr, server) = serve(vec![
        probe_response(),
        json!({ "data": [
            { "index": 0, "embedding": [1.0, 0.0] },
            { "index": 1, "embedding": [1.0, 0.0, 0.0] },
        ]}),
    ])
    .await;
    let embedder = embedder_at(addr).await;

    let err = embedder.embed_batch(&["a", "b"]).await.unwrap_err();
    assert!(
        matches!(
            err,
            EmbedError::DimMismatch {
                got: 3,
                expected: 2
            }
        ),
        "{err:?}"
    );
    server.await.unwrap();
}

#[test]
fn l2_normalize_scales_to_unit_length_and_passes_degenerate_input_through() {
    let cases: [(&str, Vec<f32>, Vec<f32>); 5] = [
        ("3-4-5", vec![3.0, 4.0], vec![0.6, 0.8]),
        ("already unit", vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]),
        (
            "f32::MAX components do not overflow",
            vec![f32::MAX, f32::MAX],
            vec![std::f32::consts::FRAC_1_SQRT_2; 2],
        ),
        ("zero vector", vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]),
        (
            "non-finite",
            vec![f32::INFINITY, 1.0],
            vec![f32::INFINITY, 1.0],
        ),
    ];
    for (label, input, expected) in cases {
        let got = l2_normalize(input);
        assert_eq!(got.len(), expected.len(), "{label}");
        for (g, e) in got.iter().zip(&expected) {
            assert!(
                g == e || (g - e).abs() < 1e-6,
                "{label}: got {got:?}, expected {expected:?}"
            );
        }
    }
}
