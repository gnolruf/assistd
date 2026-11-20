//! HTTP client for llama-server's `/v1/embeddings` endpoint.

use std::borrow::Cow;
use std::net::SocketAddr;
use std::time::Duration;

use assistd_utils::child_server::{ApiKey, ChildServerStatus};
use assistd_utils::http::{local_client_builder, read_capped};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{EmbedError, Embedder};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
/// Largest accepted `/v1/embeddings` body: a full batch of 8192-dim vectors in JSON is under 8 MiB.
const BODY_MAX: usize = 32 * 1024 * 1024;
const ERROR_BODY_MAX: usize = 64 * 1024;
const ERROR_BODY_CHARS: usize = 200;
/// What llama-server's error says of an input over its batch size. It reports that
/// refusal as a server error, which only these words tell from a failure of its own.
const INPUT_TOO_LARGE: &str = "is too large to process";

#[derive(Serialize)]
struct EmbedRequest<'a> {
    input: &'a [Cow<'a, str>],
    model: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedDatum>,
}

#[derive(Deserialize)]
struct EmbedDatum {
    index: usize,
    embedding: Vec<f32>,
}

/// The text a model is trained to see before each input, by what the input is for.
/// An empty prefix sends that input as it is.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TaskPrefixes {
    /// Goes before text that is stored to be searched.
    pub document: String,
    /// Goes before a query that searches the stored text.
    pub query: String,
}

/// [`Embedder`] backed by a llama-server running with `--embedding`.
#[derive(Debug)]
pub struct LlamaEmbedder {
    client: reqwest::Client,
    base_url: String,
    model: String,
    prefixes: TaskPrefixes,
    dim: usize,
    server: Option<ChildServerStatus>,
}

impl LlamaEmbedder {
    /// Probe the server once for the vector dimension. `request_timeout` bounds the probe and
    /// every later request, which carry `api_key` when set. With `server`, a request made while
    /// it is not serving fails with [`EmbedError::NotReady`] without being sent.
    pub async fn new(
        addr: SocketAddr,
        model: String,
        prefixes: TaskPrefixes,
        request_timeout: Duration,
        api_key: Option<&ApiKey>,
        server: Option<ChildServerStatus>,
    ) -> Result<Self, EmbedError> {
        let client = local_client_builder()
            .default_headers(api_key.map_or_default(ApiKey::authorization_headers))
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(request_timeout)
            .build()
            .map_err(EmbedError::Client)?;
        let mut embedder = Self {
            client,
            base_url: format!("http://{addr}"),
            model,
            prefixes,
            dim: 0,
            server,
        };
        embedder.dim = embedder
            .post_embeddings(&[Cow::Borrowed("x")])
            .await?
            .first()
            .map_or(0, Vec::len);
        if embedder.dim == 0 {
            return Err(EmbedError::DimProbeEmpty);
        }
        tracing::info!(
            target: "assistd::embed",
            model = %embedder.model,
            dim = embedder.dim,
            "embedder ready"
        );
        Ok(embedder)
    }

    fn require_serving(&self) -> Result<(), EmbedError> {
        match &self.server {
            Some(server) if !server.is_serving() => {
                tracing::warn!(
                    target: "assistd::embed",
                    "embed server is not ready; request not sent"
                );
                Err(EmbedError::NotReady)
            }
            _ => Ok(()),
        }
    }

    async fn post_embeddings(&self, input: &[Cow<'_, str>]) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.require_serving()?;
        let url = format!("{}/v1/embeddings", self.base_url);
        let body = EmbedRequest {
            input,
            model: &self.model,
        };
        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|source| EmbedError::Request { url, source })?;
        let status = response.status();
        if !status.is_success() {
            let body = read_capped(response, ERROR_BODY_MAX)
                .await
                .unwrap_or_default();
            return Err(refusal(status, &String::from_utf8_lossy(&body)));
        }
        let body = read_capped(response, BODY_MAX).await?;
        let parsed: EmbedResponse = serde_json::from_slice(&body).map_err(EmbedError::Decode)?;
        order_by_index(parsed.data, input.len())
    }

    async fn embed_prefixed(
        &self,
        prefix: &str,
        texts: &[&str],
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let input: Vec<Cow<'_, str>> = texts.iter().map(|text| prefixed(prefix, text)).collect();
        self.post_embeddings(&input)
            .await?
            .into_iter()
            .map(|raw| {
                if raw.len() != self.dim {
                    Err(EmbedError::DimMismatch {
                        got: raw.len(),
                        expected: self.dim,
                    })
                } else if !raw.iter().all(|value| value.is_finite()) {
                    Err(EmbedError::NonFinite)
                } else {
                    Ok(l2_normalize(raw))
                }
            })
            .collect()
    }
}

#[async_trait]
impl Embedder for LlamaEmbedder {
    async fn embed_query(&self, query: &str) -> Result<Vec<f32>, EmbedError> {
        self.embed_prefixed(&self.prefixes.query, &[query])
            .await?
            .pop()
            .ok_or(EmbedError::CountMismatch {
                got: 0,
                expected: 1,
            })
    }

    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.embed_prefixed(&self.prefixes.document, texts).await
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// The error for a response of `status` that is not a success, given its `body`.
fn refusal(status: reqwest::StatusCode, body: &str) -> EmbedError {
    let body: String = body.chars().take(ERROR_BODY_CHARS).collect();
    if body.contains(INPUT_TOO_LARGE) {
        EmbedError::InputTooLarge { body }
    } else {
        EmbedError::Status { status, body }
    }
}

fn prefixed<'a>(prefix: &str, text: &'a str) -> Cow<'a, str> {
    if prefix.is_empty() {
        Cow::Borrowed(text)
    } else {
        Cow::Owned(format!("{prefix}{text}"))
    }
}

/// Place each entry at its `index`, since the server may return entries in any order.
fn order_by_index(data: Vec<EmbedDatum>, expected: usize) -> Result<Vec<Vec<f32>>, EmbedError> {
    if data.len() != expected {
        return Err(EmbedError::CountMismatch {
            got: data.len(),
            expected,
        });
    }
    let mut slots: Vec<Option<Vec<f32>>> = vec![None; expected];
    for EmbedDatum { index, embedding } in data {
        match slots.get_mut(index) {
            Some(slot) if slot.is_none() => *slot = Some(embedding),
            _ => return Err(EmbedError::BadIndex { index, expected }),
        }
    }
    Ok(slots.into_iter().flatten().collect())
}

/// Scale finite `vector` to unit length, leaving a zero vector as it is.
fn l2_normalize(mut vector: Vec<f32>) -> Vec<f32> {
    let largest = vector
        .iter()
        .fold(0.0_f32, |largest, value| largest.max(value.abs()));
    if largest == 0.0 {
        return vector;
    }
    let norm_over_largest = vector
        .iter()
        .map(|value| (value / largest).powi(2))
        .sum::<f32>()
        .sqrt();
    for value in &mut vector {
        *value = *value / largest / norm_over_largest;
    }
    vector
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use assistd_utils::child_server::ReadyState;
    use assistd_utils::child_server::scripted::scripted_status;
    use serde_json::{Value, json};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};
    use tokio::task::JoinHandle;
    use tokio::time::timeout;

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
            .map_or(0, |v| v.trim().parse().unwrap());
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

    fn prefixes() -> TaskPrefixes {
        TaskPrefixes {
            document: "doc: ".into(),
            query: "query: ".into(),
        }
    }

    async fn embedder_at(addr: SocketAddr) -> LlamaEmbedder {
        LlamaEmbedder::new(
            addr,
            "m".into(),
            prefixes(),
            Duration::from_secs(5),
            None,
            None,
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
    async fn embed_documents_sends_one_prefixed_request_and_returns_normalised_vectors_in_input_order()
     {
        let (addr, server) = serve(vec![
            probe_response(),
            json!({ "data": [
                { "index": 1, "embedding": [0.0, 2.0] },
                { "index": 0, "embedding": [3.0, 4.0] },
            ]}),
        ])
        .await;
        let embedder = embedder_at(addr).await;

        let vectors = embedder
            .embed_documents(&["first", "second"])
            .await
            .unwrap();
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
        assert_eq!(bodies[0]["input"], json!(["x"]));
        assert_eq!(bodies[1]["input"], json!(["doc: first", "doc: second"]));
        assert_eq!(bodies[1]["model"], "m");
    }

    #[tokio::test]
    async fn embed_documents_rejects_any_vector_with_the_wrong_dimension() {
        let (addr, server) = serve(vec![
            probe_response(),
            json!({ "data": [
                { "index": 0, "embedding": [1.0, 0.0] },
                { "index": 1, "embedding": [1.0, 0.0, 0.0] },
            ]}),
        ])
        .await;
        let embedder = embedder_at(addr).await;

        let err = embedder.embed_documents(&["a", "b"]).await.unwrap_err();
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
    fn an_input_over_the_batch_size_is_the_inputs_fault_and_other_server_errors_are_not() {
        let server_error = reqwest::StatusCode::INTERNAL_SERVER_ERROR;
        let too_large = r#"{"error":{"code":500,"message":"input (600 tokens) is too large to process. increase the physical batch size (current batch size: 512)","type":"server_error"}}"#;
        let failed = r#"{"error":{"code":500,"message":"failed to decode the batch","type":"server_error"}}"#;

        let refused = refusal(server_error, too_large);
        assert!(
            matches!(refused, EmbedError::InputTooLarge { .. }) && refused.is_input_caused(),
            "{refused:?}"
        );
        let broken = refusal(server_error, failed);
        assert!(
            matches!(broken, EmbedError::Status { .. }) && !broken.is_input_caused(),
            "{broken:?}"
        );
    }

    #[test]
    fn l2_normalize_scales_to_unit_length_and_passes_degenerate_input_through() {
        let cases: [(&str, Vec<f32>, Vec<f32>); 3] = [
            ("3-4-5", vec![3.0, 4.0], vec![0.6, 0.8]),
            (
                "f32::MAX components do not overflow",
                vec![f32::MAX, f32::MAX],
                vec![std::f32::consts::FRAC_1_SQRT_2; 2],
            ),
            ("zero vector", vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]),
        ];
        for (label, input, expected) in cases {
            let got = l2_normalize(input);
            assert_eq!(got.len(), expected.len(), "{label}");
            for (g, e) in got.iter().zip(&expected) {
                assert!(
                    g.to_bits() == e.to_bits() || (g - e).abs() < 1e-6,
                    "{label}: got {got:?}, expected {expected:?}"
                );
            }
        }
    }

    #[tokio::test]
    async fn nothing_is_sent_while_the_embed_server_is_not_serving() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let cases = [
            ("starting", ReadyState::Starting, Some(7), true),
            (
                "backing off",
                ReadyState::BackingOff { attempt: 1 },
                None,
                true,
            ),
            ("degraded", ReadyState::Degraded, None, true),
            ("ready with its child gone", ReadyState::Ready, None, true),
            ("supervisor stopped", ReadyState::Ready, Some(7), false),
        ];
        for (label, state, pid, supervisor_running) in cases {
            let (ready_tx, status) = scripted_status(state, pid);
            let _supervisor = supervisor_running.then_some(ready_tx);
            let result = LlamaEmbedder::new(
                addr,
                "m".into(),
                prefixes(),
                Duration::from_secs(5),
                None,
                Some(status),
            )
            .await;
            assert!(
                matches!(result, Err(EmbedError::NotReady)),
                "{label}: {result:?}"
            );
        }
        assert!(
            timeout(Duration::from_millis(100), listener.accept())
                .await
                .is_err(),
            "a request reached the listener"
        );
    }

    #[tokio::test]
    async fn embeds_are_refused_while_the_server_restarts_and_resume_once_it_serves() {
        let (addr, server) = serve(vec![
            probe_response(),
            json!({ "data": [{ "index": 0, "embedding": [0.0, 1.0] }] }),
        ])
        .await;
        let (ready_tx, status) = scripted_status(ReadyState::Ready, Some(7));
        let embedder = LlamaEmbedder::new(
            addr,
            "m".into(),
            prefixes(),
            Duration::from_secs(5),
            None,
            Some(status),
        )
        .await
        .expect("probe succeeds while serving");

        ready_tx.send_replace(ReadyState::BackingOff { attempt: 1 });
        let refused = embedder.embed_query("private memory").await;
        assert!(matches!(refused, Err(EmbedError::NotReady)), "{refused:?}");

        ready_tx.send_replace(ReadyState::Ready);
        let vector = embedder.embed_query("after restart").await.unwrap();
        assert_eq!(vector, vec![0.0, 1.0]);

        let bodies = server.await.unwrap();
        assert_eq!(bodies.len(), 2);
        assert_eq!(bodies[1]["input"], json!(["query: after restart"]));
    }
}
