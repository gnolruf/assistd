//! HTTP client for llama-server's `/v1/embeddings` endpoint.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::{EmbedError, Embedder};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Serialize)]
struct EmbedRequest<'a> {
    input: &'a str,
    model: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedDatum>,
}

#[derive(Deserialize)]
struct EmbedDatum {
    embedding: Vec<f32>,
}

/// [`Embedder`] backed by a llama-server running with `--embedding`.
pub struct LlamaEmbedder {
    client: reqwest::Client,
    base_url: String,
    model: String,
    dim: usize,
}

impl LlamaEmbedder {
    /// Probe the server once to learn the vector dimension.
    /// `request_timeout` applies to the probe and every `embed` call.
    pub async fn new(
        host: &str,
        port: u16,
        model: String,
        request_timeout: Duration,
    ) -> Result<Self, EmbedError> {
        let client = reqwest::Client::builder()
            .no_proxy()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(request_timeout)
            .build()
            .map_err(EmbedError::Client)?;
        let base_url = format!("http://{host}:{port}");

        let probe = embed_raw(&client, &base_url, &model, "x").await?;
        let dim = probe.len();
        if dim == 0 {
            return Err(EmbedError::DimProbeEmpty);
        }
        tracing::info!(
            target: "assistd::embed",
            model = %model,
            dim,
            "embedder ready"
        );

        Ok(Self {
            client,
            base_url,
            model,
            dim,
        })
    }
}

#[async_trait]
impl Embedder for LlamaEmbedder {
    async fn embed(&self, text: String) -> Result<Vec<f32>, EmbedError> {
        let raw = embed_raw(&self.client, &self.base_url, &self.model, &text).await?;
        if raw.len() != self.dim {
            return Err(EmbedError::DimMismatch {
                got: raw.len(),
                expected: self.dim,
            });
        }
        Ok(l2_normalize(raw))
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

async fn embed_raw(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    text: &str,
) -> Result<Vec<f32>, EmbedError> {
    let url = format!("{base_url}/v1/embeddings");
    let body = EmbedRequest { input: text, model };
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|source| EmbedError::Request { url, source })?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(EmbedError::Status {
            status,
            body: body.chars().take(200).collect(),
        });
    }
    let parsed: EmbedResponse = resp.json().await.map_err(EmbedError::Decode)?;
    let first = parsed.data.into_iter().next().ok_or(EmbedError::NoData)?;
    Ok(first.embedding)
}

fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return v;
    }
    let inv = (1.0 / norm) as f32;
    for x in &mut v {
        *x *= inv;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_normalize_unit_vector() {
        let v = l2_normalize(vec![3.0, 4.0]);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "expected unit length, got {norm}"
        );
        assert!((v[0] - 0.6).abs() < 1e-5);
        assert!((v[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn l2_normalize_zero_vector_is_identity() {
        let v = l2_normalize(vec![0.0, 0.0, 0.0]);
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn l2_normalize_already_unit() {
        let v = l2_normalize(vec![1.0, 0.0, 0.0]);
        assert_eq!(v, vec![1.0, 0.0, 0.0]);
    }
}
