//! HTTP client for llama-server's `/v1/embeddings` endpoint.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::{EmbedError, Embedder};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Serialize)]
struct EmbedRequest<'a> {
    input: &'a [&'a str],
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

/// [`Embedder`] backed by a llama-server running with `--embedding`.
pub struct LlamaEmbedder {
    client: reqwest::Client,
    base_url: String,
    model: String,
    dim: usize,
}

impl LlamaEmbedder {
    /// Probe the server once to learn the vector dimension.
    /// `request_timeout` applies to the probe and every embed request.
    /// Errors if the probe fails or returns an empty vector.
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

        let dim = embed_raw(&client, &base_url, &model, &["x"])
            .await?
            .first()
            .map_or(0, Vec::len);
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
        self.embed_batch(&[text.as_str()])
            .await?
            .pop()
            .ok_or(EmbedError::CountMismatch {
                got: 0,
                expected: 1,
            })
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        embed_raw(&self.client, &self.base_url, &self.model, texts)
            .await?
            .into_iter()
            .map(|raw| {
                if raw.len() == self.dim {
                    Ok(l2_normalize(raw))
                } else {
                    Err(EmbedError::DimMismatch {
                        got: raw.len(),
                        expected: self.dim,
                    })
                }
            })
            .collect()
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
    input: &[&str],
) -> Result<Vec<Vec<f32>>, EmbedError> {
    let url = format!("{base_url}/v1/embeddings");
    let body = EmbedRequest { input, model };
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
    order_by_index(parsed.data, input.len())
}

/// Place each entry at its `index`; the server is free to return
/// entries in any order.
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
mod tests;
