//! HTTP client for llama.cpp's `/v1/embeddings` endpoint.
//!
//! Holds a `reqwest::Client` (no proxy, short connect timeout, configurable
//! per-request timeout). Vector dimensionality is **probed once at
//! construction** by sending a one-token embed request; embedders don't
//! advertise their dim out-of-band, and exposing it on the trait lets
//! callers size their `Vec<f32>` buffers without a downstream round-trip.
//!
//! L2 normalisation happens here, before the vector leaves the crate, so
//! every consumer (chunk indexing, query injection, `recall`/`reminisce`)
//! can compute cosine as a plain dot product against stored vectors that
//! were normalised the same way.

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::Embedder;

/// Connect timeout for the embed HTTP client. Should be loopback-fast;
/// 2s is generous enough to absorb a busy event loop without stalling
/// queries.
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

/// HTTP-driven implementation of [`Embedder`]. Connects to a llama-server
/// instance running with `--embedding`. Returns L2-normalised vectors.
pub struct LlamaEmbedder {
    client: reqwest::Client,
    base_url: String,
    model: String,
    dim: usize,
}

impl LlamaEmbedder {
    /// Probe the server, latch the dim, and return a ready-to-use client.
    ///
    /// `request_timeout` applies per `embed()` call. The probe itself
    /// uses the same budget; first-load may take longer, but
    /// `EmbedService::start` already waited for `/health` so the model
    /// is loaded by the time we get here.
    pub async fn new(
        host: &str,
        port: u16,
        model: String,
        request_timeout: Duration,
    ) -> Result<Self> {
        let client = reqwest::Client::builder()
            .no_proxy()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(request_timeout)
            .build()
            .context("build embed reqwest client")?;
        let base_url = format!("http://{host}:{port}");

        // Probe dim with a one-token request. Any failure here is fatal;
        // the daemon's try-warn-fallback path will swap in NoEmbedder.
        let probe = embed_raw(&client, &base_url, &model, "x").await?;
        let dim = probe.len();
        if dim == 0 {
            return Err(anyhow!(
                "embed server returned an empty vector during dim probe"
            ));
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
    async fn embed(&self, text: String) -> Result<Vec<f32>> {
        let raw = embed_raw(&self.client, &self.base_url, &self.model, &text).await?;
        if raw.len() != self.dim {
            return Err(anyhow!(
                "embed server returned dim {} but probe latched {}; refusing to mix",
                raw.len(),
                self.dim
            ));
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
) -> Result<Vec<f32>> {
    let url = format!("{base_url}/v1/embeddings");
    let body = EmbedRequest { input: text, model };
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .with_context(|| format!("POST {url}"))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(anyhow!(
            "embed server returned {status} (body: {})",
            body.chars().take(200).collect::<String>()
        ));
    }
    let parsed: EmbedResponse = resp
        .json()
        .await
        .context("parse /v1/embeddings response body")?;
    let first = parsed
        .data
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("embed response had no data entries"))?;
    Ok(first.embedding)
}

fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let mut sum_sq = 0.0f64;
    for &x in &v {
        sum_sq += (x as f64) * (x as f64);
    }
    let norm = sum_sq.sqrt();
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
