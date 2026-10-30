//! Typed client for the supervised instance's JSON search API.

use std::net::SocketAddr;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use reqwest::Url;
use serde::{Deserialize, Deserializer};

use assistd_utils::child_server::ChildServerStatus;
use assistd_utils::http::{BodyError, local_client_builder, read_capped};
use assistd_utils::readiness::{NotReady, ReadinessCell};

use crate::error::SearchError;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

/// Covers SearXNG's own per-engine timeouts plus result merging.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);

/// One page of results is tens of KiB; anything past this is not a results page.
const BODY_MAX: usize = 512 * 1024;

/// Client for one supervised SearXNG instance on loopback; clones share
/// the connection pool.
#[derive(Debug, Clone)]
pub struct SearxngClient {
    client: reqwest::Client,
    search_url: Url,
    server: Arc<ReadinessCell<ChildServerStatus>>,
}

#[derive(Debug, Clone, Copy)]
pub struct SearchQuery<'a> {
    pub text: &'a str,
    /// A locale such as `en-US`; `None` lets SearXNG pick.
    pub language: Option<&'a str>,
    pub page: NonZeroU32,
}

/// The parts of a results page worth showing; `title` and `content` of each
/// hit have their whitespace runs collapsed to single spaces.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SearchResults {
    #[serde(default)]
    pub results: Vec<SearchHit>,
    /// Instant answers, when an engine or plugin gave one.
    #[serde(default, deserialize_with = "answer_texts")]
    pub answers: Vec<String>,
    #[serde(default)]
    pub suggestions: Vec<String>,
    /// `(engine, reason)` for each engine that failed to answer.
    #[serde(default)]
    pub unresponsive_engines: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SearchHit {
    pub url: String,
    #[serde(default, deserialize_with = "collapsed_whitespace")]
    pub title: String,
    #[serde(default, deserialize_with = "collapsed_whitespace")]
    pub content: String,
    #[serde(default)]
    pub engines: Vec<String>,
    /// Publication time as the engine reported it, when it did.
    #[serde(default, rename = "publishedDate")]
    pub published: Option<String>,
}

/// SearXNG releases differ on whether an answer is a bare string or an object.
#[derive(Deserialize)]
#[serde(untagged)]
enum WireAnswer {
    Text(String),
    Object { answer: String },
}

impl SearxngClient {
    /// A client for the instance listening on `addr`, gated on the
    /// readiness of the server `server` tracks.
    pub fn new(
        addr: SocketAddr,
        server: Arc<ReadinessCell<ChildServerStatus>>,
    ) -> Result<Self, SearchError> {
        let client = local_client_builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(SearchError::Client)?;
        let search_url = Url::parse(&format!("http://{addr}/search"))
            .expect("a socket address forms a valid http URL");
        Ok(Self {
            client,
            search_url,
            server,
        })
    }

    /// Fails without sending with [`SearchError::NotServing`] while the
    /// instance is starting or not serving, and [`SearchError::Unavailable`]
    /// when it never started.
    pub async fn search(&self, query: &SearchQuery<'_>) -> Result<SearchResults, SearchError> {
        self.ensure_serving()?;
        let response = self
            .client
            .get(self.request_url(query))
            .send()
            .await
            .map_err(SearchError::Transport)?;
        let status = response.status();
        if !status.is_success() {
            return Err(SearchError::Status(status));
        }
        let body = read_capped(response, BODY_MAX).await.map_err(|e| match e {
            BodyError::TooLarge { max } => SearchError::TooLarge { max },
            BodyError::Read(source) => SearchError::Transport(source),
        })?;
        serde_json::from_slice(&body).map_err(SearchError::Decode)
    }

    fn ensure_serving(&self) -> Result<(), SearchError> {
        match self.server.get() {
            Ok(status) if status.is_serving() => Ok(()),
            Ok(_) | Err(NotReady::Starting) => Err(SearchError::NotServing),
            Err(NotReady::Unavailable(reason)) => Err(SearchError::Unavailable(reason)),
        }
    }

    fn request_url(&self, query: &SearchQuery<'_>) -> Url {
        let mut url = self.search_url.clone();
        {
            let mut pairs = url.query_pairs_mut();
            pairs
                .append_pair("q", query.text)
                .append_pair("format", "json")
                .append_pair("pageno", &query.page.to_string());
            if let Some(language) = query.language {
                pairs.append_pair("language", language);
            }
        }
        url
    }
}

fn collapsed_whitespace<'de, D: Deserializer<'de>>(deserializer: D) -> Result<String, D::Error> {
    let text = Option::<String>::deserialize(deserializer)?.unwrap_or_default();
    Ok(text.split_whitespace().collect::<Vec<_>>().join(" "))
}

fn answer_texts<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<String>, D::Error> {
    let answers = Vec::<WireAnswer>::deserialize(deserializer)?;
    Ok(answers
        .into_iter()
        .map(|answer| match answer {
            WireAnswer::Text(text) | WireAnswer::Object { answer: text } => text,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_a_captured_results_page() {
        let page: SearchResults =
            serde_json::from_str(include_str!("../fixtures/search_response.json"))
                .expect("fixture decodes");
        assert_eq!(page.results.len(), 3);
        assert_eq!(
            page.results[1],
            SearchHit {
                url: "https://md5decrypt.net/en/Sha256/".into(),
                title: "SHA-256 generator, verifier and hash lookup - Md5 Encrypt & Decrypt".into(),
                content: "The SHA-256 digest of hello is \
                    2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824 . \
                    A single changed bit, capital letter or line ending produces a ..."
                    .into(),
                engines: vec!["google cse".into()],
                published: None,
            }
        );
        assert_eq!(
            page.answers,
            ["sha256 hash digest: \
                2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"]
        );
        assert_eq!(
            page.unresponsive_engines,
            [
                ("brave".to_owned(), "too many requests".to_owned()),
                ("duckduckgo".to_owned(), "CAPTCHA".to_owned()),
            ]
        );
    }
}
