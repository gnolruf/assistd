use std::sync::Mutex;

use super::*;

/// Fails any call that includes `"bad"` as input-caused, any that includes `"down"` with a
/// server error, and refuses any that includes `"offline"` as not ready; records every call.
#[derive(Debug, Default)]
struct PickyEmbedder {
    calls: Mutex<Vec<Vec<String>>>,
}

#[async_trait]
impl Embedder for PickyEmbedder {
    async fn embed(&self, text: String) -> Result<Vec<f32>, EmbedError> {
        self.embed_batch(&[text.as_str()])
            .await?
            .pop()
            .ok_or(EmbedError::Disabled)
    }
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.calls
            .lock()
            .unwrap()
            .push(texts.iter().map(|&t| t.to_owned()).collect());
        if texts.contains(&"bad") {
            return Err(EmbedError::DimMismatch {
                got: 2,
                expected: 1,
            });
        }
        if texts.contains(&"down") {
            return Err(EmbedError::Status {
                status: reqwest::StatusCode::SERVICE_UNAVAILABLE,
                body: String::new(),
            });
        }
        if texts.contains(&"offline") {
            return Err(EmbedError::NotReady);
        }
        Ok(texts.iter().map(|t| vec![t.len() as f32]).collect())
    }
    fn model(&self) -> &'static str {
        "picky"
    }
    fn dim(&self) -> usize {
        1
    }
}

#[tokio::test]
async fn embed_each_falls_back_to_individual_embeds_on_batch_failure() {
    let embedder = PickyEmbedder::default();
    let results = embed_each(&embedder, &["a", "bad", "ccc"]).await;
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].as_ref().unwrap(), &vec![1.0]);
    assert!(results[1].is_err());
    assert_eq!(results[2].as_ref().unwrap(), &vec![3.0]);
    assert_eq!(
        *embedder.calls.lock().unwrap(),
        vec![vec!["a", "bad", "ccc"], vec!["a"], vec!["bad"], vec!["ccc"]]
    );
}

#[tokio::test]
async fn embed_each_fails_every_input_without_retrying_when_the_server_is_not_ready() {
    let embedder = PickyEmbedder::default();
    let results = embed_each(&embedder, &["a", "offline", "ccc"]).await;
    assert_eq!(results.len(), 3);
    assert!(
        results
            .iter()
            .all(|result| matches!(result, Err(EmbedError::NotReady))),
        "{results:?}"
    );
    assert_eq!(
        *embedder.calls.lock().unwrap(),
        vec![vec!["a", "offline", "ccc"]]
    );
}

#[tokio::test]
async fn embed_each_fails_every_input_without_retrying_on_a_server_error() {
    let embedder = PickyEmbedder::default();
    let results = embed_each(&embedder, &["a", "down", "ccc"]).await;
    assert_eq!(results.len(), 3);
    assert!(
        results
            .iter()
            .all(|result| matches!(result, Err(EmbedError::BatchFailed(_)))),
        "{results:?}"
    );
    assert_eq!(
        *embedder.calls.lock().unwrap(),
        vec![vec!["a", "down", "ccc"]]
    );
}
