use std::sync::Mutex;

use super::*;

/// Fails any call that includes the text `"bad"`; records the inputs of every call.
#[derive(Default)]
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
            return Err(EmbedError::Disabled);
        }
        Ok(texts.iter().map(|t| vec![t.len() as f32]).collect())
    }
    fn model(&self) -> &str {
        "picky"
    }
    fn dim(&self) -> usize {
        1
    }
}

#[tokio::test]
async fn no_embedder_errors_on_embed() {
    let embedder = NoEmbedder;
    let err = embedder
        .embed("hi".into())
        .await
        .expect_err("NoEmbedder must not embed");
    assert!(matches!(err, EmbedError::Disabled), "{err:?}");
    assert_eq!(embedder.model(), "");
    assert_eq!(embedder.dim(), 0);
}

#[tokio::test]
async fn embed_each_uses_one_batch_call_when_it_succeeds() {
    let embedder = PickyEmbedder::default();
    let results = embed_each(&embedder, &["a", "bb"]).await;
    let vectors: Vec<_> = results.into_iter().map(Result::unwrap).collect();
    assert_eq!(vectors, vec![vec![1.0], vec![2.0]]);
    assert_eq!(*embedder.calls.lock().unwrap(), vec![vec!["a", "bb"]]);
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
