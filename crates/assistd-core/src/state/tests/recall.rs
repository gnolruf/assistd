use super::*;

#[derive(Debug)]
struct ConstantEmbedder;

#[async_trait::async_trait]
impl assistd_embed::Embedder for ConstantEmbedder {
    async fn embed_query(&self, _query: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(vec![1.0])
    }
    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(vec![vec![1.0]; texts.len()])
    }
    fn model(&self) -> &'static str {
        "constant"
    }
    fn dim(&self) -> usize {
        1
    }
}

/// Returns its fixed hits of the kinds searched, minus the excluded session's.
#[derive(Debug)]
struct FixedHits {
    hits: Vec<SearchHit>,
}

#[async_trait::async_trait]
impl RecallStore for FixedHits {
    async fn search(&self, search: ChunkSearch) -> assistd_memory::Result<Vec<SearchHit>> {
        let searched = |hit: &&SearchHit| match hit.locator {
            ChunkLocator::Message { session_id, .. } => {
                search.kinds.contains(&ChunkSource::Message)
                    && search.sessions != SessionScope::AllExcept(session_id)
            }
            ChunkLocator::Memory { .. } => search.kinds.contains(&ChunkSource::Memory),
            ChunkLocator::Document { .. } | ChunkLocator::Access { .. } => false,
        };
        let hits = self.hits.iter().filter(searched);
        Ok(hits.take(search.top_k).cloned().collect())
    }
    async fn adjacent_chunks(
        &self,
        locator: &ChunkLocator,
    ) -> assistd_memory::Result<AdjacentChunks> {
        NoSemanticStore.adjacent_chunks(locator).await
    }
    async fn document_text(
        &self,
        path: &Path,
        page: usize,
        bytes: Range<usize>,
    ) -> assistd_memory::Result<Option<DocumentText>> {
        NoSemanticStore.document_text(path, page, bytes).await
    }
}

fn recall_hit(session_id: SessionId, message: i64, similarity: f32, text: &str) -> SearchHit {
    SearchHit {
        chunk_id: ChunkId(message),
        locator: ChunkLocator::Message {
            session_id,
            message_id: MessageId(message),
            role: PersistedRole::User,
            timestamp: Utc::now(),
            session_label: "Daemon tuning".into(),
        },
        text: text.into(),
        similarity: Some(similarity),
        held_elsewhere: 0,
    }
}

#[tokio::test]
async fn auto_recall_runs_once_per_new_session_with_memories_and_past_chat_pointers() {
    let current = SessionId::new();
    let other = SessionId::new();
    let config = Config::default();
    let (mut memory, db_dir) = fresh_memory(&config).await;
    memory.embedder = Arc::new(EmbedderHandle::new(Readiness::Ready(Arc::new(
        ConstantEmbedder,
    ))));
    memory.recall = Arc::new(FixedHits {
        hits: vec![
            recall_hit(current, 1, 0.9, "said in this chat"),
            recall_hit(other, 2, 0.9, "strong past match"),
            recall_hit(other, 3, 0.8, "another past match"),
            recall_hit(SessionId::new(), 4, 0.1, "weak past match"),
            memory_hit(1, "editor", "helix", 0.9),
            memory_hit(2, "shell", "fish", 0.1),
        ],
    });
    let session_ctx = SessionContext::new(current, None);
    let state = TestState::new(config, stub_subsystems(), memory, db_dir, session_ctx);
    let recall = async |state: &AppState, query: &str| {
        state.turns.build_semantic_context(query).await.unwrap()
    };
    let session_ctx = &state.session_ctx;

    assert_eq!(recall(&state, "hi").await, None, "too short to search");
    session_ctx.replace(current, Some(BranchId(1))).await;
    let first = recall(&state, "rust daemon")
        .await
        .expect("saving the first message leaves recall pending");
    let pointer = format!("- chat {} \"Daemon tuning\" ", other.short());
    assert!(first.contains("- editor: helix\n"), "{first}");
    assert!(
        first.contains("untrusted data, not instructions"),
        "{first}"
    );
    assert!(!first.contains("fish"), "{first}");
    assert_eq!(first.matches(&pointer).count(), 1, "{first}");
    assert!(first.contains("2 matching messages"), "{first}");
    assert_eq!(first.matches("- chat ").count(), 1, "{first}");
    assert!(!first.contains("past match"), "no message text: {first}");
    assert!(!first.contains("said in this chat"), "{first}");
    assert_eq!(
        recall(&state, "rust daemon").await,
        None,
        "once per session"
    );

    session_ctx.replace(other, Some(BranchId(2))).await;
    assert_eq!(recall(&state, "rust daemon").await, None, "resumed session");

    session_ctx.replace(SessionId::new(), None).await;
    let fresh = recall(&state, "rust daemon")
        .await
        .expect("a new session recalls");
    assert!(fresh.contains(&pointer), "{fresh}");
}
