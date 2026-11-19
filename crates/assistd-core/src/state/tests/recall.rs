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

/// Returns its fixed hits, minus the excluded session's.
#[derive(Debug)]
struct FixedHits {
    chunks: Vec<EmbeddingHit>,
    memories: Vec<MemoryHit>,
}

#[async_trait::async_trait]
impl SemanticStore for FixedHits {
    async fn nearest_chunks(
        &self,
        _query_vector: Vec<f32>,
        top_k: usize,
        scope: ChunkScope,
    ) -> assistd_memory::Result<Vec<EmbeddingHit>> {
        Ok(self
            .chunks
            .iter()
            .filter(|hit| match hit.locator {
                ChunkLocator::Message { session_id, .. } => {
                    scope != ChunkScope::Messages(SessionScope::AllExcept(session_id))
                }
                ChunkLocator::Document { .. } => scope == ChunkScope::Documents,
            })
            .take(top_k)
            .cloned()
            .collect())
    }
    async fn nearest_memories(
        &self,
        _query_vector: Vec<f32>,
        top_k: usize,
    ) -> assistd_memory::Result<Vec<MemoryHit>> {
        Ok(self.memories.iter().take(top_k).cloned().collect())
    }
    async fn adjacent_chunks(
        &self,
        locator: &ChunkLocator,
    ) -> assistd_memory::Result<AdjacentChunks> {
        NoSemanticStore.adjacent_chunks(locator).await
    }
    async fn adopt_embedding_space(
        &self,
        space: &EmbeddingSpace,
    ) -> assistd_memory::Result<Option<EmbeddingSpace>> {
        NoSemanticStore.adopt_embedding_space(space).await
    }
    async fn count_missing(&self) -> assistd_memory::Result<UnindexedRows> {
        NoSemanticStore.count_missing().await
    }
    async fn memories_missing_embedding(
        &self,
        after: Option<MemoryId>,
        limit: usize,
    ) -> assistd_memory::Result<Vec<(MemoryId, String)>> {
        NoSemanticStore
            .memories_missing_embedding(after, limit)
            .await
    }
    async fn chunks_missing_embedding(
        &self,
        source: ChunkSource,
        after: Option<ChunkId>,
        limit: usize,
    ) -> assistd_memory::Result<Vec<(ChunkId, String)>> {
        NoSemanticStore
            .chunks_missing_embedding(source, after, limit)
            .await
    }
    async fn store_embeddings(&self, embeddings: Vec<Embedding>) -> assistd_memory::Result<()> {
        NoSemanticStore.store_embeddings(embeddings).await
    }
    async fn clear_unembeddable(&self) -> assistd_memory::Result<u64> {
        NoSemanticStore.clear_unembeddable().await
    }
    async fn begin_index_pass(&self, root: &Path) -> assistd_memory::Result<IndexPass> {
        NoSemanticStore.begin_index_pass(root).await
    }
    async fn changed_documents(
        &self,
        pass: IndexPass,
        files: Vec<(PathBuf, FileStamp)>,
    ) -> assistd_memory::Result<Vec<(PathBuf, FileStamp)>> {
        NoSemanticStore.changed_documents(pass, files).await
    }
    async fn index_document(
        &self,
        pass: IndexPass,
        path: PathBuf,
        stamp: FileStamp,
        pages: Vec<String>,
    ) -> assistd_memory::Result<Vec<ChunkId>> {
        NoSemanticStore
            .index_document(pass, path, stamp, pages)
            .await
    }
    async fn finish_index_pass(&self, pass: IndexPass) -> assistd_memory::Result<u64> {
        NoSemanticStore.finish_index_pass(pass).await
    }
    async fn finish_index_pass_under(
        &self,
        pass: IndexPass,
        paths: Vec<PathBuf>,
    ) -> assistd_memory::Result<u64> {
        NoSemanticStore.finish_index_pass_under(pass, paths).await
    }
    async fn remove_document(&self, path: &Path) -> assistd_memory::Result<bool> {
        NoSemanticStore.remove_document(path).await
    }
    async fn index_roots(&self) -> assistd_memory::Result<Vec<PathBuf>> {
        NoSemanticStore.index_roots().await
    }
    async fn index_root_counts(&self) -> assistd_memory::Result<Vec<IndexRootCounts>> {
        NoSemanticStore.index_root_counts().await
    }
    async fn remove_index_root(&self, root: &Path) -> assistd_memory::Result<u64> {
        NoSemanticStore.remove_index_root(root).await
    }
    async fn count_document_chunks(&self) -> assistd_memory::Result<u64> {
        NoSemanticStore.count_document_chunks().await
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

fn recall_hit(session_id: SessionId, message: i64, similarity: f32, text: &str) -> EmbeddingHit {
    EmbeddingHit {
        chunk_id: ChunkId(message),
        locator: ChunkLocator::Message {
            session_id,
            message_id: MessageId(message),
            role: PersistedRole::User,
            timestamp: Utc::now(),
            session_label: "Daemon tuning".into(),
        },
        text: text.into(),
        similarity,
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
    memory.semantic = Arc::new(FixedHits {
        chunks: vec![
            recall_hit(current, 1, 0.9, "said in this chat"),
            recall_hit(other, 2, 0.9, "strong past match"),
            recall_hit(other, 3, 0.8, "another past match"),
            recall_hit(SessionId::new(), 4, 0.1, "weak past match"),
        ],
        memories: vec![
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
