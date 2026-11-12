use super::*;
use crate::sqlite::{SqliteMemoryStore, SqliteTranscriptStore};
use crate::{BranchId, MemoryStore, MessageId, PersistedMessage, PersistedRole, TranscriptStore};

/// The stamp of a file no test has touched since it was indexed.
const STAMP: FileStamp = FileStamp {
    mtime_ns: 1,
    size: 1,
};

/// The returned temp dir must outlive the stores.
async fn fresh() -> (SqliteHandle, SqliteSemanticStore, tempfile::TempDir) {
    let temp = tempfile::tempdir().unwrap();
    let handle = SqliteHandle::open(&temp.path().join("memory.db"))
        .await
        .unwrap();
    let store = SqliteSemanticStore::new(handle.clone());
    (handle, store, temp)
}

/// One saved session's main branch, to append user messages to.
struct Transcript {
    store: SqliteTranscriptStore,
    session: SessionId,
    branch: BranchId,
}

impl Transcript {
    async fn new(handle: &SqliteHandle) -> Self {
        let store = SqliteTranscriptStore::new(handle.clone());
        let session = SessionId::new();
        let branch = store.begin_session_with_main_branch(session).await.unwrap();
        Self {
            store,
            session,
            branch,
        }
    }

    /// Append user message `text`, returning its id and stored chunks.
    async fn append(&self, text: &str) -> (MessageId, Vec<ChunkId>) {
        let appended = self
            .store
            .begin_turn(self.branch, PersistedMessage::user(text))
            .await
            .unwrap()
            .message;
        (appended.id, appended.chunks)
    }

    /// Append user message `text`, short enough to be one chunk, and embed that chunk.
    async fn embedded_chunk(
        &self,
        store: &SqliteSemanticStore,
        text: &str,
        vector: &[f32],
    ) -> ChunkId {
        let (_, chunks) = self.append(text).await;
        let [chunk] = chunks[..] else {
            panic!("{text:?} should be one chunk, got {chunks:?}");
        };
        embed_chunk(store, chunk, vector).await;
        chunk
    }
}

/// 2-d unit vector at `angle` radians.
fn unit_vec(angle: f32) -> Vec<f32> {
    vec![angle.cos(), angle.sin()]
}

async fn embed_chunk(store: &SqliteSemanticStore, chunk_id: ChunkId, vector: &[f32]) {
    store
        .store_embeddings(vec![Embedding {
            target: EmbeddingTarget::Chunk(chunk_id),
            text: String::new(),
            vector: Some(vector.to_vec()),
        }])
        .await
        .unwrap();
}

async fn save_memory(handle: &SqliteHandle, key: &str, value: &str) -> MemoryId {
    SqliteMemoryStore::new(handle.clone())
        .save(key, value.into())
        .await
        .unwrap()
        .id
}

async fn embed_memory(
    store: &SqliteSemanticStore,
    memory_id: MemoryId,
    value: &str,
    vector: &[f32],
) {
    store
        .store_embeddings(vec![Embedding {
            target: EmbeddingTarget::Memory(memory_id),
            text: value.to_string(),
            vector: Some(vector.to_vec()),
        }])
        .await
        .unwrap();
}

#[tokio::test]
async fn nearest_chunks_returns_each_message_once_through_its_best_chunk() {
    let (handle, store, _temp) = fresh().await;
    let transcript = Transcript::new(&handle).await;
    let long_text = format!("{}{}", "head ".repeat(100), "tail ".repeat(60));
    let (long_message, long_chunks) = transcript.append(&long_text).await;
    let [first, second] = long_chunks[..] else {
        panic!("expected two chunks, got {}", long_chunks.len());
    };
    embed_chunk(&store, first, &unit_vec(0.3)).await;
    embed_chunk(&store, second, &unit_vec(0.0)).await;
    let short = transcript
        .embedded_chunk(&store, "short", &unit_vec(1.5))
        .await;

    let hits = store
        .nearest_chunks(unit_vec(0.0), 3, ChunkScope::Chats(SessionScope::All))
        .await
        .unwrap();
    let ids: Vec<ChunkId> = hits.iter().map(|h| h.chunk_id).collect();
    assert_eq!(ids, [second, short]);
    assert!((hits[0].similarity - 1.0).abs() < 1e-4);
    assert!(hits[0].similarity > hits[1].similarity);
    assert!(
        hits[0].text.ends_with("tail ") && hits[0].text.len() < long_text.len(),
        "a tail match must return the tail window, not the message"
    );
    assert_eq!(hits[1].text, "short");
    let ChunkLocator::Message { message_id, .. } = hits[0].locator else {
        panic!("a chat search must hit messages");
    };
    assert_eq!(message_id, long_message);
    for hit in &hits {
        let ChunkLocator::Message {
            session_id, role, ..
        } = hit.locator
        else {
            panic!("a chat search must hit messages");
        };
        assert_eq!(session_id, transcript.session);
        assert_eq!(role, PersistedRole::User);
    }
}

#[tokio::test]
async fn nearest_chunks_keeps_the_best_top_k() {
    let (handle, store, _temp) = fresh().await;
    let transcript = Transcript::new(&handle).await;
    let mut ids = Vec::new();
    for i in 0..10 {
        let angle = (i as f32) * 0.1;
        ids.push(
            transcript
                .embedded_chunk(&store, &format!("m{i}"), &unit_vec(angle))
                .await,
        );
    }

    for (top_k, expected) in [(3, &ids[..3]), (usize::MAX, &ids[..])] {
        let hits = store
            .nearest_chunks(unit_vec(0.0), top_k, ChunkScope::Chats(SessionScope::All))
            .await
            .unwrap();
        let got: Vec<ChunkId> = hits.iter().map(|h| h.chunk_id).collect();
        assert_eq!(got, expected, "top_k {top_k}");
    }
}

#[tokio::test]
async fn nearest_chunks_can_exclude_or_keep_one_session() {
    let (handle, store, _temp) = fresh().await;
    let past = Transcript::new(&handle).await;
    let current = Transcript::new(&handle).await;
    let past_chunk = past.embedded_chunk(&store, "past", &unit_vec(0.4)).await;
    let current_chunk = current
        .embedded_chunk(&store, "current", &unit_vec(0.0))
        .await;

    let all = store
        .nearest_chunks(unit_vec(0.0), 5, ChunkScope::Chats(SessionScope::All))
        .await
        .unwrap();
    let all_ids: Vec<ChunkId> = all.iter().map(|h| h.chunk_id).collect();
    assert_eq!(
        all_ids,
        [current_chunk, past_chunk],
        "the excluded session must hold the closer match"
    );

    let others = store
        .nearest_chunks(
            unit_vec(0.0),
            5,
            ChunkScope::Chats(SessionScope::AllExcept(current.session)),
        )
        .await
        .unwrap();
    let other_ids: Vec<(ChunkId, SessionId)> = others
        .iter()
        .map(|h| {
            let ChunkLocator::Message { session_id, .. } = h.locator else {
                panic!("a chat search must hit messages");
            };
            (h.chunk_id, session_id)
        })
        .collect();
    assert_eq!(other_ids, [(past_chunk, past.session)]);

    let only_past = store
        .nearest_chunks(
            unit_vec(0.0),
            5,
            ChunkScope::Chats(SessionScope::Only(past.session)),
        )
        .await
        .unwrap();
    let only_past_ids: Vec<ChunkId> = only_past.iter().map(|h| h.chunk_id).collect();
    assert_eq!(only_past_ids, [past_chunk]);
}

#[tokio::test]
async fn documents_and_chats_rank_apart_and_reindexing_replaces_document_chunks() {
    let (handle, store, _temp) = fresh().await;
    let transcript = Transcript::new(&handle).await;
    let message_chunk = transcript
        .embedded_chunk(&store, "chat", &unit_vec(0.0))
        .await;
    let path = PathBuf::from("/notes/plan.md");
    let pass = store.begin_index_pass(Path::new("/notes")).await.unwrap();
    let stale = store
        .index_document(pass, path.clone(), STAMP, "draft".into())
        .await
        .unwrap();
    embed_chunk(&store, stale[0], &unit_vec(0.1)).await;
    let section = format!("{}{}", "intro ".repeat(90), "detail ".repeat(60));
    let fresh_chunks = store
        .index_document(pass, path.clone(), STAMP, section.clone())
        .await
        .unwrap();
    assert!(
        store
            .index_document(pass, path.clone(), STAMP, section.clone())
            .await
            .unwrap()
            .is_empty(),
        "unchanged text must keep its chunks"
    );
    let [intro, detail] = fresh_chunks[..] else {
        panic!("expected two chunks, got {fresh_chunks:?}");
    };
    embed_chunk(&store, stale[0], &unit_vec(0.0)).await;
    embed_chunk(&store, detail, &unit_vec(0.2)).await;

    let chats = store
        .nearest_chunks(unit_vec(0.0), 5, ChunkScope::Chats(SessionScope::All))
        .await
        .unwrap();
    let chat_ids: Vec<ChunkId> = chats.iter().map(|h| h.chunk_id).collect();
    assert_eq!(chat_ids, [message_chunk], "chat search must skip documents");

    let documents = store
        .nearest_chunks(unit_vec(0.0), 5, ChunkScope::Documents)
        .await
        .unwrap();
    let [hit] = &documents[..] else {
        panic!("one document must hit through its re-indexed chunk, got {documents:?}");
    };
    assert_eq!(hit.chunk_id, detail);
    let ChunkLocator::Document {
        path: hit_path,
        bytes,
        ..
    } = &hit.locator
    else {
        panic!("a document search must hit documents");
    };
    assert_eq!(hit_path, &path);
    assert_eq!(hit.text, section[bytes.clone()]);
    let adjacent = store.adjacent_chunks(&hit.locator).await.unwrap();
    assert_eq!(adjacent.before.map(|chunk| chunk.chunk_id), Some(intro));
    assert!(adjacent.after.is_none() && bytes.start > 0);
    assert_eq!(
        store.count_missing().await.unwrap(),
        UnindexedRows {
            chunks: 1,
            memories: 0
        }
    );
}

#[tokio::test]
async fn a_pass_skips_unchanged_files_keeps_untouched_chunks_and_drops_unseen_documents() {
    let (_handle, store, _temp) = fresh().await;
    let root = Path::new("/notes");
    let (plan, gone) = (root.join("plan.md"), root.join("gone.md"));
    let touched = FileStamp {
        mtime_ns: 2,
        ..STAMP
    };
    let section = format!("{}{}", "intro ".repeat(90), "detail ".repeat(60));

    let first = store.begin_index_pass(root).await.unwrap();
    let indexed = store
        .index_document(first, plan.clone(), STAMP, section.clone())
        .await
        .unwrap();
    let [intro, detail] = indexed[..] else {
        panic!("expected two chunks, got {indexed:?}");
    };
    store
        .index_document(
            first,
            gone.clone(),
            STAMP,
            "deleted before the next pass".into(),
        )
        .await
        .unwrap();
    assert_eq!(store.finish_index_pass(first).await.unwrap(), 0);
    embed_chunk(&store, intro, &unit_vec(0.0)).await;
    embed_chunk(&store, detail, &unit_vec(0.1)).await;

    let second = store.begin_index_pass(root).await.unwrap();
    assert!(
        store
            .document_unchanged(second, &plan, STAMP)
            .await
            .unwrap()
    );
    assert!(
        !store
            .document_unchanged(second, &plan, touched)
            .await
            .unwrap()
    );
    let rewritten = store
        .index_document(second, plan.clone(), touched, section.clone())
        .await
        .unwrap();
    assert!(rewritten.is_empty(), "the same text must keep its chunks");
    assert!(
        store
            .document_unchanged(second, &plan, touched)
            .await
            .unwrap()
    );
    let edited = store
        .index_document(second, plan.clone(), touched, format!("{section}!"))
        .await
        .unwrap();
    assert_eq!(
        edited.len(),
        1,
        "only the chunk the edit reached is replaced"
    );
    assert_eq!(store.finish_index_pass(second).await.unwrap(), 1);
    assert!(!store.remove_document(&gone).await.unwrap());

    let hits = store
        .nearest_chunks(unit_vec(0.0), 5, ChunkScope::Documents)
        .await
        .unwrap();
    let hit_ids: Vec<ChunkId> = hits.iter().map(|hit| hit.chunk_id).collect();
    assert_eq!(hit_ids, [intro], "the untouched chunk keeps its vector");
    assert_eq!(store.count_missing().await.unwrap().chunks, 1);

    assert_eq!(store.remove_index_root(root).await.unwrap(), 1);
    assert_eq!(store.count_missing().await.unwrap().chunks, 0);
    assert!(store.begin_index_pass(root).await.unwrap() != second);
}

#[tokio::test]
async fn upsert_replaces_memory_embedding_in_place() {
    let (handle, store, _temp) = fresh().await;
    let mem_id = save_memory(&handle, "k", "v").await;
    embed_memory(&store, mem_id, "v", &unit_vec(0.0)).await;
    embed_memory(&store, mem_id, "v", &unit_vec(std::f32::consts::FRAC_PI_2)).await;

    assert_eq!(
        store.count_missing().await.unwrap(),
        UnindexedRows::default()
    );
    let hits = store.nearest_memories(unit_vec(0.0), 5).await.unwrap();
    let [hit] = hits.as_slice() else {
        panic!("expected one hit, got {hits:?}");
    };
    assert!(
        hit.similarity.abs() < 1e-4,
        "stale vector survived: {}",
        hit.similarity
    );
}

#[tokio::test]
async fn overwriting_a_memory_value_drops_its_stale_embedding() {
    let (handle, store, _temp) = fresh().await;
    let mem_id = save_memory(&handle, "editor", "vim").await;
    embed_memory(&store, mem_id, "vim", &unit_vec(0.0)).await;

    assert_eq!(save_memory(&handle, "editor", "emacs").await, mem_id);
    embed_memory(&store, mem_id, "vim", &unit_vec(0.0)).await;

    assert_eq!(
        store.count_missing().await.unwrap(),
        UnindexedRows {
            chunks: 0,
            memories: 1
        },
        "neither the trigger nor a late embedding of the old value may leave it indexed"
    );
    assert_eq!(
        store.nearest_memories(unit_vec(0.0), 5).await.unwrap(),
        Vec::<MemoryHit>::new()
    );
    assert_eq!(
        store.memories_missing_embedding(None, 100).await.unwrap(),
        [(mem_id, "emacs".to_string())]
    );
}

#[tokio::test]
async fn non_finite_vectors_never_enter_top_k() {
    let (handle, store, _temp) = fresh().await;
    let transcript = Transcript::new(&handle).await;
    transcript
        .embedded_chunk(&store, "nan", &[f32::NAN, 0.0])
        .await;
    transcript
        .embedded_chunk(&store, "inf", &[f32::INFINITY, 0.0])
        .await;
    let far = transcript
        .embedded_chunk(&store, "far", &unit_vec(1.5))
        .await;
    let near = transcript
        .embedded_chunk(&store, "near", &unit_vec(0.1))
        .await;

    let hits = store
        .nearest_chunks(unit_vec(0.0), 2, ChunkScope::Chats(SessionScope::All))
        .await
        .unwrap();
    let got: Vec<ChunkId> = hits.iter().map(|h| h.chunk_id).collect();
    assert_eq!(got, [near, far]);
    assert!(hits.iter().all(|h| h.similarity.is_finite()));
}

#[tokio::test]
async fn missing_embedding_lists_only_unindexed_rows() {
    let (handle, store, _temp) = fresh().await;
    let transcript = Transcript::new(&handle).await;

    transcript.embedded_chunk(&store, "x", &unit_vec(0.0)).await;
    let (_, first) = transcript.append("y").await;
    let first_bare = first[0];
    let (_, second) = transcript.append("z").await;
    let second_bare = second[0];

    let indexed_mem = save_memory(&handle, "indexed", "v1").await;
    embed_memory(&store, indexed_mem, "v1", &unit_vec(0.0)).await;
    let bare_mem = save_memory(&handle, "bare", "v2").await;

    assert_eq!(
        store.chunks_missing_embedding(None, 100).await.unwrap(),
        [
            (first_bare, "y".to_string()),
            (second_bare, "z".to_string())
        ]
    );
    assert_eq!(
        store.chunks_missing_embedding(None, 1).await.unwrap(),
        [(first_bare, "y".to_string())]
    );
    assert_eq!(
        store
            .chunks_missing_embedding(Some(first_bare), 100)
            .await
            .unwrap(),
        [(second_bare, "z".to_string())]
    );
    assert_eq!(
        store.memories_missing_embedding(None, 100).await.unwrap(),
        [(bare_mem, "v2".to_string())]
    );

    embed_memory(&store, bare_mem, "v2", &unit_vec(0.0)).await;
    assert_eq!(
        store.memories_missing_embedding(None, 100).await.unwrap(),
        Vec::<(MemoryId, String)>::new()
    );
}

#[tokio::test]
async fn unembeddable_rows_are_neither_missing_nor_searched_until_cleared_or_changed() {
    let (handle, store, _temp) = fresh().await;
    let transcript = Transcript::new(&handle).await;
    let (_, chunks) = transcript.append("y").await;
    let chunk = chunks[0];
    let cleared_mem = save_memory(&handle, "cleared", "v1").await;
    let changed_mem = save_memory(&handle, "changed", "v2").await;
    let unembeddable = [
        (EmbeddingTarget::Chunk(chunk), "y"),
        (EmbeddingTarget::Memory(cleared_mem), "v1"),
        (EmbeddingTarget::Memory(changed_mem), "v2"),
    ]
    .map(|(target, text)| Embedding {
        target,
        text: text.to_string(),
        vector: None,
    })
    .to_vec();
    store.store_embeddings(unembeddable).await.unwrap();

    assert_eq!(
        store.count_missing().await.unwrap(),
        UnindexedRows::default()
    );
    assert!(
        store
            .nearest_chunks(unit_vec(0.0), 5, ChunkScope::Chats(SessionScope::All))
            .await
            .unwrap()
            .is_empty()
    );
    assert!(
        store
            .nearest_memories(unit_vec(0.0), 5)
            .await
            .unwrap()
            .is_empty()
    );

    save_memory(&handle, "changed", "v3").await;
    assert_eq!(
        store.memories_missing_embedding(None, 100).await.unwrap(),
        [(changed_mem, "v3".to_string())]
    );

    assert_eq!(store.clear_unembeddable().await.unwrap(), 2);
    assert_eq!(
        store.chunks_missing_embedding(None, 100).await.unwrap(),
        [(chunk, "y".to_string())]
    );
    assert_eq!(
        store.count_missing().await.unwrap(),
        UnindexedRows {
            chunks: 1,
            memories: 2
        }
    );
}

#[tokio::test]
async fn adopting_a_different_embedding_space_drops_every_vector() {
    let (handle, store, _temp) = fresh().await;
    let transcript = Transcript::new(&handle).await;
    let memory = save_memory(&handle, "k", "v").await;
    let space = EmbeddingSpace {
        model: "owner/repo:Q8_0".to_string(),
        dim: 2,
        server_args: String::new(),
    };
    assert_eq!(store.adopt_embedding_space(&space).await.unwrap(), None);
    let chunk = transcript
        .embedded_chunk(&store, "text", &unit_vec(0.0))
        .await;
    embed_memory(&store, memory, "v", &unit_vec(0.0)).await;

    assert_eq!(store.adopt_embedding_space(&space).await.unwrap(), None);
    assert_eq!(
        store.count_missing().await.unwrap(),
        UnindexedRows::default()
    );

    let rearranged = EmbeddingSpace {
        server_args: "--embd-normalize 1".to_string(),
        ..space.clone()
    };
    assert_eq!(
        store.adopt_embedding_space(&rearranged).await.unwrap(),
        Some(space)
    );
    assert_eq!(
        store.count_missing().await.unwrap(),
        UnindexedRows {
            chunks: 1,
            memories: 1
        }
    );
    let all_chats = ChunkScope::Chats(SessionScope::All);
    let hits = store.nearest_chunks(unit_vec(0.0), 5, all_chats).await;
    assert!(hits.unwrap().is_empty());
    assert!(
        store
            .nearest_memories(unit_vec(0.0), 5)
            .await
            .unwrap()
            .is_empty()
    );

    embed_chunk(&store, chunk, &unit_vec(0.0)).await;
    let hits = store.nearest_chunks(unit_vec(0.0), 5, all_chats).await;
    assert_eq!(hits.unwrap()[0].chunk_id, chunk);
    assert_eq!(
        store.adopt_embedding_space(&rearranged).await.unwrap(),
        None
    );
}

#[tokio::test]
async fn batch_skips_targets_deleted_before_it_lands() {
    let (handle, store, _temp) = fresh().await;
    let kept = save_memory(&handle, "kept", "v").await;
    let deleted = save_memory(&handle, "deleted", "v").await;
    SqliteMemoryStore::new(handle.clone())
        .delete("deleted")
        .await
        .unwrap();

    let embeddings = [deleted, kept]
        .map(|memory_id| Embedding {
            target: EmbeddingTarget::Memory(memory_id),
            text: "v".to_string(),
            vector: Some(unit_vec(0.0)),
        })
        .to_vec();
    store.store_embeddings(embeddings).await.unwrap();

    let hits = store.nearest_memories(unit_vec(0.0), 5).await.unwrap();
    let hit_ids: Vec<MemoryId> = hits.iter().map(|hit| hit.memory_id).collect();
    assert_eq!(hit_ids, [kept]);
}

#[test]
fn vector_to_blob_round_trips() {
    let vector = vec![0.5f32, -0.25, 0.125, 1e-6];
    let blob = vector_to_blob(&vector);
    let (words, _) = blob.as_chunks::<4>();
    let decoded: Vec<f32> = words.iter().copied().map(f32::from_le_bytes).collect();
    assert_eq!(decoded, vector);
}

#[test]
fn score_against_rejects_malformed_blobs() {
    let q = [1.0f32, 2.0];
    assert_eq!(score_against(&q, &vector_to_blob(&[3.0, 4.0])), Some(11.0));
    assert_eq!(score_against(&q, &vector_to_blob(&[1.0, 0.0, 0.0])), None);
    assert_eq!(score_against(&q, &[0u8; 7]), None);
}

#[test]
fn quantised_score_tracks_the_exact_dot_product() {
    let stored: Vec<f32> = (0..768u16)
        .map(|i| (f32::from(i) * 0.37).sin() / 20.0)
        .collect();
    let query: Vec<f32> = (0..768u16)
        .map(|i| (f32::from(i) * 0.11).cos() / 20.0)
        .collect();
    let exact = score_against(&query, &vector_to_blob(&stored)).unwrap();
    let blob = quantised_blob(&stored);

    assert_eq!(blob.len(), 4 + stored.len());
    let approximate = score_against_quantised(&query, &blob).unwrap();
    assert!(
        (approximate - exact).abs() < 2e-3,
        "quantised {approximate} strays from exact {exact}"
    );
    assert_eq!(
        score_against_quantised(&query, &blob[..blob.len() - 1]),
        None
    );
    assert_eq!(score_against_quantised(&query, &[0u8; 3]), None);
}
