use std::pin::pin;

use super::*;
use crate::sqlite::SqliteTranscriptStore;
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

    /// Append user message `text`, short enough to be one chunk, and embed its text.
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
        embed_text(store, ChunkSource::Message, text, vector).await;
        chunk
    }
}

/// A file `unindexed_files` reports as not read at `stamp`.
fn unread(path: &Path, stamp: FileStamp) -> UnindexedFile {
    UnindexedFile {
        path: path.to_path_buf(),
        stamp,
        skipped: None,
    }
}

/// `text` as the pages of a file whose format has none.
fn one_page(text: impl Into<String>) -> Vec<String> {
    vec![text.into()]
}

/// Where `hit` sits in its document.
fn locator_of(hit: &DocumentHit) -> ChunkLocator {
    ChunkLocator::Document {
        document_id: hit.document_id,
        path: hit.path.clone(),
        page: hit.page,
        page_count: hit.page_count,
        bytes: hit.bytes.clone(),
        indexed_at: hit.indexed_at,
    }
}

/// 2-d unit vector at `angle` radians.
fn unit_vec(angle: f32) -> Vec<f32> {
    vec![angle.cos(), angle.sin()]
}

/// Store `vector` for `text` as the chunks cut from `source` hold it.
async fn embed_text(store: &SqliteSemanticStore, source: ChunkSource, text: &str, vector: &[f32]) {
    store
        .store_embeddings(vec![Embedding {
            target: EmbeddingTarget::ChunkText(source),
            text: text.to_string(),
            vector: Some(vector.to_vec()),
        }])
        .await
        .unwrap();
}

/// Store `vector` for `text` as the document chunks hold it.
async fn embed_document_text(store: &SqliteSemanticStore, text: &str, vector: &[f32]) {
    embed_text(store, ChunkSource::Document, text, vector).await;
}

/// The texts listed for embedding among the chunks cut from `source`.
async fn texts_to_embed(store: &SqliteSemanticStore, source: ChunkSource) -> Vec<String> {
    let listed = store.chunks_missing_embedding(source, None, 100);
    let listed = listed.await.unwrap();
    listed.into_iter().map(|(_, text)| text).collect()
}

/// How many vectors and unembeddable marks `table` holds.
async fn stored_vectors(handle: &SqliteHandle, table: &'static str) -> u64 {
    let count = move |c: &rusqlite::Connection| {
        c.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
    };
    handle.read("count vectors", count).await.unwrap()
}

/// Whether `wake` was signalled since it was last asked; asking takes the signal.
fn signalled(wake: &Notify) -> bool {
    let mut notified = pin!(wake.notified());
    notified.as_mut().enable()
}

async fn save_memory(handle: &SqliteHandle, key: &str, value: &str) -> MemoryId {
    MemoryStore::new(handle.clone())
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
    let [_, second] = long_chunks[..] else {
        panic!("expected two chunks, got {}", long_chunks.len());
    };
    let windows = texts_to_embed(&store, ChunkSource::Message).await;
    embed_text(&store, ChunkSource::Message, &windows[0], &unit_vec(0.3)).await;
    embed_text(&store, ChunkSource::Message, &windows[1], &unit_vec(0.0)).await;
    let short = transcript
        .embedded_chunk(&store, "short", &unit_vec(1.5))
        .await;

    let hits = store
        .nearest_chunks(unit_vec(0.0), 3, SessionScope::All)
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
        panic!("a message search must hit messages");
    };
    assert_eq!(message_id, long_message);
    for hit in &hits {
        let ChunkLocator::Message {
            session_id, role, ..
        } = hit.locator
        else {
            panic!("a message search must hit messages");
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
            .nearest_chunks(unit_vec(0.0), top_k, SessionScope::All)
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
        .nearest_chunks(unit_vec(0.0), 5, SessionScope::All)
        .await
        .unwrap();
    let all_ids: Vec<ChunkId> = all.iter().map(|h| h.chunk_id).collect();
    assert_eq!(
        all_ids,
        [current_chunk, past_chunk],
        "the excluded session must hold the closer match"
    );

    let others = store
        .nearest_chunks(unit_vec(0.0), 5, SessionScope::AllExcept(current.session))
        .await
        .unwrap();
    let other_ids: Vec<(ChunkId, SessionId)> = others
        .iter()
        .map(|h| {
            let ChunkLocator::Message { session_id, .. } = h.locator else {
                panic!("a message search must hit messages");
            };
            (h.chunk_id, session_id)
        })
        .collect();
    assert_eq!(other_ids, [(past_chunk, past.session)]);

    let only_past = store
        .nearest_chunks(unit_vec(0.0), 5, SessionScope::Only(past.session))
        .await
        .unwrap();
    let only_past_ids: Vec<ChunkId> = only_past.iter().map(|h| h.chunk_id).collect();
    assert_eq!(only_past_ids, [past_chunk]);
}

#[tokio::test]
async fn documents_and_messages_rank_apart_and_reindexing_replaces_document_chunks() {
    let (handle, store, _temp) = fresh().await;
    let transcript = Transcript::new(&handle).await;
    let message_chunk = transcript
        .embedded_chunk(&store, "chat", &unit_vec(0.0))
        .await;
    let path = PathBuf::from("/notes/plan.md");
    let pass = store.begin_index_pass(Path::new("/notes")).await.unwrap();
    let draft = store.index_document(
        pass,
        path.clone(),
        STAMP,
        one_page("draft"),
        EmptyPages::Final,
    );
    draft.await.unwrap();
    embed_document_text(&store, "draft", &unit_vec(0.1)).await;
    let section = format!("{}{}", "intro ".repeat(90), "detail ".repeat(60));
    let fresh_chunks = store
        .index_document(
            pass,
            path.clone(),
            STAMP,
            one_page(&section),
            EmptyPages::Final,
        )
        .await
        .unwrap();
    assert!(
        store
            .index_document(
                pass,
                path.clone(),
                STAMP,
                one_page(&section),
                EmptyPages::Final
            )
            .await
            .unwrap()
            .is_empty(),
        "unchanged text must keep its chunks"
    );
    let [intro, detail] = fresh_chunks[..] else {
        panic!("expected two chunks, got {fresh_chunks:?}");
    };
    assert_eq!(
        stored_vectors(&handle, "document_vectors").await,
        0,
        "the vector of replaced text goes with its last chunk"
    );
    embed_document_text(&store, "draft", &unit_vec(0.0)).await;
    let detail_text = &texts_to_embed(&store, ChunkSource::Document).await[1];
    embed_document_text(&store, detail_text, &unit_vec(0.2)).await;

    let messages = store
        .nearest_chunks(unit_vec(0.0), 5, SessionScope::All)
        .await
        .unwrap();
    let message_ids: Vec<ChunkId> = messages.iter().map(|h| h.chunk_id).collect();
    assert_eq!(
        message_ids,
        [message_chunk],
        "message search must skip documents"
    );

    let documents = store
        .search_documents(String::new(), unit_vec(0.0), 5)
        .await
        .unwrap();
    let [hit] = &documents[..] else {
        panic!("one document must hit through its re-indexed chunk, got {documents:?}");
    };
    assert_eq!(hit.chunk_id, detail);
    assert_eq!((&hit.path, hit.page, hit.page_count), (&path, 0, 1));
    assert_eq!(hit.text, section[hit.bytes.clone()]);
    let adjacent = store.adjacent_chunks(&locator_of(hit)).await.unwrap();
    assert_eq!(adjacent.before.map(|chunk| chunk.chunk_id), Some(intro));
    assert!(adjacent.after.is_none() && hit.bytes.start > 0);
    assert_eq!(
        store.count_missing().await.unwrap(),
        UnindexedRows {
            message_chunks: 0,
            document_chunks: 1,
            memories: 0
        }
    );
}

#[tokio::test]
async fn document_text_narrows_a_byte_range_of_one_page_to_whole_characters() {
    let (_handle, store, _temp) = fresh().await;
    let path = PathBuf::from("/notes/menu.pdf");
    let pass = store.begin_index_pass(Path::new("/notes")).await.unwrap();
    let pages = vec!["café crème".to_string(), String::new(), "thé".to_string()];
    store
        .index_document(pass, path.clone(), STAMP, pages, EmptyPages::Final)
        .await
        .unwrap();

    let inside = store.document_text(&path, 0, 4..9).await.unwrap().unwrap();
    assert_eq!(
        (inside.text.as_str(), inside.bytes, inside.page_bytes),
        (" cr", 5..8, 12),
        "the halves of é and è at either end must be left out"
    );
    let past = store
        .document_text(&path, 0, 40..50)
        .await
        .unwrap()
        .unwrap();
    assert_eq!((past.text.as_str(), past.bytes), ("", 40..40));
    let last = store.document_text(&path, 2, 0..50).await.unwrap().unwrap();
    assert_eq!(
        (last.text.as_str(), last.page, last.page_count),
        ("thé", 2, 3)
    );
    let textless = store.document_text(&path, 1, 0..50).await.unwrap().unwrap();
    assert_eq!((textless.text.as_str(), textless.page_bytes), ("", 0));
    let no_such_page = store.document_text(&path, 3, 0..50).await.unwrap().unwrap();
    assert_eq!(
        (no_such_page.text.as_str(), no_such_page.page_count),
        ("", 3)
    );
    let unindexed = store.document_text(Path::new("/notes/other.md"), 0, 0..9);
    assert_eq!(unindexed.await.unwrap(), None);
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

    let index = |pass: IndexPass, path: &Path, stamp: FileStamp, text: &str| {
        let (path, pages) = (path.to_path_buf(), one_page(text));
        store.index_document(pass, path, stamp, pages, EmptyPages::Final)
    };

    let first = store.begin_index_pass(root).await.unwrap();
    let indexed = index(first, &plan, STAMP, &section).await.unwrap();
    assert_eq!(indexed.len(), 2);
    let deleted = index(first, &gone, STAMP, "deleted before the next pass");
    deleted.await.unwrap();
    assert_eq!(store.finish_index_pass(first).await.unwrap(), 0);
    let texts = texts_to_embed(&store, ChunkSource::Document).await;
    let [intro_text, detail_text, _] = &texts[..] else {
        panic!("expected a text per chunk, got {texts:?}");
    };
    embed_document_text(&store, intro_text, &unit_vec(0.0)).await;
    embed_document_text(&store, detail_text, &unit_vec(0.1)).await;

    let second = store.begin_index_pass(root).await.unwrap();
    let never_indexed = root.join("new.md");
    let changed = store
        .unindexed_files(
            second,
            vec![
                (plan.clone(), STAMP),
                (never_indexed.clone(), STAMP),
                (plan.clone(), touched),
            ],
        )
        .await
        .unwrap();
    assert_eq!(
        changed,
        [unread(&never_indexed, STAMP), unread(&plan, touched)]
    );
    let rewritten = index(second, &plan, touched, &section).await.unwrap();
    assert!(rewritten.is_empty(), "the same text must keep its chunks");
    let changed = store
        .unindexed_files(second, vec![(plan.clone(), touched)])
        .await
        .unwrap();
    assert!(changed.is_empty(), "the stored stamp is the touched one");
    let edit = section.replace("detail", "change");
    let edited = index(second, &plan, touched, &edit).await.unwrap();
    assert_eq!(edited.len(), 1, "only the text the edit reached is new");
    assert_eq!(store.finish_index_pass(second).await.unwrap(), 1);
    assert!(!store.remove_document(&gone).await.unwrap());

    let hits = store
        .search_documents(String::new(), unit_vec(0.0), 5)
        .await
        .unwrap();
    let hit_texts: Vec<&str> = hits.iter().map(|hit| hit.text.as_str()).collect();
    assert_eq!(
        hit_texts,
        [intro_text],
        "the untouched text keeps its vector"
    );
    assert_eq!(store.count_missing().await.unwrap().document_chunks, 1);

    assert_eq!(store.index_roots().await.unwrap(), [root]);
    assert_eq!(store.count_document_chunks().await.unwrap(), 2);
    assert_eq!(
        store.index_root_counts().await.unwrap(),
        [IndexRootCounts {
            root: root.to_path_buf(),
            documents: 1,
            chunks: 2,
            unembedded_chunks: 1,
        }]
    );
    assert_eq!(store.remove_index_root(root).await.unwrap(), 1);
    assert!(store.index_roots().await.unwrap().is_empty());
    assert_eq!(store.count_document_chunks().await.unwrap(), 0);
    assert_eq!(store.count_missing().await.unwrap().document_chunks, 0);
    assert!(store.begin_index_pass(root).await.unwrap() != second);
}

#[tokio::test]
async fn a_skipped_file_is_not_read_again_until_it_changes_is_forgotten_or_goes_unseen() {
    let (_handle, store, _temp) = fresh().await;
    let root = Path::new("/notes");
    let (blob, scan) = (root.join("blob.dat"), root.join("scan.pdf"));
    let touched = FileStamp {
        mtime_ns: 2,
        ..STAMP
    };
    let check = |pass: IndexPass, path: &Path, stamp: FileStamp| {
        store.unindexed_files(pass, vec![(path.to_path_buf(), stamp)])
    };
    let skipped = |path: &Path, cause: SkippedRead| UnindexedFile {
        skipped: Some(cause),
        ..unread(path, STAMP)
    };

    let first = store.begin_index_pass(root).await.unwrap();
    let indexed = store.index_document(
        first,
        blob.clone(),
        STAMP,
        one_page("text"),
        EmptyPages::Final,
    );
    assert_eq!(indexed.await.unwrap().len(), 1);
    for (path, cause) in [(&blob, SkippedRead::Binary), (&scan, SkippedRead::TimedOut)] {
        let recorded = store.record_skipped_file(first, path.clone(), STAMP, cause);
        recorded.await.unwrap();
    }
    assert_eq!(store.count_document_chunks().await.unwrap(), 0);
    assert_eq!(store.finish_index_pass(first).await.unwrap(), 0);

    let second = store.begin_index_pass(root).await.unwrap();
    assert_eq!(
        check(second, &blob, STAMP).await.unwrap(),
        [skipped(&blob, SkippedRead::Binary)]
    );
    assert_eq!(
        check(second, &scan, touched).await.unwrap(),
        [unread(&scan, touched)]
    );
    let reread = store.index_document(
        second,
        scan.clone(),
        touched,
        one_page("text"),
        EmptyPages::Final,
    );
    reread.await.unwrap();
    assert!(check(second, &scan, touched).await.unwrap().is_empty());
    assert_eq!(store.finish_index_pass(second).await.unwrap(), 0);

    let third = store.begin_index_pass(root).await.unwrap();
    assert_eq!(store.finish_index_pass(third).await.unwrap(), 1);
    let fourth = store.begin_index_pass(root).await.unwrap();
    assert_eq!(
        check(fourth, &blob, STAMP).await.unwrap(),
        [unread(&blob, STAMP)],
        "a skipped file no pass saw is forgotten"
    );
    let recorded = store.record_skipped_file(fourth, blob.clone(), STAMP, SkippedRead::Binary);
    recorded.await.unwrap();
    assert_eq!(store.forget_skipped_files().await.unwrap(), 1);
    assert_eq!(
        check(fourth, &blob, STAMP).await.unwrap(),
        [unread(&blob, STAMP)]
    );
}

#[tokio::test]
async fn a_pass_over_part_of_a_root_drops_only_the_unseen_documents_under_it() {
    let (_handle, store, _temp) = fresh().await;
    let root = Path::new("/notes");
    let [plan, seen, moved_away, deleted, beside] = [
        "plan.md",
        "work/seen.md",
        "work/old/moved.md",
        "deleted.md",
        "workshop.md",
    ]
    .map(|name| root.join(name));
    let whole = store.begin_index_pass(root).await.unwrap();
    for path in [&plan, &seen, &moved_away, &deleted, &beside] {
        store
            .index_document(
                whole,
                path.clone(),
                STAMP,
                one_page("text"),
                EmptyPages::Final,
            )
            .await
            .unwrap();
    }

    let part = store.begin_index_pass(root).await.unwrap();
    let changed = store
        .unindexed_files(part, vec![(seen.clone(), STAMP)])
        .await
        .unwrap();
    assert!(changed.is_empty());
    let removed = store
        .finish_index_pass_under(part, vec![root.join("work"), deleted.clone()])
        .await
        .unwrap();

    assert_eq!(removed, 2);
    for (path, kept) in [
        (plan, true),
        (seen, true),
        (beside, true),
        (moved_away, false),
        (deleted, false),
    ] {
        let stored = store.document_text(&path, 0, 0..4).await.unwrap();
        assert_eq!(stored.is_some(), kept, "{}", path.display());
    }
}

#[tokio::test]
async fn a_document_stored_again_lists_only_its_new_text_wherever_the_rest_moved() {
    let (handle, store, _temp) = fresh().await;
    let path = PathBuf::from("/notes/plan.md");
    let [one, two, six] = ["# One\n\nalpha", "# Two\n\nbravo", "# Six\n\ndelta"];
    let pass = store.begin_index_pass(Path::new("/notes")).await.unwrap();
    let index = |pages: Vec<String>| {
        store.index_document(pass, path.clone(), STAMP, pages, EmptyPages::Final)
    };
    let stored = index(one_page([one, two, six].join("\n\n"))).await.unwrap();
    assert_eq!(stored.len(), 3, "a chunk per section");
    for (text, angle) in [(one, 0.0), (two, 0.1), (six, 0.2)] {
        embed_document_text(&store, text, &unit_vec(angle)).await;
    }

    let opening = "A new opening paragraph.";
    let swapped = [opening, two, one].join("\n\n");
    let added = index(vec![swapped, six.to_string()]).await.unwrap();
    assert_eq!(added.len(), 1, "only the new paragraph has no vector");
    assert_eq!(
        texts_to_embed(&store, ChunkSource::Document).await,
        [opening]
    );
    assert_eq!(stored_vectors(&handle, "document_vectors").await, 3);

    let hits = store
        .search_documents(String::new(), unit_vec(0.0), 5)
        .await
        .unwrap();
    let [hit] = &hits[..] else {
        panic!("the document must hit once, got {hits:?}");
    };
    assert_eq!((hit.text.as_str(), hit.page, hit.page_count), (one, 0, 2));
    let adjacent = store.adjacent_chunks(&locator_of(hit)).await.unwrap();
    assert_eq!(adjacent.before.map(|chunk| chunk.text), Some(two.into()));
    assert_eq!(adjacent.after.map(|chunk| chunk.text), Some(six.into()));
}

#[tokio::test]
async fn a_text_several_documents_hold_has_one_vector_until_none_of_them_holds_it() {
    let (handle, store, _temp) = fresh().await;
    let root = Path::new("/notes");
    let [original, copy, renamed] = ["a.md", "b.md", "moved/a.md"].map(|name| root.join(name));
    let shared = "the terms of the lease";
    let pass = store.begin_index_pass(root).await.unwrap();
    let index = |path: &Path| {
        let (path, pages) = (path.to_path_buf(), one_page(shared));
        store.index_document(pass, path, STAMP, pages, EmptyPages::Final)
    };
    let found = async || {
        let hits = store.search_documents(String::new(), unit_vec(0.0), 5);
        let hits = hits.await.unwrap();
        hits.into_iter().map(|hit| hit.path).collect::<Vec<_>>()
    };

    index(&original).await.unwrap();
    index(&copy).await.unwrap();
    assert_eq!(
        texts_to_embed(&store, ChunkSource::Document).await,
        [shared],
        "a text two documents hold is embedded once"
    );
    assert_eq!(store.count_missing().await.unwrap().document_chunks, 2);
    embed_document_text(&store, shared, &unit_vec(0.0)).await;
    assert_eq!(stored_vectors(&handle, "document_vectors").await, 1);
    assert_eq!(store.count_missing().await.unwrap().document_chunks, 0);
    assert_eq!(found().await, [original.clone(), copy.clone()]);

    let rows_to_embed = store.rows_to_embed();
    assert!(signalled(&rows_to_embed));
    let moved = index(&renamed).await.unwrap();
    assert!(
        moved.is_empty() && !signalled(&rows_to_embed),
        "a file stored under another path has nothing to embed"
    );

    assert!(store.remove_document(&original).await.unwrap());
    assert!(store.remove_document(&copy).await.unwrap());
    assert_eq!(found().await, [renamed.as_path()], "one holder is left");
    assert!(store.remove_document(&renamed).await.unwrap());
    assert_eq!(
        stored_vectors(&handle, "document_vectors").await,
        0,
        "the vector goes with the last chunk holding its text"
    );
    embed_document_text(&store, shared, &unit_vec(0.0)).await;
    assert_eq!(
        stored_vectors(&handle, "document_vectors").await,
        0,
        "a vector for text no chunk holds is not stored"
    );
}

#[tokio::test]
async fn a_document_with_hundreds_of_near_texts_does_not_crowd_out_the_next_nearest() {
    let (_handle, store, _temp) = fresh().await;
    let pass = store.begin_index_pass(Path::new("/notes")).await.unwrap();
    let index = |name: &str, text: String| {
        let path = Path::new("/notes").join(name);
        store.index_document(pass, path, STAMP, one_page(text), EmptyPages::Final)
    };
    let sections: Vec<String> = (0..320)
        .map(|n| format!("# Part {n}\n\nbody {n}"))
        .collect();
    let manual = index("manual.md", sections.join("\n\n")).await.unwrap();
    assert_eq!(manual.len(), sections.len(), "a chunk per section");
    index("note.md", "elsewhere".to_string()).await.unwrap();
    for section in &sections {
        embed_document_text(&store, section, &unit_vec(0.1)).await;
    }
    embed_document_text(&store, "elsewhere", &unit_vec(0.5)).await;

    let hits = store.search_documents(String::new(), unit_vec(0.0), 5);
    let found: Vec<PathBuf> = hits
        .await
        .unwrap()
        .into_iter()
        .map(|hit| hit.path)
        .collect();
    assert_eq!(
        found,
        ["/notes/manual.md", "/notes/note.md"].map(PathBuf::from)
    );
}

#[tokio::test]
async fn undoing_a_turn_deletes_the_vectors_only_its_messages_named() {
    let (handle, store, _temp) = fresh().await;
    let transcript = Transcript::new(&handle).await;
    for text in ["said twice", "said twice", "said once"] {
        transcript
            .embedded_chunk(&store, text, &unit_vec(0.0))
            .await;
    }
    assert_eq!(stored_vectors(&handle, "message_vectors").await, 2);

    for (vectors_left, undone) in [(1, "said once"), (1, "a repeat"), (0, "the last holder")] {
        let undo = transcript.store.undo_last_turn(transcript.branch);
        undo.await.unwrap();
        assert_eq!(
            stored_vectors(&handle, "message_vectors").await,
            vectors_left,
            "after undoing {undone}"
        );
    }
}

#[tokio::test]
async fn a_document_holding_the_querys_words_ranks_with_those_nearest_in_meaning() {
    let (_handle, store, _temp) = fresh().await;
    let pass = store.begin_index_pass(Path::new("/notes")).await.unwrap();
    let index = |name: &str, text: &str| {
        let path = PathBuf::from(format!("/notes/{name}"));
        store.index_document(pass, path, STAMP, one_page(text), EmptyPages::Final)
    };
    let terms = "# Terms\n\nrenewal terms of the lease\n\n# Reference\n\nfiled under REF-77";
    let terms = index("terms.md", terms).await.unwrap();
    let [terms_intro, terms_reference] = terms[..] else {
        panic!("each heading must open a chunk, got {terms:?}");
    };
    let invoice = index("invoice.md", "Invoice INV-2024-0113, due in March").await;
    let invoice = invoice.unwrap()[0];
    let ledger = index("ledger.md", "paid INV-2024-0113 in full")
        .await
        .unwrap()[0];
    let texts = texts_to_embed(&store, ChunkSource::Document).await;
    for (text, angle) in texts.iter().zip([0.0, 1.4, 1.5]) {
        embed_document_text(&store, text, &unit_vec(angle)).await;
    }
    let search = async |query: &str| {
        let hits = store.search_documents(query.to_string(), unit_vec(0.0), 5);
        let hits = hits.await.unwrap();
        hits.iter().map(|hit| hit.chunk_id).collect::<Vec<_>>()
    };

    assert_eq!(search("").await, [terms_intro, invoice]);
    assert_eq!(
        search("invoice inv-2024-0113").await,
        [invoice, terms_intro, ledger],
        "exact words must outrank a nearer vector, and find a chunk with no vector"
    );
    assert_eq!(
        search("REF-77").await,
        [terms_reference, invoice],
        "a document found both ways is shown by the chunk holding the words"
    );
    assert_eq!(
        search("paid").await,
        [ledger, terms_intro, invoice],
        "at equal rank the keyword match goes first"
    );
    let hits = store.search_documents("paid".into(), unit_vec(0.0), 5);
    let similarities: Vec<Option<f32>> = hits.await.unwrap().iter().map(|h| h.similarity).collect();
    assert!(
        matches!(similarities[..], [None, Some(near), Some(far)] if near > 0.99 && far < 0.1),
        "{similarities:?}"
    );
}

#[tokio::test]
async fn keyword_search_follows_a_document_through_reindexing_and_removal() {
    let (handle, store, _temp) = fresh().await;
    let path = PathBuf::from("/notes/scan.pdf");
    let pass = store.begin_index_pass(Path::new("/notes")).await.unwrap();
    let pages = |texts: &[&str]| texts.iter().copied().map(String::from).collect::<Vec<_>>();
    let index = |texts: &[&str]| {
        store.index_document(pass, path.clone(), STAMP, pages(texts), EmptyPages::Final)
    };
    let found = async |word: &str| {
        let hits = store.search_documents(word.to_string(), Vec::new(), 5);
        let hits = hits.await.unwrap();
        hits.iter().map(|hit| hit.page).collect::<Vec<_>>()
    };
    let indexed_chunks = async || {
        let count = |c: &rusqlite::Connection| {
            c.query_row("SELECT count(*) FROM document_chunk_fts", [], |r| {
                r.get::<_, u64>(0)
            })
        };
        handle.read("count", count).await.unwrap()
    };

    let stored = index(&["alpha", "draft"]).await.unwrap();
    assert_eq!(stored.len(), 2, "two short pages must not share a chunk");
    assert_eq!(found("alpha").await, [0]);
    assert_eq!(found("draft").await, [1]);

    index(&["Cover", "alpha", "final"]).await.unwrap();
    assert_eq!(found("alpha").await, [1], "text that moved keeps its words");
    assert!(
        found("draft").await.is_empty(),
        "replaced text must not be found"
    );
    assert_eq!(found("final").await.len(), 1);
    assert_eq!(
        indexed_chunks().await,
        3,
        "a dropped chunk leaves the index"
    );

    assert!(store.remove_document(&path).await.unwrap());
    assert!(found("alpha").await.is_empty());
    assert_eq!(
        indexed_chunks().await,
        0,
        "a removed document leaves the index"
    );
}

/// The text stored for each page of the document at `path`.
async fn page_texts(store: &SqliteSemanticStore, path: &Path, pages: usize) -> Vec<String> {
    let mut texts = Vec::with_capacity(pages);
    for page in 0..pages {
        let stored = store.document_text(path, page, 0..1024).await;
        texts.push(stored.unwrap().expect("the document is stored").text);
    }
    texts
}

#[tokio::test]
async fn a_scanned_documents_pages_are_cut_in_page_order_once_its_last_waiting_page_is_done() {
    let (_handle, store, _temp) = fresh().await;
    let path = PathBuf::from("/notes/scan.pdf");
    let pass = store.begin_index_pass(Path::new("/notes")).await.unwrap();
    let extracted = ["", "typed", "", ""].map(String::from).to_vec();
    let indexed = store.index_document(
        pass,
        path.clone(),
        STAMP,
        extracted,
        EmptyPages::AwaitTranscription,
    );
    let [typed] = indexed.await.unwrap()[..] else {
        panic!("only the page with text is cut");
    };
    let counts = async || store.count_transcriptions().await.unwrap();
    let found = async |word: &str| {
        let hits = store.search_documents(word.to_string(), Vec::new(), 5);
        let hits = hits.await.unwrap();
        hits.iter()
            .map(|hit| (hit.chunk_id, hit.page))
            .collect::<Vec<_>>()
    };
    let first = store.next_waiting_page().await.unwrap().expect("a page");
    assert_eq!(
        (first.path.as_path(), first.page, first.stamp),
        (path.as_path(), 0, STAMP)
    );
    let read = |page: usize, reading: PageReading| {
        let waiting = WaitingPage {
            page,
            ..first.clone()
        };
        store.store_page_reading(waiting, reading)
    };

    let after_last = read(3, PageReading::Text("last".into())).await.unwrap();
    let after_failed = read(2, PageReading::Failed).await.unwrap();
    assert!(
        after_last.is_empty() && after_failed.is_empty(),
        "nothing is cut while a page waits"
    );
    assert_eq!(found("last").await, []);
    assert_eq!(
        counts().await,
        TranscriptionCounts {
            waiting: 1,
            failed: 1
        }
    );

    let added = read(0, PageReading::Text("cover".into())).await.unwrap();
    let [cover, last] = added[..] else {
        panic!("each transcribed page is cut once none waits, got {added:?}");
    };
    assert_eq!(found("cover").await, [(cover, 0)]);
    assert_eq!(found("typed").await, [(typed, 1)], "a cut page is kept");
    assert_eq!(found("last").await, [(last, 3)]);
    assert_eq!(
        page_texts(&store, &path, 4).await,
        ["cover", "typed", "", "last"],
        "a failed page keeps no text"
    );
    assert_eq!(store.next_waiting_page().await.unwrap(), None);

    let late = read(0, PageReading::Text("late".into())).await.unwrap();
    assert!(late.is_empty(), "a page that no longer waits is not read");
    assert_eq!(page_texts(&store, &path, 1).await, ["cover"]);

    assert_eq!(store.retry_failed_transcriptions().await.unwrap(), 1);
    let retried = store.next_waiting_page().await.unwrap().expect("a page");
    assert_eq!(retried.page, 2);
    let reread = read(2, PageReading::Text("middle".into())).await.unwrap();
    let [middle] = reread[..] else {
        panic!("only the page read again is cut, got {reread:?}");
    };
    assert_eq!(found("middle").await, [(middle, 2)]);
    assert_eq!(found("cover").await, [(cover, 0)]);
}

#[tokio::test]
async fn a_file_whose_stamp_changes_drops_its_waiting_pages_and_its_transcriptions() {
    let (_handle, store, _temp) = fresh().await;
    let path = PathBuf::from("/notes/scan.pdf");
    let pass = store.begin_index_pass(Path::new("/notes")).await.unwrap();
    let index = |stamp: FileStamp| {
        let blank = vec![String::new(), String::new()];
        store.index_document(
            pass,
            path.clone(),
            stamp,
            blank,
            EmptyPages::AwaitTranscription,
        )
    };
    let waiting = async || store.count_transcriptions().await.unwrap().waiting;
    let read_next = async |text: &str| {
        let page = store.next_waiting_page().await.unwrap().expect("a page");
        let reading = PageReading::Text(text.into());
        (
            page.clone(),
            store.store_page_reading(page, reading).await.unwrap(),
        )
    };
    let touched = |mtime_ns| FileStamp { mtime_ns, ..STAMP };

    index(STAMP).await.unwrap();
    let (old_page, _) = read_next("old cover").await;
    index(touched(2)).await.unwrap();
    assert_eq!(waiting().await, 2, "the read page waits again");
    assert_eq!(page_texts(&store, &path, 2).await, ["", ""]);
    let stale = store
        .store_page_reading(old_page, PageReading::Text("old cover".into()))
        .await;
    assert!(stale.unwrap().is_empty());
    assert_eq!(
        (waiting().await, page_texts(&store, &path, 1).await),
        (2, vec![String::new()]),
        "a reading of the file as it was is not stored"
    );

    read_next("cover").await;
    let (_, cut) = read_next("back").await;
    assert_eq!(cut.len(), 2);
    index(touched(3)).await.unwrap();
    assert_eq!(waiting().await, 2);
    assert_eq!(page_texts(&store, &path, 2).await, ["", ""]);
    assert_eq!(store.count_document_chunks().await.unwrap(), 0);
}

#[tokio::test]
async fn a_query_is_matched_as_words_and_never_read_as_fts5_syntax() {
    let (_handle, store, _temp) = fresh().await;
    let pass = store.begin_index_pass(Path::new("/notes")).await.unwrap();
    let text = "pass --remap-path-prefix to rustc, then call read_regular_file";
    let path = PathBuf::from("/notes/build.md");
    let stored = store.index_document(pass, path, STAMP, one_page(text), EmptyPages::Final);
    let chunk = stored.await.unwrap()[0];
    let found = async |query: &str| {
        let hits = store.search_documents(query.to_string(), Vec::new(), 5);
        let hits = hits
            .await
            .unwrap_or_else(|error| panic!("{query:?} failed: {error}"));
        hits.iter().map(|hit| hit.chunk_id).collect::<Vec<_>>()
    };

    for literal in [
        "--remap-path-prefix",
        "\"read_regular_file\"",
        "READ_REGULAR_FILE",
        "rustc:",
        "(rustc",
        "prefix*",
    ] {
        assert_eq!(found(literal).await, [chunk], "{literal:?}");
    }
    for no_match in [
        "\"unterminated",
        "NEAR(",
        "nothing AND",
        "-x",
        "col:",
        "*",
        "\"\" --",
    ] {
        assert_eq!(found(no_match).await, [], "{no_match:?}");
    }
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
            message_chunks: 0,
            document_chunks: 0,
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
        .nearest_chunks(unit_vec(0.0), 2, SessionScope::All)
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
    let pass = store.begin_index_pass(Path::new("/notes")).await.unwrap();
    let document = store
        .index_document(
            pass,
            PathBuf::from("/notes/plan.md"),
            STAMP,
            one_page("draft"),
            EmptyPages::Final,
        )
        .await
        .unwrap();
    let (_, second) = transcript.append("z").await;
    let second_bare = second[0];

    let indexed_mem = save_memory(&handle, "indexed", "v1").await;
    embed_memory(&store, indexed_mem, "v1", &unit_vec(0.0)).await;
    let bare_mem = save_memory(&handle, "bare", "v2").await;

    assert_eq!(
        store
            .chunks_missing_embedding(ChunkSource::Message, None, 100)
            .await
            .unwrap(),
        [
            (first_bare, "y".to_string()),
            (second_bare, "z".to_string())
        ]
    );
    assert_eq!(
        store
            .chunks_missing_embedding(ChunkSource::Message, None, 1)
            .await
            .unwrap(),
        [(first_bare, "y".to_string())]
    );
    assert_eq!(
        store
            .chunks_missing_embedding(ChunkSource::Message, Some(first_bare), 100)
            .await
            .unwrap(),
        [(second_bare, "z".to_string())]
    );
    assert_eq!(
        store
            .chunks_missing_embedding(ChunkSource::Document, None, 100)
            .await
            .unwrap(),
        [(document[0], "draft".to_string())]
    );
    assert_eq!(
        store.memories_missing_embedding(None, 100).await.unwrap(),
        [(bare_mem, "v2".to_string())]
    );
    assert_eq!(
        store.count_missing().await.unwrap(),
        UnindexedRows {
            message_chunks: 2,
            document_chunks: 1,
            memories: 1
        }
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
        (EmbeddingTarget::ChunkText(ChunkSource::Message), "y"),
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
            .nearest_chunks(unit_vec(0.0), 5, SessionScope::All)
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
        store
            .chunks_missing_embedding(ChunkSource::Message, None, 100)
            .await
            .unwrap(),
        [(chunk, "y".to_string())]
    );
    assert_eq!(
        store.count_missing().await.unwrap(),
        UnindexedRows {
            message_chunks: 1,
            document_chunks: 0,
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
        document_prefix: "search_document: ".to_string(),
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
            message_chunks: 1,
            document_chunks: 0,
            memories: 1
        }
    );
    let all_messages = SessionScope::All;
    let hits = store.nearest_chunks(unit_vec(0.0), 5, all_messages).await;
    assert!(hits.unwrap().is_empty());
    assert!(
        store
            .nearest_memories(unit_vec(0.0), 5)
            .await
            .unwrap()
            .is_empty()
    );

    embed_text(&store, ChunkSource::Message, "text", &unit_vec(0.0)).await;
    let hits = store.nearest_chunks(unit_vec(0.0), 5, all_messages).await;
    assert_eq!(hits.unwrap()[0].chunk_id, chunk);
    assert_eq!(
        store.adopt_embedding_space(&rearranged).await.unwrap(),
        None
    );
}

#[tokio::test]
async fn adopting_a_space_that_differs_only_in_its_document_prefix_drops_every_vector() {
    let (handle, store, _temp) = fresh().await;
    let transcript = Transcript::new(&handle).await;
    let unprefixed = EmbeddingSpace {
        model: "owner/repo:Q8_0".to_string(),
        dim: 2,
        server_args: String::new(),
        document_prefix: String::new(),
    };
    assert_eq!(
        store.adopt_embedding_space(&unprefixed).await.unwrap(),
        None
    );
    transcript
        .embedded_chunk(&store, "text", &unit_vec(0.0))
        .await;

    let prefixed = EmbeddingSpace {
        document_prefix: "search_document: ".to_string(),
        ..unprefixed.clone()
    };
    assert_eq!(
        store.adopt_embedding_space(&prefixed).await.unwrap(),
        Some(unprefixed)
    );
    assert_eq!(
        store.count_missing().await.unwrap(),
        UnindexedRows {
            message_chunks: 1,
            document_chunks: 0,
            memories: 0
        }
    );
}

#[tokio::test]
async fn batch_skips_targets_deleted_before_it_lands() {
    let (handle, store, _temp) = fresh().await;
    let kept = save_memory(&handle, "kept", "v").await;
    let deleted = save_memory(&handle, "deleted", "v").await;
    MemoryStore::new(handle.clone())
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

#[tokio::test]
async fn a_write_signals_the_embedding_worker_only_when_it_leaves_a_row_to_embed() {
    let (handle, store, _temp) = fresh().await;
    let rows_to_embed = store.rows_to_embed();
    let memories = MemoryStore::new(handle.clone());
    let transcript = Transcript::new(&handle).await;
    assert!(!signalled(&rows_to_embed), "a session with no message");

    transcript.append("said").await;
    assert!(signalled(&rows_to_embed), "a message cut into a chunk");
    let uncut = SqliteTranscriptStore::new(handle.clone()).without_message_chunks();
    let stored_uncut = uncut.begin_turn(transcript.branch, PersistedMessage::user("uncut"));
    stored_uncut.await.unwrap();
    assert!(
        !signalled(&rows_to_embed),
        "a message stored without chunks"
    );
    assert_eq!(
        transcript.store.chunk_unchunked_messages().await.unwrap(),
        1
    );
    assert!(signalled(&rows_to_embed), "a stored message cut later");
    assert_eq!(
        transcript.store.chunk_unchunked_messages().await.unwrap(),
        0
    );
    assert!(!signalled(&rows_to_embed), "no message left to cut");

    let memory = memories.save("k", "v".into()).await.unwrap().id;
    assert!(signalled(&rows_to_embed), "a new memory");
    memories.save("k", "v".into()).await.unwrap();
    assert!(
        !signalled(&rows_to_embed),
        "a memory saved with the value it had"
    );
    memories.save("k", "changed".into()).await.unwrap();
    assert!(signalled(&rows_to_embed), "a memory saved with a new value");

    let pass = store.begin_index_pass(Path::new("/notes")).await.unwrap();
    let index = || {
        let path = PathBuf::from("/notes/a.md");
        store.index_document(pass, path, STAMP, one_page("words"), EmptyPages::Final)
    };
    index().await.unwrap();
    assert!(signalled(&rows_to_embed), "a document stored with a chunk");
    index().await.unwrap();
    assert!(!signalled(&rows_to_embed), "a document stored unchanged");

    let refused = Embedding {
        target: EmbeddingTarget::Memory(memory),
        text: "changed".to_string(),
        vector: None,
    };
    store.store_embeddings(vec![refused]).await.unwrap();
    assert!(!signalled(&rows_to_embed), "a row marked unembeddable");
    assert_eq!(store.clear_unembeddable().await.unwrap(), 1);
    assert!(signalled(&rows_to_embed), "a refusal mark cleared");
    assert_eq!(store.clear_unembeddable().await.unwrap(), 0);
    assert!(!signalled(&rows_to_embed), "no refusal mark to clear");

    let space = EmbeddingSpace {
        model: "owner/repo:Q8_0".to_string(),
        dim: 2,
        server_args: String::new(),
        document_prefix: String::new(),
    };
    store.adopt_embedding_space(&space).await.unwrap();
    assert!(!signalled(&rows_to_embed), "the first space recorded");
    let other_space = EmbeddingSpace { dim: 3, ..space };
    store.adopt_embedding_space(&other_space).await.unwrap();
    assert!(
        signalled(&rows_to_embed),
        "the vectors dropped for a new space"
    );
    store.adopt_embedding_space(&other_space).await.unwrap();
    assert!(
        !signalled(&rows_to_embed),
        "the recorded space adopted again"
    );
}

#[tokio::test]
async fn a_write_signals_page_transcription_only_when_it_leaves_a_page_waiting() {
    let (_handle, store, _temp) = fresh().await;
    let rows_to_embed = store.rows_to_embed();
    let pages_to_transcribe = store.pages_to_transcribe();
    let pass = store.begin_index_pass(Path::new("/notes")).await.unwrap();
    let index = |name: &str, pages: &[&str], empty_pages| {
        let path = Path::new("/notes").join(name);
        let pages = pages.iter().map(|page| (*page).to_string()).collect();
        store.index_document(pass, path, STAMP, pages, empty_pages)
    };

    let typed = index("typed.pdf", &["typed"], EmptyPages::AwaitTranscription);
    typed.await.unwrap();
    assert!(!signalled(&pages_to_transcribe), "every page has text");
    let finished = index("slides.odp", &["", "shown"], EmptyPages::Final);
    finished.await.unwrap();
    assert!(
        !signalled(&pages_to_transcribe),
        "empty pages stored as final"
    );
    let scan = index(
        "scan.pdf",
        &["", "", "typed"],
        EmptyPages::AwaitTranscription,
    );
    scan.await.unwrap();
    assert!(
        signalled(&pages_to_transcribe),
        "a document with scanned pages"
    );
    assert!(signalled(&rows_to_embed));

    let first = store.next_waiting_page().await.unwrap().expect("a page");
    let read = |page: usize, reading: PageReading| {
        let waiting = WaitingPage {
            page,
            ..first.clone()
        };
        store.store_page_reading(waiting, reading)
    };
    read(0, PageReading::Text("scanned".into())).await.unwrap();
    assert!(
        !signalled(&rows_to_embed),
        "nothing is cut while a page waits"
    );
    read(1, PageReading::Failed).await.unwrap();
    assert!(signalled(&rows_to_embed), "the read page cut into a chunk");
    assert!(
        !signalled(&pages_to_transcribe),
        "readings leave no page waiting"
    );

    assert_eq!(store.retry_failed_transcriptions().await.unwrap(), 1);
    assert!(
        signalled(&pages_to_transcribe),
        "a failed page set to wait again"
    );
    assert_eq!(store.retry_failed_transcriptions().await.unwrap(), 0);
    assert!(!signalled(&pages_to_transcribe), "no failed page to retry");
}
