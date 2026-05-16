//! Per-turn transient context assembly: semantic recall, window snapshot,
//! and the rendering helpers that turn them into prompt fragments.

use super::AppState;
use anyhow::Result;

/// Truncate `s` to at most `max_chars` *characters* (not bytes), appending
/// an ellipsis when truncated. Walks `char_indices` so we never split a
/// multi-byte character. Used to keep injected context lines short.
fn truncate_for_context(s: &str, max_chars: usize) -> String {
    let total = s.chars().count();
    if total <= max_chars {
        return s.replace('\n', " ");
    }
    let cutoff = s
        .char_indices()
        .nth(max_chars)
        .map(|(b, _)| b)
        .unwrap_or(s.len());
    let mut head = s[..cutoff].replace('\n', " ");
    head.push('…');
    head
}

impl AppState {
    /// Embed the user query, find the top-K nearest conversation chunks,
    /// and return a "Relevant past context: …" block to inject as a
    /// transient system message. Returns `Ok(None)` when retrieval is a
    /// no-op (short query, NoEmbedder, no hits).
    ///
    /// Best-effort: errors propagate but the caller treats every failure
    /// (embedder down, dim mismatch, …) as "skip injection".
    pub(super) async fn build_semantic_context(&self, query: &str) -> Result<Option<String>> {
        if query.trim().chars().count() < 3 {
            return Ok(None);
        }
        let vec = self.memory.embedder.embed(query.to_string()).await?;
        let model = self.memory.embedder.model().to_string();
        if model.is_empty() {
            return Ok(None);
        }
        let top_k = self.memory.embedding_cfg.top_k as usize;
        let hits = self
            .memory
            .semantic
            .nearest_chunks(vec, top_k, &model)
            .await?;
        if hits.is_empty() {
            return Ok(None);
        }
        let mut block = String::from("Relevant past context:\n");
        for h in hits {
            let snippet = truncate_for_context(&h.content, 200);
            block.push_str(&format!(
                "- [{} {} sim={:.0}%] {}\n",
                h.timestamp,
                h.role.as_wire(),
                h.similarity * 100.0,
                snippet
            ));
        }
        Ok(Some(block))
    }

    /// Format the focused-window snapshot as a `Current desktop context`
    /// block for the LLM's per-turn transient system message. Returns
    /// `None` when the window manager has no opinion (no compositor
    /// connected, nothing focused, all fields empty).
    ///
    /// Errors from the WM backend degrade silently to `None` so a flaky
    /// compositor never blocks the user's turn.
    pub(super) async fn build_window_context(&self) -> Option<String> {
        let ctx = match self.subsystems.window_manager.focused_context().await {
            Ok(Some(c)) => c,
            Ok(None) => return None,
            Err(e) => {
                tracing::debug!(
                    target: "assistd::context",
                    error = %e,
                    "WindowManager::focused_context failed; skipping window context",
                );
                return None;
            }
        };
        format_window_context_block(&ctx)
    }
}

/// Render a [`assistd_wm::FocusedWindowContext`] into the prompt
/// fragment injected as a transient system message. Pure (no async,
/// no `&self`) so it's directly unit-testable. Returns `None` when
/// every field is empty.
pub(super) fn format_window_context_block(
    ctx: &assistd_wm::FocusedWindowContext,
) -> Option<String> {
    if ctx.class.is_none() && ctx.title.is_none() && ctx.workspace.is_none() {
        return None;
    }
    let mut block = String::from("Current desktop context:\n");
    let class_for_line = ctx.class.as_deref();
    let title_for_line = ctx.title.as_deref();
    if class_for_line.is_some() || title_for_line.is_some() {
        match (class_for_line, title_for_line) {
            (Some(c), Some(t)) => block.push_str(&format!("- Focused window: {c} - {t}\n")),
            (Some(c), None) => block.push_str(&format!("- Focused window: {c}\n")),
            (None, Some(t)) => block.push_str(&format!("- Focused window: (unknown) - {t}\n")),
            (None, None) => unreachable!(),
        }
    }
    if let Some(ws) = ctx.workspace.as_deref() {
        block.push_str(&format!("- Workspace: {ws}\n"));
    }
    let is_term = ctx
        .class
        .as_deref()
        .map(assistd_wm::is_terminal_class)
        .unwrap_or(false);
    let kind = if is_term { "terminal" } else { "non-terminal" };
    block.push_str(&format!("The user is interacting with a {kind} window."));
    if is_term {
        block.push_str(
            " If the user asks to run a command, build, or test, prefer calling `run` \
             with `command: \"bash\"` (executing the command in this terminal context) over \
             launching a new terminal via `run` with `command: \"wm\"`.",
        );
    }
    Some(block)
}

/// Concatenate the semantic and window context blocks with a blank
/// line between them. Returns `None` only when both inputs are `None`.
/// Pure / unit-testable.
pub(super) fn combine_context_blocks(
    semantic: Option<String>,
    window: Option<String>,
) -> Option<String> {
    match (semantic, window) {
        (None, None) => None,
        (Some(s), None) => Some(s),
        (None, Some(w)) => Some(w),
        (Some(s), Some(w)) => Some(format!("{}\n{}", s.trim_end(), w)),
    }
}
