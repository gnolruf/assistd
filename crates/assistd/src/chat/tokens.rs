//! Per-turn token count. llama.cpp emits roughly one SSE chunk per token,
//! so streamed chunks approximate generated tokens.

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct TokenCounter {
    chunk_count: u64,
    /// Turn tokens that never streamed as chunks, such as tool calls and their results.
    unstreamed: u64,
    /// The daemon's latest conversation size and the chunk count when it arrived.
    context: Option<(u64, u64)>,
}

impl TokenCounter {
    pub(super) const fn new() -> Self {
        Self {
            chunk_count: 0,
            unstreamed: 0,
            context: None,
        }
    }

    pub(super) fn on_delta(&mut self) {
        self.chunk_count += 1;
    }

    /// Tokens the turn has added to the conversation since the last reset.
    pub(super) const fn tokens(&self) -> u64 {
        self.chunk_count + self.unstreamed
    }

    /// Record the conversation size the daemon sent with its latest request.
    /// Growth beyond the chunks streamed since the previous size counts toward
    /// the turn; shrinkage from compaction does not.
    pub(super) fn on_context_size(&mut self, tokens: u32) {
        if let Some(expected) = self.context_tokens() {
            self.unstreamed += u64::from(tokens).saturating_sub(expected);
        }
        self.context = Some((u64::from(tokens), self.chunk_count));
    }

    /// Conversation tokens: the latest request's size plus chunks streamed since.
    pub(super) fn context_tokens(&self) -> Option<u64> {
        self.context
            .map(|(tokens, chunks_at)| tokens + self.chunk_count - chunks_at)
    }

    pub(super) fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turn_counts_unstreamed_growth_but_not_compaction() {
        let mut m = TokenCounter::new();
        m.on_context_size(1000);
        m.on_delta();
        m.on_delta();
        m.on_context_size(1050);
        assert_eq!((m.tokens(), m.context_tokens()), (50, Some(1050)));

        m.on_context_size(400);
        m.on_delta();
        assert_eq!((m.tokens(), m.context_tokens()), (51, Some(401)));
    }
}
