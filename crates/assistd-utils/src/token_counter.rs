//! Per-turn token count from a turn's streamed events. llama.cpp emits roughly
//! one SSE chunk per token, so streamed chunks approximate generated tokens.

/// The daemon's latest conversation sizes and the chunk count when they arrived.
#[derive(Debug, Clone, Copy)]
struct ContextSnapshot {
    tokens: u64,
    total: u64,
    chunks_at: u64,
}

/// Tokens a turn has added to the conversation, and the conversation's size.
#[derive(Debug, Default, Clone, Copy)]
pub struct TokenCounter {
    chunk_count: u64,
    /// Turn tokens that never streamed as chunks, such as tool calls and their results.
    unstreamed: u64,
    context: Option<ContextSnapshot>,
}

impl TokenCounter {
    pub const fn new() -> Self {
        Self {
            chunk_count: 0,
            unstreamed: 0,
            context: None,
        }
    }

    /// Count one streamed text or reasoning chunk.
    pub fn on_delta(&mut self) {
        self.chunk_count += 1;
    }

    /// Tokens the turn has added to the conversation since the last reset.
    pub const fn tokens(&self) -> u64 {
        self.chunk_count + self.unstreamed
    }

    /// Record the conversation sizes the daemon sent with its latest request:
    /// what it holds now, and its `total` counting what compaction removed.
    /// Growth beyond the chunks streamed since the previous size counts toward
    /// the turn; shrinkage from compaction does not.
    pub fn on_context_size(&mut self, tokens: u32, total: u32) {
        if let Some(expected) = self.context_tokens() {
            self.unstreamed += u64::from(tokens).saturating_sub(expected);
        }
        self.context = Some(ContextSnapshot {
            tokens: u64::from(tokens),
            total: u64::from(total),
            chunks_at: self.chunk_count,
        });
    }

    /// Conversation tokens: the latest request's size plus chunks streamed since.
    pub fn context_tokens(&self) -> Option<u64> {
        self.context.map(|c| c.tokens + self.streamed_since(c))
    }

    /// Tokens the conversation has held in all, compacted ones included.
    pub fn total_tokens(&self) -> Option<u64> {
        self.context.map(|c| c.total + self.streamed_since(c))
    }

    const fn streamed_since(&self, context: ContextSnapshot) -> u64 {
        self.chunk_count - context.chunks_at
    }

    /// Start counting a new turn.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turn_counts_unstreamed_growth_but_not_compaction() {
        let mut m = TokenCounter::new();
        m.on_context_size(1000, 1000);
        m.on_delta();
        m.on_delta();
        m.on_context_size(1050, 1050);
        assert_eq!((m.tokens(), m.context_tokens()), (50, Some(1050)));

        m.on_context_size(400, 1050);
        m.on_delta();
        assert_eq!(
            (m.tokens(), m.context_tokens(), m.total_tokens()),
            (51, Some(401), Some(1051))
        );
    }
}
