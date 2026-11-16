//! A list's selected index and the window of entries drawn around it.

use std::ops::Range;

/// Index of a list's selected entry, moved one entry at a time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Selection(usize);

impl Selection {
    pub(crate) fn at(index: usize) -> Self {
        Self(index)
    }

    pub(crate) fn index(self) -> usize {
        self.0
    }

    /// Selects the entry before, staying on the first.
    pub(crate) fn previous(&mut self) {
        self.0 = self.0.saturating_sub(1);
    }

    /// Selects the entry after, staying on the last of `len`.
    pub(crate) fn next(&mut self, len: usize) {
        self.0 += 1;
        self.clamp(len);
    }

    /// Moves a selection past the end of `len` entries back to the last.
    pub(crate) fn clamp(&mut self, len: usize) {
        self.0 = self.0.min(len.saturating_sub(1));
    }
}

/// The entries of a `total`-entry list to draw in `visible` rows: a window
/// that keeps `selected` in view with `margin` entries below it, or as many
/// as half the window allows.
pub(crate) fn list_window(
    selected: usize,
    total: usize,
    visible: usize,
    margin: usize,
) -> Range<usize> {
    if total <= visible {
        return 0..total;
    }
    let margin = margin.min(visible.saturating_sub(1) / 2);
    let start = (selected + 1 + margin)
        .saturating_sub(visible)
        .min(total - visible);
    start..start + visible
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_follows_the_selection_and_stops_at_the_ends() {
        assert_eq!(list_window(0, 3, 5, 0), 0..3);
        assert_eq!(list_window(4, 10, 5, 0), 0..5);
        assert_eq!(list_window(5, 10, 5, 0), 1..6);
        assert_eq!(list_window(4, 10, 5, 2), 2..7);
        assert_eq!(list_window(9, 10, 5, 2), 5..10);
        assert_eq!(list_window(3, 10, 0, 0), 4..4);
    }
}
