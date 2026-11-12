//! The `/agents` panel: every sub-agent the transcript holds, and one's own
//! transcript.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::{App, Panel};
use crate::chat::output::{OutputPane, RelayStepTrace};

/// The `/agents` panel, drawn in place of the transcript.
#[derive(Debug, Clone, Copy)]
pub(in crate::chat) struct AgentPanel {
    /// Index into the output pane's agents, oldest first.
    pub selected: usize,
    /// The selected agent's transcript is shown instead of the list.
    pub open: bool,
}

impl App {
    /// Open the list with the newest sub-agent selected.
    pub(super) fn open_agent_panel(&mut self) {
        match self.output.agents().len() {
            0 => self.set_notice("no sub-agents yet"),
            count => {
                self.panel = Some(Panel::Agents(AgentPanel {
                    selected: count - 1,
                    open: false,
                }));
            }
        }
    }

    /// Close the panel when it is the `/agents` one, whose indices a changed
    /// transcript no longer matches.
    pub(super) fn close_agent_panel(&mut self) {
        self.panel
            .take_if(|panel| matches!(panel, Panel::Agents(_)));
    }

    fn shown_agent_panel(&self) -> Option<AgentPanel> {
        match self.shown_panel() {
            Some(Panel::Agents(panel)) => Some(*panel),
            _ => None,
        }
    }

    /// The sub-agent whose transcript is on screen.
    pub(in crate::chat) fn viewed_agent_mut(&mut self) -> Option<&mut RelayStepTrace> {
        let panel = self.shown_agent_panel().filter(|panel| panel.open)?;
        self.output.agent_mut(panel.selected)
    }

    fn viewed_agent(&self) -> Option<&RelayStepTrace> {
        let panel = self.shown_agent_panel().filter(|panel| panel.open)?;
        self.output.agents().get(panel.selected).copied()
    }

    /// Whether a thinking block, compaction line or refining status on screen is animating.
    pub(in crate::chat) fn glyph_live(&self) -> bool {
        self.output.glyph_live()
            || self
                .viewed_agent()
                .is_some_and(|agent| agent.transcript().glyph_live())
            || matches!(&self.panel, Some(Panel::Tasks(panel)) if panel.refining())
    }

    pub(in crate::chat) fn advance_live_glyphs(&mut self) {
        self.output.advance_live_glyphs();
        if let Some(agent) = self.viewed_agent_mut() {
            agent.transcript_mut().advance_live_glyphs();
        }
        if let Some(Panel::Tasks(panel)) = self.panel.as_mut() {
            panel.advance_refine_dots();
        }
    }

    /// Run `scroll` on the pane on screen: the open sub-agent's transcript,
    /// else the chat's.
    pub(super) fn scroll_shown_pane(&mut self, scroll: impl FnOnce(&mut OutputPane)) {
        if let Some(agent) = self.viewed_agent_mut() {
            scroll(agent.transcript_mut());
        } else {
            scroll(&mut self.output);
        }
    }

    /// Ctrl+C closes the panel and Ctrl+O toggles verbose mode from either view. In the
    /// list, arrows select, Enter opens the transcript and Esc closes; in a transcript,
    /// Esc or Left returns to the list and the rest scroll or expand it.
    pub(super) fn on_agent_panel_key(&mut self, ev: KeyEvent) {
        let Some(Panel::Agents(panel)) = self.panel.as_mut() else {
            return;
        };
        let control = ev.modifiers.contains(KeyModifiers::CONTROL);
        match ev.code {
            KeyCode::Char('c') if control => self.panel = None,
            KeyCode::Char('o') if control => self.toggle_verbose(),
            KeyCode::Esc | KeyCode::Left if panel.open => panel.open = false,
            _ if panel.open => self.on_agent_transcript_key(ev),
            KeyCode::Esc => self.panel = None,
            KeyCode::Enter => panel.open = true,
            KeyCode::Up => panel.selected = panel.selected.saturating_sub(1),
            KeyCode::Down => {
                let last = self.output.agents().len().saturating_sub(1);
                panel.selected = (panel.selected + 1).min(last);
            }
            _ => {}
        }
    }

    fn on_agent_transcript_key(&mut self, ev: KeyEvent) {
        let height = self.last_output_height;
        self.scroll_shown_pane(|pane| match ev.code {
            KeyCode::PageUp => pane.scroll_page_up(height),
            KeyCode::PageDown => pane.scroll_page_down(height),
            KeyCode::Up => pane.scroll_lines_up(1),
            KeyCode::Down => pane.scroll_lines_down(1),
            KeyCode::End => pane.reset_scroll(),
            KeyCode::Tab => {
                pane.toggle_last_expandable();
            }
            _ => {}
        });
    }
}
