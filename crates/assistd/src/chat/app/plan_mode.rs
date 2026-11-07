//! Plan mode: messages run planning turns in the conversation, and once a plan is written the
//! chat offers to carry it out in the permission mode the user picks.

use assistd_ipc::{PermissionMode, Request};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use uuid::Uuid;

use super::App;

/// What the chat sends to carry out the plan it was just given.
const EXECUTE_PLAN_TEXT: &str = "Go ahead with the plan.";

/// What this chat's messages do; Shift+Tab cycles it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::chat) enum ChatMode {
    /// Run agent turns whose prompts are answered under the mode.
    Act(PermissionMode),
    /// Run planning turns, which change nothing; `offer` is the execute choice's selected row
    /// while it is open.
    Plan { offer: Option<usize> },
}

/// An answer the execute choice offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::chat) enum ExecuteChoice {
    Execute(PermissionMode),
    /// Close the choice, so a typed message revises the plan.
    KeepPlanning,
}

impl App {
    /// The execute choice's rows in display order and the selected one, while it is open.
    pub(in crate::chat) fn execute_choices(&self) -> Option<(Vec<ExecuteChoice>, usize)> {
        let ChatMode::Plan {
            offer: Some(selected),
        } = self.mode
        else {
            return None;
        };
        let mut choices = vec![
            ExecuteChoice::Execute(PermissionMode::Manual),
            ExecuteChoice::Execute(PermissionMode::AcceptEdits),
        ];
        if self.auto_mode_offered {
            choices.push(ExecuteChoice::Execute(PermissionMode::Auto));
        }
        choices.push(ExecuteChoice::KeepPlanning);
        Some((choices, selected))
    }

    /// Cycle manual, accept-edits, auto when offered, then plan mode, applying an acting mode
    /// to the open turn too.
    pub(super) fn toggle_mode(&mut self) {
        let next = match self.mode {
            ChatMode::Act(PermissionMode::Manual) => ChatMode::Act(PermissionMode::AcceptEdits),
            ChatMode::Act(PermissionMode::AcceptEdits) if self.auto_mode_offered => {
                ChatMode::Act(PermissionMode::Auto)
            }
            ChatMode::Act(_) => ChatMode::Plan { offer: None },
            ChatMode::Plan { .. } => ChatMode::Act(PermissionMode::Manual),
        };
        self.set_mode(next);
    }

    /// Record whether the daemon honors Auto, leaving Auto for Manual when it no longer does.
    pub(super) fn offer_auto_mode(&mut self, offered: bool) {
        self.auto_mode_offered = offered;
        match self.mode {
            ChatMode::Act(PermissionMode::Auto) if !offered => {
                self.set_mode(ChatMode::Act(PermissionMode::Manual));
            }
            ChatMode::Plan { offer: Some(_) } if !offered => {
                self.mode = ChatMode::Plan { offer: None };
            }
            ChatMode::Act(_) | ChatMode::Plan { .. } => {}
        }
    }

    /// Switch to `mode`, applying an acting mode to the open turn too.
    fn set_mode(&mut self, mode: ChatMode) {
        self.mode = mode;
        let ChatMode::Act(mode) = mode else {
            return;
        };
        if let Some(writer) = self
            .active_reply
            .as_ref()
            .and_then(|reply| reply.writer.clone())
        {
            let req = Request::SetPermissionMode {
                id: Uuid::new_v4().to_string(),
                mode,
            };
            self.tasks.spawn(async move {
                if let Err(e) = writer.send(req).await {
                    tracing::warn!("SetPermissionMode send failed: {e}");
                }
            });
        }
    }

    /// Offer to carry out the plan this chat's planning turn just wrote, if still in plan mode.
    pub(super) fn offer_execution(&mut self) {
        if let ChatMode::Plan { offer } = &mut self.mode {
            *offer = Some(0);
        }
    }

    /// Arrows move the selection, Enter or a row's number answers, and Esc keeps planning.
    pub(super) fn on_execute_choice_key(&mut self, ev: KeyEvent) {
        let Some((choices, selected)) = self.execute_choices() else {
            return;
        };
        let answer = match ev.code {
            KeyCode::Esc => Some(ExecuteChoice::KeepPlanning),
            KeyCode::Char('c') if ev.modifiers.contains(KeyModifiers::CONTROL) => {
                Some(ExecuteChoice::KeepPlanning)
            }
            KeyCode::Enter => choices.get(selected).copied(),
            KeyCode::Char(digit) => numbered_choice(&choices, digit),
            KeyCode::Up => {
                self.mode = ChatMode::Plan {
                    offer: Some(selected.saturating_sub(1)),
                };
                None
            }
            KeyCode::Down => {
                self.mode = ChatMode::Plan {
                    offer: Some((selected + 1).min(choices.len() - 1)),
                };
                None
            }
            _ => None,
        };
        match answer {
            Some(ExecuteChoice::Execute(mode)) => self.execute_plan(mode),
            Some(ExecuteChoice::KeepPlanning) => self.mode = ChatMode::Plan { offer: None },
            None => {}
        }
    }

    /// Switch to `mode` and ask the agent to carry out the plan it just wrote.
    fn execute_plan(&mut self, mode: PermissionMode) {
        self.set_mode(ChatMode::Act(mode));
        self.begin_turn(EXECUTE_PLAN_TEXT, &[]);
        self.spawn_query(EXECUTE_PLAN_TEXT.to_string(), Vec::new());
    }
}

/// The choice shown as `digit`, counting from 1.
fn numbered_choice(choices: &[ExecuteChoice], digit: char) -> Option<ExecuteChoice> {
    let number = usize::try_from(digit.to_digit(10)?).ok()?;
    choices.get(number.checked_sub(1)?).copied()
}
