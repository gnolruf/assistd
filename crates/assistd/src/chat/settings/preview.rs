//! The Theme tab's preview of one theme field: sample transcript drawn by the
//! chat's own renderers, beside samples of the interface that use the field.

use std::time::Duration;

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::output::OutputPane;
use super::super::theme::Theme;
use super::panel::SAVED_THEMES_KEY;

const THEME_KEY_PREFIX: &str = "chat.theme.";
const THEMES_ROLE: &str = "themes";
const TITLE: &str = "Preview ";
const RULE: &str = "─";
const COLUMN_GAP: u16 = 2;
const TABLE: &str = "| Folder | Size |\n|---|---|\n| Videos | 4.2G |\n| Downloads | 1.1G |";
const LINK_REPLY: &str = "See [the guide](https://example.org).";

/// A titled rule, then the transcript and interface samples that use theme
/// setting `key`, side by side when there are both; nothing for other keys.
pub(crate) fn render_theme_preview(frame: &mut Frame<'_>, area: Rect, theme: &Theme, key: &str) {
    let [title_area, body_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).areas(area);
    frame.render_widget(Paragraph::new(title_line(area.width, theme)), title_area);
    let role = theme_role(key);
    let pane = sample_pane(role);
    let chrome_width = match pane {
        Some(_) => body_area.width.saturating_sub(COLUMN_GAP) / 2,
        None => body_area.width,
    };
    let chrome = sample_chrome(role, chrome_width, theme);
    match (pane, chrome.is_empty()) {
        (Some(mut pane), false) => {
            let [pane_area, chrome_area] = Layout::horizontal([Constraint::Ratio(1, 2); 2])
                .spacing(COLUMN_GAP)
                .areas(body_area);
            render_pane(frame, pane_area, &mut pane, theme);
            frame.render_widget(Paragraph::new(chrome), chrome_area);
        }
        (Some(mut pane), true) => render_pane(frame, body_area, &mut pane, theme),
        (None, _) => frame.render_widget(Paragraph::new(chrome), body_area),
    }
}

/// The role a Theme tab setting previews: a color's field name, or
/// `themes` for the theme list.
fn theme_role(key: &str) -> &str {
    if key == SAVED_THEMES_KEY {
        THEMES_ROLE
    } else {
        key.strip_prefix(THEME_KEY_PREFIX).unwrap_or_default()
    }
}

fn title_line(width: u16, theme: &Theme) -> Line<'static> {
    let rest = usize::from(width).saturating_sub(TITLE.len() + 3);
    Line::from(vec![
        Span::styled(format!("{RULE}{RULE} "), Style::default().fg(theme.border)),
        Span::styled(TITLE, Style::default().fg(theme.status_text)),
        Span::styled(RULE.repeat(rest), Style::default().fg(theme.border)),
    ])
}

fn render_pane(frame: &mut Frame<'_>, area: Rect, pane: &mut OutputPane, theme: &Theme) {
    let (lines, _) = pane.render_view(area.width, u16::MAX, theme);
    let shown: Vec<Line<'static>> = lines
        .iter()
        .take(usize::from(area.height))
        .cloned()
        .collect();
    frame.render_widget(Paragraph::new(shown), area);
}

/// Transcript items drawn in `role`'s color, or `None` when the transcript
/// never uses it.
fn sample_pane(role: &str) -> Option<OutputPane> {
    let mut pane = OutputPane::new();
    pane.set_verbose(true);
    match role {
        "accent" => {
            push_reply(&mut pane, "## Disk usage\n\n- Videos\n- Downloads");
            push_passed_tool(&mut pane);
            pane.push_info("model is drowsy");
        }
        "user_message" => pane.push_user("What's using my disk?"),
        "assistant_text" => push_reply(
            &mut pane,
            &format!("Your **Videos** folder takes the most space.\n\n{TABLE}"),
        ),
        "status_text" => {
            push_thinking(&mut pane);
            push_reply(&mut pane, LINK_REPLY);
            pane.push_turn_summary("Worked for 4s (312 tokens)".into());
            pane.begin_tool_block("ncdu ~/Videos".into());
        }
        THEMES_ROLE => {
            pane.push_user("What's using my disk?");
            push_thinking(&mut pane);
            push_passed_tool(&mut pane);
            push_reply(
                &mut pane,
                &format!("## Disk usage\n\n**Videos** is largest; run `ncdu ~`.\n\n{TABLE}"),
            );
        }
        "code" => {
            push_reply(
                &mut pane,
                "Run `ncdu ~` to dig in:\n\n```\nncdu ~/Videos\n```",
            );
            push_passed_tool(&mut pane);
        }
        "border" => push_reply(&mut pane, &format!("{TABLE}\n\n---")),
        "success" => push_passed_tool(&mut pane),
        "error" => {
            pane.push_tool_block(
                "cat notes.txt".into(),
                "[stderr] cat: notes.txt: No such file".into(),
                1,
                3,
            );
            pane.push_error("[error] daemon connection lost");
        }
        "link" => push_reply(&mut pane, LINK_REPLY),
        _ => return None,
    }
    Some(pane)
}

fn push_thinking(pane: &mut OutputPane) {
    pane.push_replayed_thinking(
        "Check the home folders first.".into(),
        Duration::from_secs(2),
    );
}

fn push_reply(pane: &mut OutputPane, markdown: &str) {
    pane.begin_assistant();
    pane.append_assistant(markdown);
    pane.finish_assistant();
}

fn push_passed_tool(pane: &mut OutputPane) {
    pane.push_tool_block(
        "du -sh ~/*".into(),
        "4.2G /home/me/Videos\n1.1G /home/me/Downloads".into(),
        0,
        120,
    );
}

/// Interface rows outside the transcript drawn in `role`'s color; empty
/// when none are.
fn sample_chrome(role: &str, width: u16, theme: &Theme) -> Vec<Line<'static>> {
    let accent = Style::default().fg(theme.accent);
    let muted = Style::default().fg(theme.status_text);
    match role {
        "accent" => vec![
            Line::styled("manual mode on (shift+tab)", accent),
            popup_entry(" /resume ", "pick a session", accent, muted),
        ],
        "status_text" => vec![
            status_row(theme),
            Line::styled("3h ago · 12 messages (current)", muted),
            relay_row("├─ ", "● researcher · Running…", muted, theme),
        ],
        "border" => border_rows(width, theme),
        "success" => outcome_rows(
            theme.success,
            "✓ writer · Done",
            &["✓ heard the wake word", "saved"],
            theme,
        ),
        "error" => outcome_rows(
            theme.error,
            "✗ writer · Failed",
            &["not applied: bad color"],
            theme,
        ),
        "warning" => warning_rows(theme),
        THEMES_ROLE => {
            let mut rows = border_rows(width, theme);
            rows.push(Line::default());
            rows.extend(sample_chrome("selection", width, theme));
            rows.extend(warning_rows(theme));
            rows
        }
        "selection" => {
            let selected = theme.selection();
            vec![
                popup_entry(
                    " /settings ",
                    "edit the config",
                    selected.add_modifier(Modifier::BOLD),
                    selected,
                ),
                popup_entry(" /resume ", "pick a session", accent, muted),
            ]
        }
        _ => Vec::new(),
    }
}

fn popup_entry(command: &str, hint: &str, style: Style, hint_style: Style) -> Line<'static> {
    Line::from(vec![
        Span::styled(command.to_string(), style),
        Span::styled(hint.to_string(), hint_style),
    ])
}

fn relay_row(branch: &str, text: &str, style: Style, theme: &Theme) -> Line<'static> {
    Line::from(vec![
        Span::styled(branch.to_string(), Style::default().fg(theme.border)),
        Span::styled(text.to_string(), style),
    ])
}

/// The input between its rules, a relay's tree and the status line's separators.
fn border_rows(width: u16, theme: &Theme) -> Vec<Line<'static>> {
    let rule = Line::styled(
        RULE.repeat(usize::from(width)),
        Style::default().fg(theme.border),
    );
    vec![
        rule.clone(),
        Line::raw("> type a message"),
        rule,
        Line::default(),
        relay_row("├─ ", "● researcher", Style::default(), theme),
        relay_row("└─ ", "✓ writer", Style::default(), theme),
        Line::default(),
        status_row(theme),
    ]
}

/// A finished sub-agent's row, then `others`, all in `color`.
fn outcome_rows(color: Color, step: &str, others: &[&str], theme: &Theme) -> Vec<Line<'static>> {
    let style = Style::default().fg(color);
    let mut rows = vec![relay_row("└─ ", step, style, theme)];
    rows.extend(
        others
            .iter()
            .map(|text| Line::styled((*text).to_string(), style)),
    );
    rows
}

fn warning_rows(theme: &Theme) -> Vec<Line<'static>> {
    let warning = Style::default().fg(theme.warning);
    vec![
        Line::styled(
            "Confirm bash: rm -rf build/",
            warning.add_modifier(Modifier::BOLD),
        ),
        Line::styled("· unsaved changes", warning),
        Line::styled("! chat.temperature must be in the range 0.0..=2.0", warning),
    ]
}

fn status_row(theme: &Theme) -> Line<'static> {
    let status = Style::default().fg(theme.status_text);
    Line::from(vec![
        Span::styled("Listening…", status),
        Span::styled(" · ", Style::default().fg(theme.border)),
        Span::styled("Esc to interrupt", status),
    ])
}

#[cfg(test)]
mod tests;
